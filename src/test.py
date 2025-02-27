import json
import logging
import torch
import argparse
from transformers import AutoModel, AutoTokenizer
import context_pruning
from tqdm import tqdm

# Configure argument parsing
parser = argparse.ArgumentParser(description="Process queries and prune contexts based on similarity.")
parser.add_argument("-i", "--input_json", type=str, required=True, help="Path to input JSON file.")
parser.add_argument("-o", "--output_json", type=str, required=True, help="Path to output JSON file.")
parser.add_argument("-l", "--log_file", type=str, default="process.log", help="Log file name (default: process.log)")
args = parser.parse_args()

# Configure logging
logging.basicConfig(
    filename=args.log_file,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Load the models globally (avoid reloading in functions)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
QUERY_ENCODER_NAME = "ncbi/MedCPT-Query-Encoder"
ARTICLE_ENCODER_NAME = "ncbi/MedCPT-Article-Encoder"
logging.info(f"Loading models {QUERY_ENCODER_NAME} and {ARTICLE_ENCODER_NAME} on {DEVICE}")

tokenizer_query = AutoTokenizer.from_pretrained(QUERY_ENCODER_NAME)
tokenizer_article = AutoTokenizer.from_pretrained(ARTICLE_ENCODER_NAME)
model_query = AutoModel.from_pretrained(QUERY_ENCODER_NAME).to(DEVICE)
model_article = AutoModel.from_pretrained(ARTICLE_ENCODER_NAME).to(DEVICE)
model_query.eval()  # Set model to evaluation mode
model_article.eval()


def json_loader(filepath):
    """ Load JSON file safely """
    try:
        with open(filepath, "r", encoding="utf-8") as file:
            return json.load(file)
    except Exception as e:
        logging.error(f"Error loading JSON file {filepath}: {str(e)}")
        return None


def load_pubmed_data(pubmed_json_path):
    """ Load PubMed abstracts once to avoid repeated file access """
    try:
        with open(pubmed_json_path, "r", encoding="utf-8") as file:
            return json.load(file)
    except Exception as e:
        logging.error(f"Error loading PubMed abstracts from {pubmed_json_path}: {str(e)}")
        return {}


def fetch_pubmed_abstract(pmid, pubmed_data):
    """ Fetch abstract from pre-loaded PubMed dictionary """
    return pubmed_data.get(pmid, {}).get("a", "")  # Return empty string if not found


def encode_query(query):
    """ Encode a single query using query encoder """
    try:
        inputs = tokenizer_query(query, return_tensors="pt", truncation=True, padding=True, max_length=512).to(DEVICE)
        with torch.no_grad():
            output = model_query(**inputs).last_hidden_state[:, 0, :].cpu().numpy().flatten()
        return output
    except Exception as e:
        logging.error(f"Error encoding query: {query[:30]}... {str(e)}")
        return None


def encode_texts(texts):
    """ Encode multiple retrieved texts using article encoder in batch mode """
    try:
        inputs = tokenizer_article(texts, return_tensors="pt", truncation=True, padding=True, max_length=512).to(DEVICE)
        with torch.no_grad():
            output = model_article(**inputs).last_hidden_state[:, 0, :].cpu().numpy()
        return output
    except Exception as e:
        logging.error(f"Error encoding texts: {str(e)}")
        return None


def process_queries(input_file, output_file, pubmed_json_path):
    """
    Loads a JSON file, encodes queries & retrieved documents, performs similarity-based pruning,
    and stores pruned results (PMIDs) into an output JSON file.
    """
    data = json_loader(input_file)
    none_flag = 0
    if data is None:
        logging.error("Failed to load input JSON file. Exiting.")
        return

    logging.info(f"Loaded {len(data)} queries from {input_file}")

    # Load PubMed data **once** instead of inside the loop
    pubmed_data = load_pubmed_data(pubmed_json_path)

    pruned_results = []
    with tqdm(total=len(data), desc="Processing Queries", unit="query") as pbar:
        for entry in data:
            query = entry["query"]
            retrieved = entry["retrieved"]
            retrieved_pmids = [doc["pmid"] for doc in retrieved]

            logging.info(f"Processing query: {query[:30]}...")
            retrieved_texts = [fetch_pubmed_abstract(pmid, pubmed_data) for pmid in retrieved_pmids]

            if not retrieved_texts:
                logging.warning(f"No retrieved texts for query: {query[:30]}...")
                pbar.update(1)
                continue

            # Batch encode query and retrieved texts
            query_embedding = encode_query(query)
            snippet_embeddings = encode_texts(retrieved_texts)

            if query_embedding is None or snippet_embeddings is None:
                logging.warning(f"Skipping query due to encoding failure: {query[:30]}...")
                pbar.update(1)
                continue

            # Perform context pruning
            pruned_pmids, similarities, threshold = context_pruning.similarity_check(query_embedding, snippet_embeddings, retrieved_pmids)
            logging.info(f"Pruned {len(pruned_pmids)} out of {len(retrieved_pmids)} PMIDs")
            logging.info(f"Similarities: {similarities}")
            logging.info(f"Threshold: {threshold}")
            if len(pruned_pmids) == 0:
                none_flag += 1
                logging.info(f"None flag: {none_flag}")

            pruned_results.append({"query": query, "pruned_pmids": pruned_pmids})
            pbar.update(1)

    # Save results to output JSON file
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(pruned_results, f, indent=4)
    logging.info(f"Pruned contexts saved to {output_file}")
    print(f"Pruned contexts saved to {output_file}")
    logging.info(f"None flag: {none_flag}")


# Execute the script with provided arguments
logging.info("Starting query processing...")
PUBMED_JSON_PATH = "pubmed_chunk_36.json"  # Change this to your actual path
process_queries(args.input_json, args.output_json, PUBMED_JSON_PATH)
logging.info("Query processing complete.")
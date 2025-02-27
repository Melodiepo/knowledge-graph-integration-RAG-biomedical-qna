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
    """ Load JSON file """
    try:
        with open(filepath, "r", encoding="utf-8") as file:
            return json.load(file)
    except Exception as e:
        logging.error(f"Error loading JSON file {filepath}: {str(e)}")
        return None


def fetch_pubmed_abstract(pmid):
    """ Fetch abstract from local PubMed json file"""
    try:
        with open("pubmed_chunk_36.json", "r", encoding="utf-8") as file:
            data = json.load(file)
        return data.get(pmid, {}).get("a", "")  # Return empty string if not found
    except Exception as e:
        logging.error(f"Error fetching PubMed abstract for PMID {pmid}: {str(e)}")
        return ""


def encode_query(query):
    """ Encode a single query using query encoder """
    try:
        inputs = tokenizer_query(query, return_tensors="pt", truncation=True, padding=True, max_length=512).to(DEVICE)
        with torch.no_grad():
            output = model_query(**inputs).last_hidden_state[:, 0, :].cpu().numpy().flatten()
        return output
    except Exception as e:
        logging.error(f"Error encoding query: {query}: {str(e)}")
        return None


def encode_texts(texts):
    """ Encode multiple retrieved texts using article encoder """
    try:
        inputs = tokenizer_article(texts, return_tensors="pt", truncation=True, padding=True, max_length=512).to(DEVICE)
        with torch.no_grad():
            output = model_article(**inputs).last_hidden_state[:, 0, :].cpu().numpy()
        return output
    except Exception as e:
        logging.error(f"Error encoding texts: {str(e)}")
        return None


def test_query_preprocess_instruction(input_file, output_file):
    """
    Loads a JSON file, encodes queries & retrieved documents, performs similarity-based pruning,
    and stores pruned results (PMIDs) into an output JSON file.
    """
    data = json_loader(input_file)
    if data is None:
        logging.error("Failed to load input JSON file. Exiting.")
        return
    logging.info(f"Loaded {len(data)} queries from {input_file}")
    pruned_results = []

    with tqdm(total=len(data), desc="Processing Queries", unit="query") as pbar:
        for entry in data:
            logging.info(f"Processing query: {entry['query']}")
            query = entry["query"]
            retrieved = entry["retrieved"]

            logging.info(f"Retrieved {len(retrieved)} documents for query")
            retrieved_pmids = [doc["pmid"] for doc in retrieved]
            logging.info(f"Retrieved PMIDs: {retrieved_pmids}")
            retrieved_texts = [fetch_pubmed_abstract(pmid) for pmid in retrieved_pmids]

            if not retrieved_texts:
                logging.warning(f"No retrieved texts for query: {query}")
                continue
            logging.info(f"Embedding {len(retrieved_texts)} retrieved texts...")
            query_embedding = encode_query(query)
            snippet_embeddings = encode_texts(retrieved_texts)

            if query_embedding is None or snippet_embeddings is None:
                logging.warning(f"Skipping query due to encoding failure: {query}")
                continue
            logging.info(f"Calculating similarity for query")

            pruned_pmids = context_pruning.similarity_check(query_embedding, snippet_embeddings, retrieved_pmids)
            pruned_results.append({"query": query, "pruned_pmids": pruned_pmids})
            logging.info(f"Pruned PMIDs: {pruned_pmids}")

            pbar.update(1)
            logging.info(f"Processed query")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(pruned_results, f, indent=4)

    logging.info(f"Pruned contexts saved to {output_file}")
    print(f"Pruned contexts saved to {output_file}")


# Execute the script with provided arguments
logging.info("Starting query processing...")
test_query_preprocess_instruction(args.input_json, args.output_json)

import json
import requests
import torch
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification
import context_pruning
from bs4 import BeautifulSoup

# Load the model globally (avoid reloading in functions)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "ncbi/MedCPT-Query-Encoder"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)


def json_loader(filepath):
    """ Load JSON file """
    with open(filepath, "r") as file:
        return json.load(file)


def fetch_pubmed_abstract(pmid):
    """ Fetch abstract from PubMed using scraping"""
    # TODO: Use metadata instead of scraping
    url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")

    abstract_section = soup.find("div", class_="abstract-content")
    return abstract_section.text.strip() if abstract_section else "No abstract found"


def encode_query(query):
    """ Encode a single query using preloaded model """
    inputs = tokenizer(query, return_tensors="pt", truncation=True, padding=True, max_length=512).to(DEVICE)
    with torch.no_grad():
        return model(**inputs).logits.cpu().numpy().flatten()


def encode_texts(texts):
    """ Encode multiple retrieved texts using preloaded model """
    inputs = tokenizer(texts, return_tensors="pt", truncation=True, padding=True, max_length=512).to(DEVICE)
    with torch.no_grad():
        return model(**inputs).logits.cpu().numpy()


def test_query_preprocess_instruction(input_file, output_file):
    """
    Loads a JSON file, encodes queries & retrieved documents, performs similarity-based pruning,
    and stores pruned results (PMIDs) into an output JSON file.
    """
    # Load the queries and retrieved documents
    data = json_loader(input_file)

    pruned_results = []

    for entry in data:
        query = entry["query"]
        retrieved = entry["retrieved"]

        # Extract PMIDs and fetch abstracts
        retrieved_pmids = [doc["pmid"] for doc in retrieved]
        retrieved_texts = [fetch_pubmed_abstract(pmid) for pmid in retrieved_pmids]

        # Encode query & retrieved documents
        query_embedding = encode_query(query)
        snippet_embeddings = encode_texts(retrieved_texts)

        # Prune context based on similarity
        pruned_pmids = context_pruning.similarity_check(query_embedding, snippet_embeddings, retrieved_pmids)

        # Store pruned results
        pruned_results.append({"query": query, "pruned_pmids": pruned_pmids})

    # Save pruned PMIDs to JSON
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(pruned_results, f, indent=4)

    print(f"Pruned contexts saved to {output_file}")



input_json = "evidence_medqa_llama_cot.json"
output_json = "pruned_contexts.json"

test_query_preprocess_instruction(input_json, output_json)

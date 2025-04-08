#!/usr/bin/env python

import os
import gc
import json
import random
import sys
from tqdm import tqdm

# Optional: attempt to import psutil for memory usage checking
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

def print_memory_usage(stage=""):
    """
    Print the current and peak memory usage in MB if psutil is available.
    Otherwise, do nothing.
    """
    if not HAS_PSUTIL:
        return  # Skip if psutil isn't installed
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    rss_mb = mem_info.rss / (1024 ** 2)
    print(f"[MEM USAGE] {stage} - RSS: {rss_mb:.2f} MB")

def load_pubmed_chunk(articles_dir, chunk_index):
    """
    Safely load a PubMed JSON file (PubMed_Articles_{chunk_index}.json) in binary mode,
    decoding with 'errors=replace' to avoid segfaults on invalid bytes.
    Returns a dict of pmid_str -> {"t": title, "a": abstract} or an empty dict on failure.
    """
    file_name = f"PubMed_Articles_{chunk_index}.json"
    file_path = os.path.join(articles_dir, file_name)

    if not os.path.exists(file_path):
        print(f"Warning: file missing: {file_name}")
        return {}

    try:
        # Read in binary mode
        with open(file_path, "rb") as f:
            raw_data = f.read()

        # Decode with 'replace' to avoid crashes on invalid chars
        text_data = raw_data.decode("utf-8", errors="replace")

        # Parse JSON
        articles_data = json.loads(text_data)
        return articles_data

    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return {}

def preprocess_labeled_data(
    labeled_data_path,
    articles_dir,
    output_path,
    max_queries=None,
    max_docs_per_query=None,
    seed=42
):
    """
    Preprocess a labeled dataset for training a sequence classification model. The dataset is expected to be in the
    format of the labeled_perplexity_data.json file output by generate_labels.py. The output is a JSON file containing
    a list of samples, each with a "text" field and a "label" field.

    Args:
        labeled_data_path (str): Path to the labeled data JSON file.
        articles_dir (str): Directory containing PubMed articles JSON files.
        output_path (str): Path to write the preprocessed data.
        max_queries (int, optional): Maximum number of queries to process. Defaults to None (no limit).
        max_docs_per_query (int, optional): Maximum number of documents per query to include. Defaults to None (no limit).
        seed (int, optional): Random seed for reproducibility. Defaults to 42.
    """

    # --- Load the main labeled data file ---
    with open(labeled_data_path, "r", encoding="utf-8") as f:
        labeled_data = json.load(f)

    # Convert to a list of entries (since keys might be "0", "1", etc.)
    entries = list(labeled_data.values())

    # Optionally limit the queries
    if max_queries is not None:
        entries = entries[:max_queries]

    # We'll store final pairs of (text, label)
    all_samples = []

    # For reproducibility
    random.seed(seed)

    pbar = tqdm(entries, desc="Preprocessing")
    for idx, entry in enumerate(pbar):
        query = entry["query"]
        docs = entry["retrieved_docs"]
        labels = entry["labels"]

        # Only include queries that have at least one positive label
        if 0 not in labels:
            continue

        # If limiting number of docs
        if max_docs_per_query is not None:
            docs = docs[:max_docs_per_query]
            labels = labels[:max_docs_per_query]

        # Cache chunk files for this single query
        chunk_file_cache = {}

        for doc, label in zip(docs, labels):
            doc_text = ""

            # If doc is a dict with "pmid" and "chunk_index"
            if isinstance(doc, dict) and "pmid" in doc and "chunk_index" in doc:
                pmid_str = str(doc["pmid"])
                chunk_index = doc["chunk_index"]

                # Load chunk file if not already cached
                if chunk_index not in chunk_file_cache:
                    chunk_file_cache[chunk_index] = load_pubmed_chunk(articles_dir, chunk_index)

                chunk_data = chunk_file_cache[chunk_index]
                doc_data = chunk_data.get(pmid_str, {})
                title = doc_data.get("t", "")
                abstract = doc_data.get("a", "")
                doc_text = f"{title}. {abstract}".strip()

            # If doc is a raw string
            elif isinstance(doc, str):
                doc_text = doc.strip()

            # Construct final text: "Query: ... Document: ..."
            if doc_text:
                combined_text = f"Query: {query} Document: {doc_text}"
            else:
                combined_text = f"Query: {query} Document: "

            # Add to the final dataset
            all_samples.append({
                "text": combined_text,
                "label": label
            })

        # Clear chunk cache after finishing this query to free memory
        chunk_file_cache.clear()
        gc.collect()

        # (Optional) Print memory usage after each query to see if it spikes
        if idx % 50 == 0:
            print_memory_usage(stage=f"After query #{idx}")

    # Done processing all queries
    print_memory_usage(stage="Final")

    # --- Write everything to a single JSON file ---
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as out_f:
        json.dump(all_samples, out_f, ensure_ascii=False, indent=2)

    print(f"Preprocessing complete. Wrote {len(all_samples)} samples to {output_path}")

if __name__ == "__main__":
    # Edit these paths as needed
    LABELED_DATA_PATH = "/cs/student/projects1/ml/2024/yihanli/uandreic/rag2-filtering/output/labeled_perplexity_data.json"
    ARTICLES_DIR = "/cs/student/projects1/ml/2024/yihanli/retriever/articles/pubmed/"
    OUTPUT_JSON = "/cs/student/projects1/ml/2024/yihanli/uandreic/rag2-filtering/output/training_data.json"

    # (Optional) subsetting
    with open(LABELED_DATA_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    MAX_QUERIES = int(len(dataset)*0.5)
    MAX_DOCS_PER_QUERY = 7


    preprocess_labeled_data(
        labeled_data_path=LABELED_DATA_PATH,
        articles_dir=ARTICLES_DIR,
        output_path=OUTPUT_JSON,
        max_queries=MAX_QUERIES,
        max_docs_per_query=MAX_DOCS_PER_QUERY
    )

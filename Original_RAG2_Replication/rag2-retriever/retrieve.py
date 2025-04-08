import json
import glob
import numpy as np
import torch
import os
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import re
import faiss

def pubmed_index_create(pubmed_embeddings_dir):
    index_path = os.path.join(pubmed_embeddings_dir, "faiss_index_pubmed")

    # Check if FAISS index already exist, if so load it directly
    if os.path.exists(index_path):
        print(f"Loading FAISS index from {index_path}")
        return faiss.read_index(index_path)
    
    print("Using IndexIVFPQ for memory-efficient FAISS indexing.")

    # We’ll use an IP quantizer for IVFPQ with inner-product metric
    quantizer = faiss.IndexFlatIP(768)
    
    # nlist = number of coarse clusters (tune based on your data size)
    # m = number of PQ segments (each segment is 8 bits => 1 byte)
    # Example: m=64 => each vector is compressed to 64 bytes
    nlist = 4096
    m = 64
    
    # This creates an IVF-PQ index for approximate nearest-neighbor with compression
    pubmed_index = faiss.IndexIVFPQ(
        quantizer,  # quantizer
        768,        # dimension
        nlist,      # nlist
        m,          # number of bytes (subvectors)
        8           # bits per code (8 = 1 byte)
    )
    pubmed_index.metric_type = faiss.METRIC_INNER_PRODUCT

    embedding_files = sorted(glob.glob(os.path.join(pubmed_embeddings_dir, "PubMed_Embeds_*.npy")))
    if not embedding_files:
        raise RuntimeError(f"No PubMed embedding files found in {pubmed_embeddings_dir}!")

    # 1) TRAIN on a sample chunk first (e.g., only the first file or part of it)
    print(f"Training IVFPQ index on {embedding_files[0]}")
    train_vectors = np.load(embedding_files[0]).astype(np.float32)
    sample_count = min(200000, train_vectors.shape[0])  # use up to 200k vectors for training
    pubmed_index.train(train_vectors[:sample_count])    # must train before adding

    # 2) ADD embeddings in smaller batches
    total_added = 0
    batch_size = 50000

    for file in embedding_files:
        print(f"Loading PubMed embeddings from: {file} in chunks...")
        chunk_embeds = np.load(file).astype(np.float32)
        for start in range(0, len(chunk_embeds), batch_size):
            pubmed_index.add(chunk_embeds[start:start + batch_size])
        total_added += chunk_embeds.shape[0]

    print(f"Total PubMed vectors added: {total_added}")

    # 3) Save index to disk
    print(f"Saving IVFPQ index to {index_path}")
    faiss.write_index(pubmed_index, index_path)

    return pubmed_index

def pmc_index_create(pmc_embeddings_dir):
    pmc_index = faiss.IndexFlatIP(768)
    pmc_filenames = ["PMC_Main_Embeds.npy", "PMC_Abs_Embeds.npy"]

    for embed_file in pmc_filenames:
        embed_path = os.path.join(pmc_embeddings_dir, embed_file)
        if not os.path.exists(embed_path):
            print(f"Warning: {embed_path} not found. Skipping...")
            continue
        
        embeds = np.load(embed_path).astype(np.float32)
        pmc_index.add(embeds)
        del embeds

    if pmc_index.ntotal == 0:
        print("Warning: No valid PMC embeddings were loaded into the index!")
    return pmc_index


def cpg_index_create(cpg_embeddings_dir):
    """
    Create a FAISS index with all CPG embeddings in cpg_embeddings_dir,
    e.g. CPG_128_embed_0.npy, ..., CPG_128_embed_3634.npy
    """
    cpg_index = faiss.IndexFlatIP(768)

    embedding_files = sorted(glob.glob(os.path.join(cpg_embeddings_dir, "CPG_128_embed_*.npy")))
    if not embedding_files:
        print(f"Warning: no chunked CPG embeddings found in {cpg_embeddings_dir}!")
        return cpg_index

    total_added = 0
    for file in embedding_files:
        print(f"Loading CPG embeddings from: {file}")
        chunk = np.load(file).astype(np.float32)
        cpg_index.add(chunk)
        total_added += chunk.shape[0]

    print(f"Total CPG vectors added to FAISS index: {total_added}")
    return cpg_index


def textbook_index_create(textbook_embeddings_dir):
    """
    Create a FAISS index with all Textbook embeddings, e.g. chunked .npy files.
    """
    textbook_index = faiss.IndexFlatIP(768)

    embedding_files = glob.glob(os.path.join(textbook_embeddings_dir, "*.npy"))
    if not embedding_files:
        print(f"Warning: no chunked Textbook embeddings found in {textbook_embeddings_dir}!")
        return textbook_index

    for file in embedding_files:
        embeds = np.load(file).astype(np.float32)
        textbook_index.add(embeds)
    print(f"Total Textbook vectors added: {textbook_index.ntotal}")
    return textbook_index


def find_value_by_index(articles, target_index):
    return articles[target_index]


def pubmed_decode(pubmed_I_array, pubmed_articles_dir):
    article_files = glob.glob(os.path.join(pubmed_articles_dir, "PubMed_Articles_*.json"))
    def numeric_key(filename):
        match = re.search(r'PubMed_Articles_(\d+)\.json', filename)
        if match:
            return int(match.group(1))
        return 9999999 # fallback if for some reason it doesn't match
    # Sort the article JSON chunks in ascending order
    article_files = sorted(article_files, key=numeric_key)
    
    all_articles = []
    for chunk_id, path in enumerate(article_files):
        print(f"Loading PubMed articles from: {path}")
        with open(path) as f:
            # chunk_articles is a dict of the form: { "1": {...}, "2": {...}, ... }
            chunk_articles = json.load(f)
            for pmid_str, article_data in chunk_articles.items():
                # article_data = {"d": "...", "t": "...", "a": "...", "m": "..."}
                the_abstract = article_data.get("a", "")
                
                all_articles.append({
                    "pmid": pmid_str,
                    "chunk_index": chunk_id,
                    "abstract": the_abstract,
                    "corpus": "PubMed"
                })
                
    pubmed_evidences = []
    for indices in tqdm(pubmed_I_array, desc="PubMed decode"):
        evidence_list = []
        for idx in indices:
            if 0 <= idx < len(all_articles):
                evidence_list.append(all_articles[idx])
            else:
                evidence_list.append({"error": f"Index {idx} out of range"})
        pubmed_evidences.append(evidence_list)

    return pubmed_evidences



def pmc_decode(pmc_I_array, pmc_articles_dir):
    """
    Decodes PMC indices by loading each file if it exists.
    """
    def load_article(pmc_articles_dir):
        pmc_articles_local = []
        pmc_files = ["PMC_Main_Articles.json", "PMC_Abs_Articles.json"]
        for filename in pmc_files:
            path = os.path.join(pmc_articles_dir, filename)
            if not os.path.exists(path):
                print(f"Warning: {path} not found. Skipping...")
                continue
            with open(path, 'r') as jsfile:
                pmc_articles_local.extend(json.load(jsfile))
        return pmc_articles_local

    pmc_articles = load_article(pmc_articles_dir)
    if not pmc_articles:
        print("Warning: No PMC articles found. The decoded results will be empty.")

    pmc_evidences = []
    for ith, indices in tqdm(enumerate(pmc_I_array), desc="decode and add", dynamic_ncols=True):
        evidence_list = []
        for j in indices:
            if j < len(pmc_articles):
                evidence_list.append(pmc_articles[j])
            else:
                evidence_list.append({"error": f"Index {j} out of range"})
        pmc_evidences.append(evidence_list)

    return pmc_evidences


def cpg_decode(cpg_I_array, cpg_articles_dir):
    path = os.path.join(cpg_articles_dir, "CPG_128_Total_Articles.json")
    if not os.path.exists(path):
        print(f"Warning: {path} not found.")
        return []
    with open(path) as f:
        raw = json.load(f)

    cpg_articles = []
    for art in raw:
        if isinstance(art, dict):
            art["corpus"] = "CPG"
            cpg_articles.append(art)
        else:
            cpg_articles.append({"text": str(art), "corpus": "CPG"})

    cpg_evidences = []
    for indices in tqdm(cpg_I_array, desc="CPG decode"):
        cpg_evidences.append([
            cpg_articles[i] if 0 <= i < len(cpg_articles)
            else {"error": f"Index {i} out of range"}
            for i in indices
        ])
    return cpg_evidences


def textbook_decode(textbook_I_array, textbook_articles_dir):
    path = os.path.join(textbook_articles_dir, "Textbook_128_Total_Articles.json")
    if not os.path.exists(path):
        print(f"Warning: {path} not found.")
        return []
    with open(path) as f:
        raw = json.load(f)

    textbook_articles = []
    for art in raw:
        if isinstance(art, dict):
            art["corpus"] = "Textbook"
            textbook_articles.append(art)
        else:
            textbook_articles.append({"text": str(art), "corpus": "Textbook"})

    textbook_evidences = []
    for indices in tqdm(textbook_I_array, desc="Textbook decode"):
        textbook_evidences.append([
            textbook_articles[i] if 0 <= i < len(textbook_articles)
            else {"error": f"Index {i} out of range"}
            for i in indices
        ])
    return textbook_evidences












''' (Comment out Statspearls as not needed.)
def statpearls_index_create(statpearls_embeddings_dir):
    statpearls_index = faiss.IndexFlatIP(768)
    embed_path = os.path.join(statpearls_embeddings_dir, "Statpearls_Total_Embeds.npy")

    with tqdm(total=1, desc="statpearls load and add", dynamic_ncols=True) as pbar:
        if not os.path.exists(embed_path):
            print(f"Warning: {embed_path} not found. Skipping...")
        else:
            embeddings = np.load(embed_path).astype(np.float32)
            statpearls_index.add(embeddings)
            del embeddings
        pbar.update(1)

    if statpearls_index.ntotal == 0:
        print("Warning: No valid Statpearls embeddings were loaded into the index!")
    return statpearls_index


def statpearls_decode(statpearls_index, statpearls_articles_dir):
    """
    Decodes Statpearls indices by loading Statpearls_Total_Articles.json if it exists.
    """
    def load_articles(statpearls_articles_dir):
        sp_path = os.path.join(statpearls_articles_dir, "Statpearls_Total_Articles.json")
        if not os.path.exists(sp_path):
            print(f"Warning: {sp_path} not found. Returning empty list.")
            return []
        with open(sp_path, 'r') as jsfile:
            return json.load(jsfile)

    statpearls_articles = load_articles(statpearls_articles_dir)
    if not statpearls_articles:
        print("Warning: No Statpearls articles found. The decoded results will be empty.")

    statpearls_evidences = []
    for ith, indices in tqdm(enumerate(statpearls_index), desc="decode and add", dynamic_ncols=True):
        evidence_list = []
        for j in indices:
            if j < len(statpearls_articles):
                evidence_list.append(statpearls_articles[j])
            else:
                evidence_list.append({"error": f"Index {j} out of range"})
        statpearls_evidences.append(evidence_list)

    return statpearls_evidences

'''
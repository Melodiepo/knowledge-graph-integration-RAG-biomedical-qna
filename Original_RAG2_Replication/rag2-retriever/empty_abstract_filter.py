import os
import glob
import json
import re
import numpy as np
import faiss
from tqdm import tqdm
import random

def extract_number_from_filename(path):
    """
    Helper to extract the numeric portion from filenames like:
        PubMed_Articles_0.json
        PubMed_Embeds_0.npy
    so we can match them up by chunk index.
    """
    # e.g. search for "_(\d+)."
    match = re.search(r'_(\d+)\.', os.path.basename(path))
    return int(match.group(1)) if match else 9999999

def filter_pubmed_chunks(
    pubmed_embeddings_dir,
    pubmed_articles_dir,
    output_filtered_dir,
    sample_size_per_chunk=5000,
    random_seed=42
):
    """
    PASS A:
    Command: python -c 'import os; from empty_abstract_filter import filter_pubmed_chunks; train_vecs = filter_pubmed_chunks(pubmed_embeddings_dir="/cs/student/projects1/ml/2024/yihanli/retriever/embeddings/pubmed", pubmed_articles_dir="/cs/student/projects1/ml/2024/yihanli/retriever/articles/pubmed", output_filtered_dir="/cs/student/projects1/ml/2024/yihanli/retriever/PubMed_filtered", sample_size_per_chunk=5000, random_seed=42); print("Train vectors shape:", train_vecs.shape)'
    -------
    1) For each JSON + .npy chunk pair (PubMed_Articles_i.json, PubMed_Embeds_i.npy):
       - Load them
       - Filter out empty 'a' (abstract) entries
       - Save the filtered chunk as:
            output_filtered_dir/PubMed_Articles_filtered_i.json
            output_filtered_dir/PubMed_Embeds_filtered_i.npy
       - Meanwhile, pick some random subset from this chunk to help train FAISS later.

    2) Return a big list of random sample vectors (across all chunks)
       so we can train FAISS on them in a second pass.
    """

    random.seed(random_seed)

    # Make sure the output directory exists
    os.makedirs(output_filtered_dir, exist_ok=True)

    embedding_files = sorted(
        glob.glob(os.path.join(pubmed_embeddings_dir, "PubMed_Embeds_*.npy")),
        key=extract_number_from_filename
    )

    article_files = sorted(
        glob.glob(os.path.join(pubmed_articles_dir, "PubMed_Articles_*.json")),
        key=extract_number_from_filename
    )

    if len(embedding_files) != len(article_files):
        print("WARNING: Mismatch in number of .npy vs .json chunk files!")
        print(f"Found {len(embedding_files)} embedding files, {len(article_files)} article files.")

    all_train_samples = []  # We'll accumulate random vectors for training

    for emb_path, art_path in zip(embedding_files, article_files):
        chunk_id = extract_number_from_filename(emb_path)
        # e.g. chunk_id might be 0, 1, 2, ...

        filtered_json_path = os.path.join(output_filtered_dir, f"PubMed_Articles_filtered_{chunk_id}.json")
        filtered_npy_path  = os.path.join(output_filtered_dir, f"PubMed_Embeds_filtered_{chunk_id}.npy")

        # --- Skip if these already exist ---
        if os.path.exists(filtered_json_path) and os.path.exists(filtered_npy_path):
            print(f"Chunk {chunk_id} already processed, skipping...")
            continue

        print(f"\n=== Filtering chunk {chunk_id} ===")
        print(f"Embeds: {emb_path}")
        print(f"Articles: {art_path}")

        chunk_vectors = np.load(emb_path).astype(np.float32)
        with open(art_path, 'r') as f:
            chunk_articles = json.load(f)

        # Sort the keys so row i corresponds to the i-th item
        sorted_keys = sorted(chunk_articles.keys(), key=lambda x: int(x))
        if len(sorted_keys) != chunk_vectors.shape[0]:
            print(f"ERROR: Mismatch in chunk {chunk_id}: {len(sorted_keys)} articles vs {chunk_vectors.shape[0]} vectors.")
            continue

        filtered_vectors = []
        filtered_articles = []

        for i, pmid_str in enumerate(sorted_keys):
            data = chunk_articles[pmid_str]
            abstract_text = data.get("a", "").strip()

            if abstract_text == "":
                # skip
                continue

            # Keep this doc
            filtered_vectors.append(chunk_vectors[i])
            filtered_articles.append({
                "pmid": pmid_str,
                "d": data.get("d",""),
                "t": data.get("t",""),
                "a": abstract_text,
                "m": data.get("m",""),
                "corpus": "PubMed"
            })

        filtered_vectors = np.array(filtered_vectors, dtype=np.float32)

        # Save chunk to disk
        filtered_json_path = os.path.join(output_filtered_dir, f"PubMed_Articles_filtered_{chunk_id}.json")
        filtered_npy_path  = os.path.join(output_filtered_dir, f"PubMed_Embeds_filtered_{chunk_id}.npy")

        with open(filtered_json_path, 'w') as jf:
            json.dump(filtered_articles, jf, indent=2)
        np.save(filtered_npy_path, filtered_vectors)
        print(f"   => saved {len(filtered_articles)} filtered articles to {filtered_json_path}")
        print(f"   => saved shape {filtered_vectors.shape} filtered vectors to {filtered_npy_path}")

        # Collect random sample from this chunk for training
        if len(filtered_vectors) > 0:
            # pick min(sample_size_per_chunk, len(filtered_vectors)) random rows
            sample_indices = random.sample(range(filtered_vectors.shape[0]),
                                           k=min(sample_size_per_chunk, filtered_vectors.shape[0]))
            chunk_samples = filtered_vectors[sample_indices]
            all_train_samples.append(chunk_samples)

    # Combine all chunk samples into one big array
    if all_train_samples:
        train_samples = np.concatenate(all_train_samples, axis=0)
        print(f"\n[PASS A done] Gathered total train samples shape={train_samples.shape}.")
    else:
        train_samples = np.array([], dtype=np.float32).reshape(0,768)
        print("\n[PASS A done] No non-empty articles found? Something is wrong or everything was empty.")

    return train_samples



def build_filtered_pubmed_index(
    filtered_dir,
    output_index_path,
    train_vectors,
    dimension=768,
    nlist=4096,
    m=64
):
    """
    PASS B:
    -------
    1) Use `train_vectors` to train an IVFPQ index.
    2) Then read each filtered chunk .npy file from `filtered_dir`, add them to index.
    3) Save final index to `output_index_path`.
    4) Also (optionally) combine all filtered chunk JSON into a single big JSON or keep chunk by chunk.
    """

    # 1) Create an IVFPQ index
    quantizer = faiss.IndexFlatIP(dimension)
    index = faiss.IndexIVFPQ(
        quantizer,
        dimension,
        nlist,
        m,
        8
    )
    index.metric_type = faiss.METRIC_INNER_PRODUCT

    # Train
    if train_vectors.shape[0] == 0:
        raise RuntimeError("No train vectors found—cannot train FAISS index!")
    sample_count = min(200000, train_vectors.shape[0])
    print(f"Training IVFPQ index on sample_count={sample_count} from train_vectors")
    index.train(train_vectors[:sample_count])

    # 2) Add each filtered chunk's vectors
    filtered_npys = sorted(
        glob.glob(os.path.join(filtered_dir, "PubMed_Embeds_filtered_*.npy")),
        key=extract_number_from_filename
    )

    total_added = 0
    for npy_path in filtered_npys:
        filtered_vecs = np.load(npy_path).astype(np.float32)
        if filtered_vecs.shape[0] == 0:
            continue
        index.add(filtered_vecs)
        total_added += filtered_vecs.shape[0]
        print(f"Added {filtered_vecs.shape[0]} from {npy_path}, total={total_added}")

    # 3) Save final index
    print(f"Writing final index to {output_index_path}")
    faiss.write_index(index, output_index_path)

    # 4) Optionally combine all chunked JSON into one big file
    combined_json_path = os.path.join(filtered_dir, "PubMed_Articles_filtered_ALL.json")
    print(f"Combining all chunked JSON -> {combined_json_path} ...")
    combined_docs = []
    filtered_jsons = sorted(
        glob.glob(os.path.join(filtered_dir, "PubMed_Articles_filtered_*.json")),
        key=extract_number_from_filename
    )
    for jpath in filtered_jsons:
        with open(jpath,'r') as jf:
            part = json.load(jf)
            combined_docs.extend(part)

    with open(combined_json_path, 'w') as cf:
        json.dump(combined_docs, cf, indent=2)

    print(f"Done. Combined total docs: {len(combined_docs)}")
    return index



def build_filtered_pubmed_index(
    filtered_dir,
    output_index_path,
    train_vectors,
    dimension=768,
    nlist=4096,
    m=64
):
    """
    PASS B:
    -------
    1) Use `train_vectors` to train an IVFPQ index.
    2) Then read each filtered chunk .npy file from `filtered_dir`, add them to index.
    3) Save final index to `output_index_path`.
    4) Also (optionally) combine all filtered chunk JSON into a single big JSON or keep chunk by chunk.
    """

    # 1) Create an IVFPQ index
    quantizer = faiss.IndexFlatIP(dimension)
    index = faiss.IndexIVFPQ(
        quantizer,
        dimension,
        nlist,
        m,
        8
    )
    index.metric_type = faiss.METRIC_INNER_PRODUCT

    # Train
    if train_vectors.shape[0] == 0:
        raise RuntimeError("No train vectors found—cannot train FAISS index!")
    sample_count = min(200000, train_vectors.shape[0])
    print(f"Training IVFPQ index on sample_count={sample_count} from train_vectors")
    index.train(train_vectors[:sample_count])

    # 2) Add each filtered chunk's vectors
    filtered_npys = sorted(
        glob.glob(os.path.join(filtered_dir, "PubMed_Embeds_filtered_*.npy")),
        key=extract_number_from_filename
    )

    total_added = 0
    for npy_path in filtered_npys:
        filtered_vecs = np.load(npy_path).astype(np.float32)
        if filtered_vecs.shape[0] == 0:
            continue
        index.add(filtered_vecs)
        total_added += filtered_vecs.shape[0]
        print(f"Added {filtered_vecs.shape[0]} from {npy_path}, total={total_added}")

    # 3) Save final index
    print(f"Writing final index to {output_index_path}")
    faiss.write_index(index, output_index_path)

    # 4) Optionally combine all chunked JSON into one big file
    combined_json_path = os.path.join(filtered_dir, "PubMed_Articles_filtered_ALL.json")
    print(f"Combining all chunked JSON -> {combined_json_path} ...")
    combined_docs = []
    filtered_jsons = sorted(
        glob.glob(os.path.join(filtered_dir, "PubMed_Articles_filtered_*.json")),
        key=extract_number_from_filename
    )
    for jpath in filtered_jsons:
        with open(jpath,'r') as jf:
            part = json.load(jf)
            combined_docs.extend(part)

    with open(combined_json_path, 'w') as cf:
        json.dump(combined_docs, cf, indent=2)

    print(f"Done. Combined total docs: {len(combined_docs)}")
    return index

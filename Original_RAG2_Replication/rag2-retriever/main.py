"""
This script processes MCQ queries (e.g. MedQA) using precomputed embeddings from corpora (PubMed, PMC, CPG, and textbooks in RAG2) to retrieve relevant documents.

### Input:
- A JSON file (`input_path`) containing MCQ queries. The expected format is a list of dictionaries where each entry represents a question.
- Precomputed embeddings for various corpora stored in the `embeddings_dir`. 
- Corresponding article text files stored in the `articles_dir`.

### Processing Steps:
1. **Query Preprocessing**: If specified, preprocesses the queries for instructional purposes.
2. **Query Encoding**: Encodes MCQ queries into vector representations using a query encoder.
3. **Document Retrieval**:
   - Uses FAISS-based nearest neighbor search to retrieve top-k relevant document embeddings from the chosen corpora.
   - Retrieves relevant document indices separately from PubMed, PMC, CPG, and textbooks.
4. **Document Decoding**: Converts retrieved document indices into actual text snippets from the corpora.
5. **Evidence Combination & Reranking**:
   - Combines retrieved document texts with the original query.
   - Uses a reranker to order the documents based on their relevance.

### Output:
- A JSON file (`output_path`) containing reranked evidences for each query. Each MCQ query will have a list of retrieved and reranked document texts that can be used as supporting evidence.

### Usage Example (updated 3.11):

python /cs/student/projects1/ml/2024/yihanli/retriever/main.py \
-e /cs/student/projects1/ml/2024/yihanli/retriever/embeddings \
-a /cs/student/projects1/ml/2024/yihanli/retriever/articles \
-i /cs/student/projects1/ml/2024/yihanli/retriever/input/input_cot/medmcqa_cot.json \
-c pubmed cpg textbook \
-k 10 \
-o /cs/student/projects1/ml/2024/yihanli/retriever/output/evidence_medmcqa_cot_test_k_10.json \
-spc True

### To do:
Generate retriever output for 
+ k = 1/2/4/8/16/32  
+ each QA set's train/val/test respctively
+ both raw query and CoT

```

"""
import os
import json
import torch
import faiss
import argparse
import scispacy
import spacy
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import numpy as np

import query_encode as qe
import retrieve as rt
import rerank as rr

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--embeddings_dir', help='embeddings directory', default='embeddings')
    parser.add_argument('-a', '--articles_dir', help='articles directory', default='articles')
    parser.add_argument('-i', '--input_path', help='JSON file path for query input or CoT input', required=True)
    parser.add_argument('-c', '--corpus', nargs='+', help='Which corpus to use', 
                        default=['cpg', 'textbook', 'pubmed'])
    parser.add_argument('-k', '--top_k', help='Number of retrieved documents', default=1, type=int)
    parser.add_argument('-inst', '--instruction_preprocess', 
                        help='Optionally preprocess the query for instruction set', 
                        default='False')
    parser.add_argument('-o', '--output_path', help='Output file path', required=True)
    parser.add_argument('-spc', '--use_spacy', help='Use scispacy to insert [SEP] token', 
                        default='False')

    args = parser.parse_args()

    embeddings_dir = args.embeddings_dir
    articles_dir = args.articles_dir
    input_path = args.input_path
    output_path = args.output_path
    corpora = set(args.corpus)
    top_k = args.top_k
    inst = (args.instruction_preprocess == "True")
    use_spacy = (args.use_spacy == "True")

    # -------------------- 1) Load Queries/CoT -------------------- #
    with open(input_path, 'r') as f:
        input_data = json.load(f)

    # Determine if the input is CoT or a normal query.
    # We assume that CoT format has a key "model_output" in the first element.
    is_cot = False
    if isinstance(input_data, list) and len(input_data) > 0:
        if "model_output" in input_data[0]:
            is_cot = True
    
    # Prepare Query Text
    if is_cot:
        # For CoT input, we use the "model_output" as the query
        queries = [entry["model_output"] for entry in input_data]
    else:
        # For normal query input, you can use an instruction preprocess if desired
        if inst:
            queries = qe.query_preprocess_instruction(input_path, use_spacy=use_spacy)
        else:
            queries = [entry["question"] for entry in input_data]
            
    # Encode the queries
    xq = qe.query_encode(queries)

    # if inst == True:
    #     # query preprocess for instruction_set 
    #     input_list = qe.query_preprocess_instruction(input_path, use_spacy = use_spacy)
    # else:
    #     with open(input_path, 'r') as input_file:
    #         input_list = json.load(input_file)
    #     questions = [entry["question"] for entry in input_list]
    #     xq = qe.query_encode(questions)

    # -------------------- 3) Retrieve from corpora -------------------- #
    pubmed_evidences = []
    pmc_evidences = []
    cpg_evidences = []
    textbook_evidences = []

    # PubMed
    if 'pubmed' in corpora:
        pubmed_index = rt.pubmed_index_create(os.path.join(embeddings_dir, "pubmed"))
        pubmed_I_array = []
        batch_size = 1024
        for start in tqdm(range(0, len(xq), batch_size), desc="PubMed FAISS MIPS"):
            batch = xq[start : start + batch_size]
            D, I = pubmed_index.search(batch, top_k)
            pubmed_I_array.extend(I)
        pubmed_evidences = rt.pubmed_decode(pubmed_I_array, os.path.join(articles_dir, "pubmed"))

    # PMC
    if 'pmc' in corpora and not inst:
        pmc_index = rt.pmc_index_create(os.path.join(embeddings_dir, "pmc"))
        pmc_I_array = []
        batch_size = 1024
        for start in tqdm(range(0, len(xq), batch_size), desc="PMC FAISS MIPS"):
            batch = xq[start : start + batch_size]
            D, I = pmc_index.search(batch, top_k)
            pmc_I_array.extend(I)
        del pmc_index
        pmc_evidences = rt.pmc_decode(pmc_I_array, os.path.join(articles_dir, "pmc"))

    # CPG
    if 'cpg' in corpora:
        cpg_index = rt.cpg_index_create(os.path.join(embeddings_dir, "cpg"))
        cpg_I_array = []
        batch_size = 1024
        for start in tqdm(range(0, len(xq), batch_size), desc="CPG FAISS MIPS"):
            batch = xq[start : start + batch_size]
            D, I = cpg_index.search(batch, top_k)
            cpg_I_array.extend(I)
        cpg_evidences = rt.cpg_decode(cpg_I_array, os.path.join(articles_dir, "cpg"))

    # Textbook
    if 'textbook' in corpora:
        textbook_index = rt.textbook_index_create(os.path.join(embeddings_dir, "textbook"))
        textbook_I_array = []
        batch_size = 1024
        for start in tqdm(range(0, len(xq), batch_size), desc="Textbook FAISS MIPS"):
            batch = xq[start : start + batch_size]
            D, I = textbook_index.search(batch, top_k)
            textbook_I_array.extend(I)
        textbook_evidences = rt.textbook_decode(textbook_I_array, os.path.join(articles_dir, "textbook"))

    # -------------------- 4) Rerank & Save -------------------- #
    query_evidences, evidences = rr.combine_query_evidence(
        input_data,
        pubmed_evidences,
        cpg_evidences,
        textbook_evidences,
        pmc_evidences,
        []
    )

    reranked_evidences, confidence_scores = rr.rerank(query_evidences, evidences, top_k)
    
    final_output = []
    for query_obj, retrieved_ids, scores in zip(input_data, reranked_evidences, confidence_scores):
        # Only extract the "question" field from the query object.
        if is_cot:
            q_text = query_obj.get("model_output", query_obj.get("question", ""))
        else:
            q_text = query_obj.get("question", "")
        entry = {"query": q_text, "retrieved": []}
        for obj, score in zip(retrieved_ids, scores):
            # Check if it is a PubMed hit (has a pmid or if it is a string of digits)
            if isinstance(obj, dict) and obj.get("pmid") is not None:
                # Add the chunk index from the article
                entry["retrieved"].append({
                    "pmid": str(obj["pmid"]),
                    "chunk_index": obj.get("chunk_index", "unknown"),
                    "abstract": obj.get("abstract", ""), 
                    "confidence": float(score)
                })
            elif isinstance(obj, str) and obj.isdigit():
                entry["retrieved"].append({
                    "pmid": obj,
                    "chunk_index": "unknown",  # if you don’t have the chunk info for string hits
                    "confidence": float(score)
                })
            else:
                # For non-PubMed hits (CPG/Textbook)
                if isinstance(obj, dict):
                    snippet = (
                        obj.get("text") or
                        obj.get("content") or
                        obj.get("abstract") or
                        str(obj)
                    )
                    corpus = obj.get("corpus", "unknown")
                else:
                    snippet = str(obj)
                    corpus = "Unknown"
                # Later we add the corpus information (see next update)
                entry["retrieved"].append({
                    "text": snippet,
                    "corpus": corpus,
                    "confidence": float(score)
                })
        final_output.append(entry)

    with open(output_path, 'w') as jsfile:
        json.dump(final_output, jsfile, indent=2)
    print(f"\n✅ Reranked evidences saved to {output_path}")


if __name__ == "__main__":
    main()

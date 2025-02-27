import os
import json
import numpy as np
import torch
from sentence_transformers.models import Transformer, Pooling
from sentence_transformers import SentenceTransformer
import tqdm
from sklearn.metrics.pairwise import cosine_similarity


def similarity_check(query, retrieved_texts, retrieved_pmids,
                     context_sim_threshold=75, context_sim_min_threshold=0.5):
    """
    Computes cosine similarity between a query and retrieved documents.
    Args:
        query: query embedding (1D numpy array of shape (768,))
        retrieved_texts: list of retrieved document embeddings (each should be 1D of shape (768,))
        retrieved_pmids: list of corresponding PMIDs for retrieved documents
        context_sim_threshold (float): Percentile threshold for pruning.
        context_sim_min_threshold (float): Minimum similarity threshold.

    Returns:
        pruned_pmids: list of most relevant document PMIDs
    """
    # Ensure query is 2D (1, 768)
    query = np.array(query).reshape(1, -1)

    # Ensure all retrieved document embeddings are 2D (N, 768)
    retrieved_texts = np.array(retrieved_texts)

    if len(retrieved_texts.shape) == 1:  # If single document was mistakenly 1D
        retrieved_texts = retrieved_texts.reshape(1, -1)

    # Compute cosine similarity
    similarities = cosine_similarity(query, retrieved_texts)[0]  # Shape: (N,)
    print(f"Similarities: {similarities}")

    # Compute percentile-based threshold for filtering
    percentile_threshold = np.percentile(similarities, context_sim_threshold)
    print(f"Percentile threshold: {percentile_threshold}")

    # Select snippets above threshold
    pruned_pmids = [
        pmid for i, pmid in enumerate(retrieved_pmids)
        if similarities[i] > percentile_threshold and similarities[i] > context_sim_min_threshold
    ]

    return pruned_pmids


def retrieve_and_prune_context(question, index, metadatas,
                               embedding_model="ncbi/MedCPT-Query-Encoder", k=32,
                               context_sim_threshold=75, context_sim_min_threshold=0.5):
    """
    Retrieves context from FAISS and applies context pruning using cosine similarity.

    Args:
    - question (str): User query.
    - index (faiss.Index): FAISS index containing embedded documents.
    - metadatas (list): Metadata mapping index entries to source documents.
    - retriever_name (str): Name of retriever model.
    - embedding_model (str): Model used for embedding queries and documents.
    - k (int): Number of documents to retrieve from FAISS.
    - context_sim_threshold (float): Percentile threshold for pruning.
    - context_sim_min_threshold (float): Minimum similarity threshold.

    Returns:
    - pruned_contexts (list): List of most relevant document snippets.
    """

    # Step 1: Load embedding model
    device = "cuda" if torch.cuda.is_available() else "cpu"

    embedding_function = SentenceTransformer(embedding_model, device=device)

    # Step 2: Embed the question one at a time
    question_embedding = embedding_function.encode([question])[0]

    # Step 3: Retrieve top-k relevant documents from FAISS
    _, indices = index.search(np.array([question_embedding]), k)

    retrieved_texts = []
    for idx in indices[0]:
        if idx < len(metadatas):  # Prevent index out of range errors
            source = metadatas[idx]['source']
            retrieved_texts.append(source)

    # Step 4: Embed retrieved documents
    snippet_embeddings = embedding_function.encode(retrieved_texts)

    # Step 5: Compute cosine similarity
    similarities = [
        cosine_similarity(np.array(question_embedding).reshape(1, -1), np.array(snippet_embedding).reshape(1, -1))[0][0]
        for snippet_embedding in snippet_embeddings
    ]


    # Step 6: Compute percentile-based threshold for filtering
    percentile_threshold = np.percentile(similarities, context_sim_threshold)

    # Step 7: Select snippets above threshold
    pruned_contexts = [
        retrieved_texts[i] for i in range(len(retrieved_texts))
        if similarities[i] > percentile_threshold and similarities[i] > context_sim_min_threshold
    ]

    return pruned_contexts
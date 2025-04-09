import os
import json
import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModel, AutoTokenizer


class SimilarityPruning:
    def __init__(self, query_model="ncbi/MedCPT-Query-Encoder", article_model="ncbi/MedCPT-Article-Encoder",
                 context_sim_min_threshold=0.5, context_percentile=75):
        self.threshold = context_sim_min_threshold
        self.percentile = context_percentile
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_query = AutoModel.from_pretrained(query_model).to(self.device).eval()
        self.model_tokens = AutoModel.from_pretrained(article_model).to(self.device).eval()
        self.tokenizer_query = AutoTokenizer.from_pretrained(query_model)
        self.tokenizer_tokens = AutoTokenizer.from_pretrained(article_model)

        self.pubmed_cache = {}

    def load_pubmed_articles(self, chunk_index):
        file_name = f"PubMed_Articles_{chunk_index}.json"
        file_path = os.path.join(self.articles_dir, file_name)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                articles_data = json.load(f)
            self.pubmed_cache[chunk_index] = articles_data
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            articles_data = {}
            self.pubmed_cache[chunk_index] = articles_data
        return articles_data

    def compute_similarity(self, query_embedding, text_embedding):
        similarities = [cosine_similarity(np.array(query_embedding).reshape(1, -1), np.array(t).reshape(1, -1))[0][0]
                        for t in text_embedding]

        percentile_threshold = np.percentile(similarities, self.percentile)

        pruned_contexts = []
        ids = []
        for i, text_emb in enumerate(text_embedding):
            if similarities[i] > percentile_threshold and similarities[i] > self.threshold:
                pruned_contexts.append(text_emb)
                ids.append(i)
            else:
                continue

        return ids

    def encode_text(self, content, is_query=True):
        try:
            if is_query:
                inputs = self.tokenizer_query(content, return_tensors="pt", truncation=True, padding=True,
                                              max_length=512).to(self.device)
                with torch.no_grad():
                    outputs = self.model_query(**inputs)
                    embedding = outputs.last_hidden_state[:, 0, :]  # CLS-like token
                    return embedding.cpu().numpy().flatten()
            else:
                inputs = self.tokenizer_tokens(content, return_tensors="pt", truncation=True, padding=True,
                                               max_length=512).to(self.device)
                with torch.no_grad():
                    outputs = self.model_tokens(**inputs)
                    embedding = outputs.last_hidden_state[:, 0, :]
                    return embedding.cpu().numpy()
        except Exception as e:
            print(f"[ERROR] Encoding failed: {str(e)}, is query:{is_query}, content:{content[:50]}")
            return None

    def prune(self, input_retrieved_path, output_path):
        with open(input_retrieved_path, "r") as f:
            retrieved_data = json.load(f)

        pruned_data = {}
        seen_queries = set()
        for idx, content in tqdm(enumerate(retrieved_data), total=len(retrieved_data), desc="Similarity Pruning"):
            query = content["query"]

            if query in seen_queries:
                continue
            else:
                seen_queries.add(query)

            retrieved = content["retrieved"]

            query_emb = self.encode_text(query, is_query=True)
            if query_emb is None:
                raise ValueError(f"Query {query[:10]} is None")

            docs = []
            # print(f"query: {query[:10]}")
            for doc in retrieved:
                if "text" in doc:
                    doc_content = doc["text"]
                elif "abstract" in doc:
                    doc_content = doc["abstract"]
                    if doc_content == "":
                        continue
                else:
                    raise Error("Invalid Source Property")

                docs.append(doc_content)

            # print(f"docs: {docs[i][:10] for i in len(docs)}")

            if not docs:
                pruned_data[idx] = {
                    "query": query,
                    "retrieved_docs": []
                }
            else:
                docs_emb = self.encode_text(docs, is_query=False)
                ids = self.compute_similarity(query_emb, docs_emb)

                pruned_data[idx] = {
                    "query": query,
                    "retrieved_docs": [docs[i] for i in ids]
                }

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(pruned_data, f, indent=4)

        print(f"Labeled data saved to {output_path}")


if __name__ == "__main__":
    INPUT_PATH = "/cs/student/projects1/ml/2024/yihanli/retriever/output/evidence_mmlu_cot_test_k_10.json"
    OUTPUT_PATH = "/cs/student/projects1/ml/2024/yihanli/similarity_pruning/output/mmlu_cot/similarity_pruning_test_k_10.json"

    similarity_pruning = SimilarityPruning()
    print("Model Initialized")
    similarity_pruning.prune(input_retrieved_path=INPUT_PATH, output_path=OUTPUT_PATH)

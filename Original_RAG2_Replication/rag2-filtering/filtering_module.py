import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

class FilteringModule:
    def __init__(self, model_name_or_path: str, articles_path: str, threshold: float = 0.25):
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.threshold = threshold
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

        # Load full document texts
        with open(articles_path, "r") as f:
            self.articles_data = json.load(f)

    def compute_score(self, text: str) -> float:
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            helpful_score = probs[0, 1].item()
        return helpful_score

    def get_document_text(self, pmid: str) -> str:
        doc_data = self.articles_data.get(str(pmid))
        if doc_data:
            title = doc_data.get("t", "")
            abstract = doc_data.get("a", "")
            return f"{title}. {abstract}"
        return ""

    def filter_documents(self, query: str, candidate_docs: list) -> list:
        query_score = self.compute_score(query)
        filtered_docs = []
        for doc in candidate_docs:
            doc_text = self.get_document_text(doc["pmid"])  # Extract full article text
            if not doc_text:
                continue  # Skip if no content is available

            combined_text = f"Query: {query} Document: {doc_text}"
            doc_score = self.compute_score(combined_text)
            delta_score = doc_score - query_score
            if delta_score >= self.threshold:
                filtered_docs.append({
                    "pmid": doc["pmid"],
                    "confidence": doc["confidence"],
                    "text": doc_text
                })
        return filtered_docs

    def run_filtering(self, input_path: str, output_path: str):
        with open(input_path, "r") as f:
            retrieved_data = json.load(f)

        filtered_results = {}
        for idx, content in tqdm(enumerate(retrieved_data), total=len(retrieved_data), desc="Run Filtering"):
            query = content["query"]
            documents = content["retrieved"]
            filtered_docs = self.filter_documents(query, documents)
            filtered_results[idx] = {
                "query": query,
                "filtered_docs": filtered_docs
            }

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(filtered_results, f, indent=4)

if __name__ == "__main__":
    MODEL_PATH = "./fine_tuned_flan_t5"
    ARTICLES_PATH = "/cs/student/projects1/ml/2024/yihanli/retriever/articles/pubmed/PubMed_Articles_0.json"
    INPUT_EVIDENCE_PATH = "/cs/student/projects1/ml/2024/yihanli/retriever/output/evidence_medqa_llama_cot.json"
    OUTPUT_FILTERED_PATH = "rag2-filtering/output/filtered_evidence_medqa_llama_cot.json"

    filtering_module = FilteringModule(model_name_or_path=MODEL_PATH, articles_path=ARTICLES_PATH, threshold=0.25)
    filtering_module.run_filtering(INPUT_EVIDENCE_PATH, OUTPUT_FILTERED_PATH)
    print(f"Filtering complete. Filtered evidence saved to {OUTPUT_FILTERED_PATH}")

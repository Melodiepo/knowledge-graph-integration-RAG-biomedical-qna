import os
import json
import math
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

class FilteringModule:
    def __init__(self, model_name_or_path: str, threshold: float = 0.25):
        """
        Initializes the filtering module.
        Args:
            model_name_or_path (str): Path to the fine-tuned Flan-T5 classification model.
            threshold (float): Minimum classification score difference required to consider a document helpful.
        """
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.threshold = threshold
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def compute_score(self, text: str) -> float:
        """
        Compute the classification score (logit for the 'helpful' class).
        """
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits  # Shape: [1, 2] for binary classification
            probs = torch.softmax(logits, dim=-1)  # Convert logits to probabilities
            helpful_score = probs[0, 1].item()  # Probability for the 'helpful' (positive) class
        return helpful_score

    def filter_documents(self, query: str, candidate_docs: list) -> list:
        """
        Filters documents using classification score differentials.
        """
        query_score = self.compute_score(query)
        filtered_docs = []
        for doc in candidate_docs:
            combined_text = f"{query} {doc}"
            doc_score = self.compute_score(combined_text)
            delta_score = doc_score - query_score
            if delta_score >= self.threshold:
                filtered_docs.append(doc)
        return filtered_docs

    def run_filtering(self, input_path: str, output_path: str):
        """
        Reads retrieved evidence from JSON, filters them, and writes the output to a new JSON.
        """
        with open(input_path, "r") as f:
            retrieved_data = json.load(f)

        filtered_results = {}
        for idx, content in tqdm(enumerate(retrieved_data), total=len(retrieved_data), desc="Run Filtering"):
            query = content["query"]
            documents = content["retrieved_docs"]
            filtered_docs = self.filter_documents(query, documents)
            filtered_results[idx] = {
                "query": query,
                "filtered_docs": filtered_docs
            }

        # Save filtered evidence to output_path
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(filtered_results, f, indent=4)

if __name__ == "__main__":
    MODEL_PATH = "./fine_tuned_flan_t5"  # Path to the fine-tuned Flan-T5 model
    INPUT_EVIDENCE_PATH = "rag2-retriever/output/evidence_medqa_llama_cot.json"
    OUTPUT_FILTERED_PATH = "rag2-filtering/output/filtered_evidence_medqa_llama_cot.json"

    filtering_module = FilteringModule(model_name_or_path=MODEL_PATH, threshold=0.25)
    filtering_module.run_filtering(INPUT_EVIDENCE_PATH, OUTPUT_FILTERED_PATH)
    print(f"Filtering complete. Filtered evidence saved to {OUTPUT_FILTERED_PATH}")

import os
import json
import math
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm

class PerplexityLabelGenerator:
    def __init__(self, model_name_or_path: str, articles_path: str, threshold: float = 0.25):
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.threshold = threshold
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        
        # Load all articles from the JSON file at once
        with open(articles_path, "r", encoding="utf-8") as f:
            self.articles_data = json.load(f)

    def compute_perplexity(self, text: str) -> float:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss.item()
        return math.exp(loss)

    def load_document_content(self, pmid: str) -> str:
        """
        Fetch document content by combining title and abstract from the JSON data.
        """
        doc_data = self.articles_data.get(str(pmid))
        if doc_data:
            title = doc_data.get("t", "")
            abstract = doc_data.get("a", "")
            return f"{title}. {abstract}"
        return ""

    def generate_labels(self, input_path: str, output_path: str):
        with open(input_path, "r") as f:
            retrieved_data = json.load(f)

        labeled_data = {}
        for idx, content in tqdm(enumerate(retrieved_data), total=len(retrieved_data), desc="Generating Labels"):
            query = content["query"]
            docs = content["retrieved"]
            labels = []

            base_perplexity = self.compute_perplexity(query)

            for doc in docs:
                pmid = doc["pmid"]
                doc_content = self.load_document_content(pmid)

                if doc_content:
                    combined_text = f"{query} {doc_content}"
                else:
                    combined_text = query  # If content isn't available, use the query alone

                doc_perplexity = self.compute_perplexity(combined_text)
                delta_ppl = base_perplexity - doc_perplexity
                label = 1 if delta_ppl >= self.threshold else 0
                labels.append(label)

            labeled_data[idx] = {
                "query": query,
                "retrieved_docs": [doc['pmid'] for doc in docs],
                "labels": labels
            }

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(labeled_data, f, indent=4)

        print(f"Labeled data saved to {output_path}")

if __name__ == "__main__":
    MODEL_NAME = "google/flan-t5-base"
    INPUT_EVIDENCE_PATH = "/cs/student/projects1/ml/2024/yihanli/retriever/output/evidence_medqa_llama_cot.json"
    OUTPUT_LABELED_PATH = "/cs/student/projects1/ml/2024/yihanli/uandreic/rag2-filtering/output/labeled_perplexity_data.json"
    ARTICLES_PATH = "/cs/student/projects1/ml/2024/yihanli/retriever/articles/pubmed/PubMed_Articles_0.json"  # Path to JSON file

    label_generator = PerplexityLabelGenerator(
        model_name_or_path=MODEL_NAME,
        articles_path=ARTICLES_PATH,
        threshold=0.25
    )
    label_generator.generate_labels(INPUT_EVIDENCE_PATH, OUTPUT_LABELED_PATH)
    print("Label generation completed.")

    
import os
import json
import math
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm


class PerplexityLabelGenerator:
    def __init__(
        self, model_name_or_path: str, articles_dir: str, threshold: float = 0.25
    ):
        """
        Initialize the PerplexityLabelGenerator. This class generates labels for evidence data based on the perplexity
        differential between a query and the query combined with each retrieved document.

        Args:
            model_name_or_path (str): Path or identifier for the fine-tuned sequence-to-sequence model.
            articles_dir (str): Directory containing JSON files with PubMed articles.
            threshold (float): Minimum perplexity differential required to assign a label.
        """
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.threshold = threshold
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

        # Store the articles directory; we use a cache to load PubMed files as needed
        self.articles_dir = articles_dir
        self.pubmed_cache = {}  # Cache mapping chunk_index -> articles data (dict)

    def load_pubmed_articles(self, chunk_index: int) -> dict:
        """
        Load and cache a PubMed articles JSON file corresponding to a given chunk_index.
        """
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

    def compute_perplexity(self, text: str) -> float:
        """
        Compute perplexity using the model's loss (exponential of the loss).
        """
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss.item()
        return math.exp(loss)

    def get_pubmed_document_text(self, pmid: str, chunk_index: int) -> str:
        """
        Load the relevant PubMed JSON file based on chunk_index (if not already loaded)
        and return the document text (title and abstract) for the given pmid.
        """
        if chunk_index not in self.pubmed_cache:
            self.load_pubmed_articles(chunk_index)
        articles_data = self.pubmed_cache.get(chunk_index, {})
        doc_data = articles_data.get(str(pmid))
        if doc_data:
            title = doc_data.get("t", "")
            abstract = doc_data.get("a", "")
            return f"{title}. {abstract}"
        return ""

    def generate_labels(self, input_path: str, output_path: str):
        """
        For each query in the evidence file, compute a perplexity differential between
        the query and the query concatenated with the relevant document text. For PubMed
        documents, load the corresponding file based on the chunk_index.

        The final output preserves the chunk ids for each retrieved document that has both
        "pmid" and "chunk_index". For other cases (i.e. when a "text" field is present),
        the text is kept as-is.
        """
        with open(input_path, "r") as f:
            retrieved_data = json.load(f)

        labeled_data = {}
        for idx, content in tqdm(
            enumerate(retrieved_data),
            total=len(retrieved_data),
            desc="Generating Labels",
        ):
            query = content["query"]
            docs = content["retrieved"]
            labels = []

            base_perplexity = self.compute_perplexity(query)

            for doc in docs:
                # Process PubMed evidence if "pmid" and "chunk_index" are provided.
                if isinstance(doc, dict) and "pmid" in doc and "chunk_index" in doc:
                    pmid = doc["pmid"]
                    chunk_index = doc["chunk_index"]
                    doc_content = self.get_pubmed_document_text(pmid, chunk_index)
                # Process other corpora using the "text" field.
                elif isinstance(doc, dict) and "text" in doc:
                    doc_content = doc["text"]
                else:
                    doc_content = ""

                combined_text = f"{query} {doc_content}" if doc_content else query
                doc_perplexity = self.compute_perplexity(combined_text)
                delta_ppl = base_perplexity - doc_perplexity
                label = 1 if delta_ppl >= self.threshold else 0
                labels.append(label)

            # Preserve chunk IDs for PubMed docs.
            output_docs = []
            for doc in docs:
                if isinstance(doc, dict) and "pmid" in doc and "chunk_index" in doc:
                    output_docs.append(
                        {"pmid": doc["pmid"], "chunk_index": doc["chunk_index"]}
                    )
                elif isinstance(doc, dict) and "text" in doc:
                    output_docs.append(doc["text"])
                else:
                    output_docs.append(doc)

            labeled_data[idx] = {
                "query": query,
                "retrieved_docs": output_docs,
                "labels": labels,
            }

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(labeled_data, f, indent=4)

        print(f"Labeled data saved to {output_path}")


if __name__ == "__main__":
    MODEL_NAME = "google/flan-t5-large"
    INPUT_EVIDENCE_PATH = "/cs/student/projects1/ml/2024/yihanli/retriever/output/evidence_medmcqa_test_k_10.json"
    OUTPUT_LABELED_PATH = "/cs/student/projects1/ml/2024/yihanli/uandreic/rag2-filtering/output/labeled_perplexity_data.json"
    ARTICLES_DIR = "/cs/student/projects1/ml/2024/yihanli/retriever/articles/pubmed/"  # Directory containing PubMed JSON files

    label_generator = PerplexityLabelGenerator(
        model_name_or_path=MODEL_NAME, articles_dir=ARTICLES_DIR, threshold=0.25
    )
    label_generator.generate_labels(INPUT_EVIDENCE_PATH, OUTPUT_LABELED_PATH)
    print("Label generation completed.")
    
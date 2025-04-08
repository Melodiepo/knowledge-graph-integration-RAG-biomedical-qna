import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm


class FilteringModule:
    """
    A module for filtering candidate documents based on their helpfulness scores relative to a query.

    This module uses a fine-tuned sequence classification model to compute a helpfulness score for text.
    It compares the score for the query alone with the score for the query combined with each candidate
    document. Documents whose score difference (delta) is above a given threshold are retained.

    For PubMed documents, the module retrieves article data (title and abstract) on-demand from JSON files,
    caching them to avoid repeated disk reads.
    """

    def __init__(
        self, model_name_or_path: str, articles_dir: str, threshold: float = 0.25
    ):
        """
        Initialize the FilteringModule.

        Args:
            model_name_or_path (str): Path or identifier for the fine-tuned sequence classification model.
            articles_dir (str): Directory containing JSON files with PubMed articles.
            threshold (float): Minimum score difference required to retain a document.
        """
        # Load the fine-tuned classifier and tokenizer.
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.threshold = threshold
        # Set device to GPU if available, else CPU.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()  # Set the model to evaluation mode.

        # Store the articles directory and initialize a cache for PubMed articles.
        self.articles_dir = articles_dir
        self.pubmed_cache = {}  # Cache: maps chunk_index -> articles data (dict)

    def load_pubmed_articles(self, chunk_index: int) -> dict:
        """
        Load and cache a PubMed articles JSON file given its chunk_index.

        Args:
            chunk_index (int): The index of the chunk file to load.

        Returns:
            dict: The dictionary containing articles data from the JSON file.
        """
        file_name = f"PubMed_Articles_{chunk_index}.json"
        file_path = os.path.join(self.articles_dir, file_name)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                articles_data = json.load(f)
            # Cache the loaded articles for future use.
            self.pubmed_cache[chunk_index] = articles_data
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            articles_data = {}
            self.pubmed_cache[chunk_index] = articles_data
        return articles_data

    def get_document_text(self, doc: dict) -> str:
        """
        Retrieve the text of a document from a candidate evidence item.

        For PubMed documents, uses the 'pmid' and 'chunk_index' fields to load the corresponding
        article from the cached JSON file. For other corpora (e.g., Textbook or CPG), returns the 'text'
        field directly.

        Args:
            doc (dict): A dictionary representing a candidate document.

        Returns:
            str: The document text (combined title and abstract for PubMed) or an empty string if unavailable.
        """
        if "pmid" in doc and "chunk_index" in doc:
            chunk_index = doc["chunk_index"]
            pmid = doc["pmid"]
            # Load articles for this chunk if not already cached.
            if chunk_index not in self.pubmed_cache:
                self.load_pubmed_articles(chunk_index)
            articles_data = self.pubmed_cache.get(chunk_index, {})
            doc_data = articles_data.get(str(pmid))
            if doc_data:
                title = doc_data.get("t", "")
                abstract = doc_data.get("a", "")
                return f"{title}. {abstract}"
            return ""
        elif "text" in doc:
            return doc["text"]
        else:
            return ""

    def compute_score(self, text: str) -> float:
        """
        Compute the helpfulness score for the given text using the classifier.

        Assumes a binary classification setup where class 0 is "not helpful" and class 1 is "helpful".
        The score for the "helpful" class is returned.

        Args:
            text (str): The input text to score.

        Returns:
            float: The probability score for the "helpful" class.
        """
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            # Convert logits to probabilities.
            probs = torch.softmax(logits, dim=-1)
            helpful_score = probs[0, 1].item()  # Score for the "helpful" class.
        return helpful_score

    def filter_documents(self, query: str, candidate_docs: list) -> list:
        """
        Filter candidate documents by comparing their helpfulness scores.

        Computes the helpfulness score for the query alone and for the query combined with each document's text.
        The difference (delta score) is compared to the threshold; if the delta is equal or above the threshold,
        the document is retained.

        Args:
            query (str): The input query text.
            candidate_docs (list): A list of candidate document dictionaries.

        Returns:
            list: A list of filtered document dictionaries that meet the threshold criteria.
        """
        # Compute score for the query without any document.
        query_score = self.compute_score(query)
        filtered_docs = []
        for doc in candidate_docs:
            doc_text = self.get_document_text(doc)
            if not doc_text:
                continue  # Skip documents with no text.
            # Create a combined string containing both the query and the document text.
            combined_text = f"Query: {query} Document: {doc_text}"
            doc_score = self.compute_score(combined_text)
            delta_score = doc_score - query_score
            # Retain the document if the score difference meets or exceeds the threshold.
            if delta_score >= self.threshold:
                filtered_entry = {
                    "confidence": doc.get("confidence", None),
                    "text": doc_text,
                }
                if "pmid" in doc:
                    filtered_entry["pmid"] = doc["pmid"]
                if "corpus" in doc:
                    filtered_entry["corpus"] = doc["corpus"]
                filtered_docs.append(filtered_entry)
        return filtered_docs

    def run_filtering(self, input_path: str, output_path: str):
        """
        Execute the filtering process on retrieved evidence.

        Loads retrieved evidence from a JSON file, applies document filtering for each query,
        and writes the filtered results to an output JSON file.

        Args:
            input_path (str): Path to the JSON file with retrieved evidence.
            output_path (str): Path to save the filtered evidence JSON.
        """
        # Load retrieved evidence data.
        with open(input_path, "r") as f:
            retrieved_data = json.load(f)

        filtered_results = {}
        # Iterate through each query's evidence.
        for idx, content in tqdm(
            enumerate(retrieved_data),
            total=len(retrieved_data),
            desc="Filtering Documents",
        ):
            query = content["query"]
            documents = content["retrieved"]
            # Filter candidate documents for this query.
            filtered_docs = self.filter_documents(query, documents)
            filtered_results[idx] = {"query": query, "filtered_docs": filtered_docs}

        # Ensure the output directory exists.
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        # Write the filtered results to a JSON file.
        with open(output_path, "w") as f:
            json.dump(filtered_results, f, indent=4)

        print(f"Filtering complete. Filtered evidence saved to {output_path}")


if __name__ == "__main__":
    # Define model and file paths.
    MODEL_PATH = "/cs/student/projects1/ml/2024/yihanli/uandreic/rag2-filtering/fine_tuned_flan_t5_large"
    ARTICLES_DIR = "/cs/student/projects1/ml/2024/yihanli/retriever/articles/pubmed/"
    INPUT_EVIDENCE_PATH = "/cs/student/projects1/ml/2024/yihanli/retriever/output/evidence_medmcqa_test_k_10.json"
    OUTPUT_FILTERED_PATH = "/cs/student/projects1/ml/2024/yihanli/uandreic/rag2-filtering/output/filtered_evidence_medmcqa_test_k_10.json"

    # Instantiate and run the filtering module.
    filtering_module = FilteringModule(
        model_name_or_path=MODEL_PATH, articles_dir=ARTICLES_DIR, threshold=0.25
    )
    filtering_module.run_filtering(INPUT_EVIDENCE_PATH, OUTPUT_FILTERED_PATH)

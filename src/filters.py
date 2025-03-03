from transformers import AutoModel, AutoTokenizer
import sys
import os
import json
import torch
from tqdm import tqdm

# Add the path to the Original_RAG2_Replication directory to the system path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Original_RAG2_Replication'))
from Original_RAG2_Replication.rag2_filtering.filtering_module import FilteringModule


def filter_provence(question, context):
    """
    This function calls Provence context pruning filter. 

    Input:
    --------------------
    question: user query
    context: the context returned by the retriever

    Returns:
    --------------------
    pruned context    
    """

    provence = AutoModel.from_pretrained("naver/provence-reranker-debertav3-v1", trust_remote_code=True)
    provence_output = provence.process(question, context)
    return provence_output['pruned_context']


class ProvenceFilteringModule(FilteringModule):
    """
    A filtering module that uses the Provence model for context pruning.
    Inherits from the FilteringModule class and overrides the filter_documents method.
    """
    
    def __init__(self, articles_path: str):
        """
        Initialize the ProvenceFilteringModule.
        
        Args:
            articles_path (str): Path to the articles data file.
        """
        # Initialize parent class without model_name_or_path since we'll use Provence instead
        # We're setting a dummy threshold since we won't use it
        super().__init__(model_name_or_path=None, articles_path=articles_path, threshold=0)
        
        # Override the model and tokenizer with Provence
        self.model = AutoModel.from_pretrained("naver/provence-reranker-debertav3-v1", trust_remote_code=True)
        self.tokenizer = None  # Provence handles tokenization internally
        
        # We don't need to move the model to a device as Provence handles this internally
        
    def filter_documents(self, query: str, candidate_docs: list) -> list:
        """
        Filter documents using the Provence model.
        
        Args:
            query (str): The user query.
            candidate_docs (list): List of candidate documents to filter.
            
        Returns:
            list: Filtered documents.
        """
        filtered_docs = []
        
        # Prepare context from candidate documents
        context = ""
        doc_map = {}  # Map to keep track of original document data
        
        for doc in candidate_docs:
            doc_text = self.get_document_text(doc["pmid"])
            if not doc_text:
                continue
                
            # Add document to context with a separator
            context += f"\n\nDocument {doc['pmid']}:\n{doc_text}"
            
            # Store original document data
            doc_map[str(doc["pmid"])] = {
                "pmid": doc["pmid"],
                "confidence": doc["confidence"],
                "text": doc_text
            }
        
        # If no valid documents, return empty list
        if not context:
            return []
            
        # Use Provence to prune the context
        pruned_context = filter_provence(query, context)
        
        # Extract the documents that remain in the pruned context
        # This is a simple approach - we check if the document ID appears in the pruned context
        for pmid, doc_data in doc_map.items():
            if f"Document {pmid}:" in pruned_context:
                filtered_docs.append(doc_data)
        
        return filtered_docs
    
    def run_filtering(self, input_path: str, output_path: str):
        """
        Run the filtering process on the input data and save the results.
        
        Args:
            input_path (str): Path to the input data file.
            output_path (str): Path to save the filtered results.
        """
        # This method is inherited from the parent class and works as is
        super().run_filtering(input_path, output_path)


# # Example usage
# if __name__ == "__main__":
#     ARTICLES_PATH = "/path/to/articles/data.json"
#     INPUT_EVIDENCE_PATH = "/path/to/input/evidence.json"
#     OUTPUT_FILTERED_PATH = "/path/to/output/filtered_evidence.json"
    
#     filtering_module = ProvenceFilteringModule(articles_path=ARTICLES_PATH)
#     filtering_module.run_filtering(INPUT_EVIDENCE_PATH, OUTPUT_FILTERED_PATH)
#     print(f"Filtering complete. Filtered evidence saved to {OUTPUT_FILTERED_PATH}")

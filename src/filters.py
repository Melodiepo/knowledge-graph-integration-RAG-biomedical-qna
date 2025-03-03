from transformers import AutoModel
import json
import nltk


def load_articles(articles_path: str) -> dict:
    """
    Loads the articles JSON file which maps PMIDs to document data.
    Expected JSON format per document:
      {
         "t": "Title text",
         "a": "Abstract text"
      }
    """
    with open(articles_path, "r") as f:
        articles_data = json.load(f)
        return articles_data

def get_document_text(pmid:str, articles_data:dict) -> str:
    """
    Retrieves and concatenates the title and abstract for a given PMID.
    """
    doc_data = articles_data.get(str(pmid))
    if doc_data:
        title = doc_data.get("t", "")
        abstract = doc_data.get("a", "")
        return f"{title}. {abstract}"
    return ""

def filter_provence(query: str, 
                    pmids: list,
                    articles_path:str,
                    model_name: str = "naver/provence-reranker-debertav3-v1",
                    always_select_title: bool = True, 
                    threshold: float = 0.1) -> dict:
    """
    Prunes the context of documents using the Provence model.
    
    Args:
        query (str): The user question.
        pmids (list): A list of PMIDs corresponding to retrieved documents.
        articles_path (str): Path to the JSON file containing article data.
        model_name (str): Model identifier for the Provence pruner.
        always_select_title (bool): Whether to always include the title (first sentence).
        threshold (float): Threshold for pruning; lower values are more conservative.
        
    Returns:
        dict: A dictionary mapping each PMID to its pruned context.
    """

    # Load articles data from the JSON file
    articles_data = load_articles(articles_path)

    # Load the Provence model (ensure that you have internet access or a cached version)
    provence = AutoModel.from_pretrained(model_name, trust_remote_code=True)

    pruned_results = []
    for pmid in pmids:
        doc_text = get_document_text(pmid, articles_data)
        if not doc_text:
            # Skip PMIDs with no available document text
            continue
        
        # Use Provence to prune the document context
        result = provence.process(query, doc_text, always_select_title=always_select_title, threshold=threshold)
        pruned_context = result.get("pruned_context", doc_text)
        pruned_score = result.get("reranking_score", 0)
        pruned_results.append(
            {
            "pmid": pmid,
            "pruned_context": pruned_context,
            "score": pruned_score
        }
        )

    # Sort the results by score in descending order
    sorted_results = sorted(pruned_results, key=lambda x: x["score"], reverse=True)
    return sorted_results

"""
if __name__ == "__main__":
    # Example to test this code:
    example_query = ("A junior orthopaedic surgery resident is completing a carpal tunnel repair "
                     "with the department chairman as the attending physician. During the case, "
                     "the resident inadvertently cuts a flexor tendon. The tendon is repaired without "
                     "complication. The attending tells the resident that the patient will do fine, "
                     "and there is no need to report this minor complication that will not harm the patient, "
                     "as he does not want to make the patient worry unnecessarily. He tells the resident to "
                     "leave this complication out of the operative report. Which of the following is the correct "
                     "next action for the resident to take?")
    
    # Example list of PMIDs from the retriever output
    example_pmids = ["36895610", "36339070", "36212594", "36576536", "36727955"]
    
    # Provide the correct path to your articles JSON file
    articles_file_path = "/cs/student/projects1/ml/2024/yihanli/retriever/articles/pubmed/PubMed_Articles_0.json"
    
    pruned_docs = filter_provence(example_query, example_pmids, articles_file_path)
    
    # Output the sorted pruned contexts for each PMID along with their score
    for doc in pruned_docs:
        print(f"PMID: {doc['pmid']}")
        print(f"Reranking Score: {doc['score']}")
        print(f"Pruned Context: {doc['pruned_context']}")
        print("-" * 80)
"""

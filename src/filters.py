from transformers import AutoModel


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
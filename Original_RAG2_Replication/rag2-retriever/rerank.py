import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def combine_query_evidence(queries, list1, list2, list3, list4, list5):
    """ Ensure each list has exactly as many elements as there are queries. """
    max_len = len(queries)

    # Pad or truncate each evidence list so that all are length == #queries
    for lst in [list1, list2, list3, list4, list5]:
        while len(lst) < max_len:
            lst.append([])  # append empty if missing
        if len(lst) > max_len:
            # If a list is too long, truncate extra items
            lst[:] = lst[:max_len]

    # Now build evidences_5 by concatenating sublists for each query index
    evidences_5 = []
    for i in range(max_len):
        combined_sublist = (
            list1[i] + list2[i] + list3[i] + list4[i] + list5[i]
        )
        evidences_5.append(combined_sublist)

    q_5a_list = []
    for ith, q in tqdm(enumerate(queries)):
        q_5a = []
        for a in evidences_5[ith]:
            if not isinstance(a, str):
                a = str(a)
            q_a = [str(q), a]  
            q_5a.append(q_a)
        q_5a_list.append(q_5a)

    return q_5a_list, evidences_5

def rerank(q_5a_list, evidences_5, top_k):
    """ Reranks the retrieved evidence based on relevance scores using a cross-encoder. """

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    tokenizer = AutoTokenizer.from_pretrained("ncbi/MedCPT-Cross-Encoder")
    model = AutoModelForSequenceClassification.from_pretrained("ncbi/MedCPT-Cross-Encoder")
    model.eval()
    model = model.to(device)  

    reranked_results = []
    confidence_scores = []

    for query_idx, q_5a in tqdm(enumerate(q_5a_list), desc="Reranking Queries"):

        if not q_5a:  # Skip empty query-evidence pairs
            reranked_results.append([])
            confidence_scores.append([])
            continue

        with torch.no_grad():
            encoded_q_5a = tokenizer(
                q_5a,
                truncation=True,
                padding=True,
                return_tensors="pt",
                max_length=512,
            ).to(device)

            logits_q_5a = model(**encoded_q_5a).logits.squeeze(dim=1).detach().cpu().numpy()

        # Sort by confidence scores (descending order)
        ranked_indices = logits_q_5a.argsort()[::-1][:top_k]
        ranked_evidence = [evidences_5[query_idx][i] for i in ranked_indices]
        ranked_scores = [float(logits_q_5a[i]) for i in ranked_indices]  # Convert to float for JSON serialization

        reranked_results.append(ranked_evidence)
        confidence_scores.append(ranked_scores)

    return reranked_results, confidence_scores  

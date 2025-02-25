import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def combine_query_evidence(queries, list1, list2, list3, list4, list5):
    evidences_5 = []
    evidences_5 = [sublist1 + sublist2 + sublist3 + sublist4 + sublist5 for sublist1, sublist2, sublist3, sublist4, sublist5 in zip(list1, list2, list3, list4, list5)]
    q_5a_list = []
    for ith, q in tqdm(enumerate(queries)):
        q_5a = []
        for a in evidences_5[ith]:
            q_a = [q, a]
            q_5a.append(q_a)
        q_5a_list.append(q_5a)
    return q_5a_list, evidences_5

'''
def combine_query_evidence(queries, list1):
    print(f"Number of queries received: {len(queries)}")
    print(f"Number of evidences received: {len(list1)}")

    evidences_combined = []
    for i, evidence_list in enumerate(list1):
        if len(evidence_list) == 0:
            print(f"Warning: Empty evidence list at index {i}")
        evidences_combined.append(evidence_list)

    q_evidence_pairs = []
    for ith, q in enumerate(queries):
        q_evidence = []
        for a in evidences_combined[ith]:
            q_a = [q, a]
            q_evidence.append(q_a)
        q_evidence_pairs.append(q_evidence)

    return q_evidence_pairs, evidences_combined
'''

def rerank(q_5a_list, evidences_5, top_k):
    # Automatically use GPU if available, otherwise CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("ncbi/MedCPT-Cross-Encoder")
    model = AutoModelForSequenceClassification.from_pretrained("ncbi/MedCPT-Cross-Encoder")
    model.eval()
    model = model.to(device)

    logits_list = []
    for idx, q_5a in tqdm(enumerate(q_5a_list), total=len(q_5a_list)):
        with torch.no_grad():
            '''
            encoded_q_5a = tokenizer(
                q_5a,
                truncation=True,
                padding=True,
                return_tensors="pt",
                max_length=512,
            )
            '''

            # Separate queries and evidences for proper tokenization
            queries = [str(pair[0]) for pair in q_5a if isinstance(pair[0], str)]  # Convert to string
            evidences = [str(pair[1]) for pair in q_5a if isinstance(pair[1], str)]  # Convert to string

            # Debugging: Check for any non-string or missing data
            # Add detailed debug logs
            if len(queries) != len(evidences):
                print(f"\n❌ Data mismatch at index {idx}:")
                print(" - Queries length:", len(queries))
                print(" - Evidences length:", len(evidences))
                print(" - Queries Sample:", queries[:5])
                print(" - Evidences Sample:", evidences[:5])
                continue  # Skip problematic entries

            if len(queries) == 0 or len(evidences) == 0:
                print(f"\n⚠️ Empty evidence for query at index {idx}")
                continue

            # Tokenize with queries and evidences as separate lists
            encoded_q_5a = tokenizer(
                queries,
                evidences,
                truncation=True,
                padding=True,
                return_tensors="pt",
                max_length=512,
            )

            encoded_q_5a = {key: tensor.to(device) for key, tensor in encoded_q_5a.items()}
            logits_q_5a = model(**encoded_q_5a).logits.squeeze(dim=1)
            logits_q_5a = logits_q_5a.detach().cpu()
            logits_list.append(logits_q_5a)

    #logits_list_serializable = [tensor.numpy().tolist() for tensor in logits_list]
    #with open('logits_list.json', 'w') as f:
    #    json.dump(logits_list_serializable, f)

    sorted_indices = [sorted(range(len(logits_5)), key=lambda k: logits_5[k], reverse=True) for logits_5 in logits_list]
    top_k_indices = [sorted_indices_i[:top_k] for sorted_indices_i in sorted_indices]
    sorted_evidence_list = []
    for index, data in enumerate(evidences_5):
        sorted_evidence_list.append([data[i] for i in top_k_indices[index]])
        
    return sorted_evidence_list

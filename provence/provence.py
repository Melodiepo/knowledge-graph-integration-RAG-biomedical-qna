from transformers import AutoModel
import json
import argparse
import nltk
import os
import torch
nltk.download('punkt_tab')

CACHE_DIR = "/cs/student/projects1/ml/2024/yihanli/hf_cache"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR 
os.environ["TMPDIR"] = "/cs/student/projects1/ml/2024/yihanli/tmp"

def get_document_text(item: dict) -> str:
    """
    Given one item from retriever output, returns the raw string to feed to the Provence model.
    - If there's a 'corpus' key (CPG or textbook), use 'text'
    - Otherwise, assume it's PubMed: use 'abstract'
    """
    if "corpus" in item:
        # This is from CPG or textbook or some other corpus
        return item.get("text", "")
    else:
        # This is presumably PubMed
        return item.get("abstract", "")

provence = AutoModel.from_pretrained(
    "naver/provence-reranker-debertav3-v1",
    cache_dir=CACHE_DIR,
    trust_remote_code=True
).to(device).eval()


def filter_provence_on_retriever_output(
    query: str,
    retrieved_list: list,
    model_name: str = "naver/provence-reranker-debertav3-v1",
    always_select_title: bool = True,
    threshold: float = 0.01
) -> list:
    """
    Prunes the context from the retrieved list of documents/snippets using the Provence model.
    
    Args:
        query (str): The user's query/question.
        retrieved_list (list): The JSON objects from the retriever output
                               (each dict has pmid/title/abstract or corpus/text).
        model_name (str): HF model name for the Provence pruner.
        always_select_title (bool): Whether to always include the first sentence from doc_text.
        threshold (float): Pruning threshold (lower => keep more text).
    
    Returns:
        list: A list of dicts. Each dict has:
          {
            "id": str or None,
            "pruned_context": str,
            "score": float,
            "corpus": str or None
          }
        sorted by descending score.
    """

    pruned_results = []

    for item in retrieved_list:
        doc_text = get_document_text(item)
        if not doc_text.strip():
            continue  # Skip empty text

        # ID or PMID or chunk_index, etc. — whichever you prefer
        doc_id = item.get("pmid") or item.get("chunk_index") or None
        corpus = item.get("corpus", None)

        # ---------------
        #   PROVENCE
        # ---------------
        # The snippet below assumes your provence model has a `.process()` method
        # that returns {"pruned_context": "...", "reranking_score": ...}.
        # Adapt as needed based on your actual Provence code.
        result = provence.process(
            query,
            doc_text,
            always_select_title=always_select_title,
            threshold=threshold
        )

        pruned_context = result.get("pruned_context", doc_text)
        reranking_score = result.get("reranking_score", 0.0)

        pruned_results.append({
            "id": str(doc_id),
            "corpus": corpus,
            "pruned_context": pruned_context,
            "score": float(reranking_score)
        })

    # Sort in descending order by score:
    sorted_results = sorted(pruned_results, key=lambda x: x["score"], reverse=True)
    return sorted_results



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--retriever_output", required=True, 
                        help="Path to the JSON from your retriever step.")
    parser.add_argument("--output_path", required=True,
                        help="Path to the final JSON that Provence will write.")
    parser.add_argument("--model_name", default="naver/provence-reranker-debertav3-v1",
                        help="Which Provence model to use.")
    args = parser.parse_args()
    
    # Check for a checkpoint file (use output_path + ".ckpt")
    checkpoint_path = "/cs/student/projects1/ml/2024/yihanli/provence/output/provence_mmlu_cot_ckpt.json"
    if os.path.exists(checkpoint_path) and os.path.getsize(checkpoint_path) > 0:
        try:
            with open(checkpoint_path, "r") as cp:
                final_dict = json.load(cp)
            print(f"Loaded checkpoint with {len(final_dict)} processed queries.")
        except json.JSONDecodeError:
            print("Checkpoint file is corrupt or empty. Reinitializing checkpoint.")
            final_dict = {}
    else:
        final_dict = {}
    
    # 1) Load the retriever output (a list of queries)
    with open(args.retriever_output, "r") as f:
        retriever_data = json.load(f)
    
    total_queries = len(retriever_data)
    
    # 2) Process each query
    for i, entry in enumerate(retriever_data):
        # If this query has already been processed, skip it
        if str(i) in final_dict:
            continue

        try:
            print(f"[{i + 1}/{total_queries}] Processing query...")
            q = entry["query"]
            retrieved = entry.get("retrieved", [])
            
            # Filter using the Provence-based function
            filtered_docs = filter_provence_on_retriever_output(
                query=q,
                retrieved_list=retrieved,
                model_name=args.model_name
            )
            
            # Save the result for this query in the checkpoint dictionary
            final_dict[str(i)] = {
                "query": q,
                "filtered_docs": filtered_docs
            }
        except Exception as e:
            # Log error and store an error entry so you don't reprocess this query endlessly
            print(f"Error processing query {i}: {e}")
            final_dict[str(i)] = {
                "query": entry.get("query", ""),
                "filtered_docs": [],
                "error": str(e)
            }
        
        # Write out checkpoint after each query
        with open(checkpoint_path, "w") as cp:
            json.dump(final_dict, cp, indent=2)
        print(f"Checkpoint updated: {i + 1} queries processed.")

    # 3) Write out final JSON once all queries have been processed
    with open(args.output_path, "w") as out_f:
        json.dump(final_dict, out_f, indent=2)
    print(f"Done. Wrote Provence-pruned JSON to {args.output_path}")


if __name__ == "__main__":
    main()
    
    
"""
Command (run):
setenv TRANSFORMERS_CACHE /cs/student/projects1/ml/2024/yihanli/hf_cache
setenv HF_HOME /cs/student/projects1/ml/2024/yihanli/hf_cache

python /cs/student/projects1/ml/2024/yihanli/provence/provence.py \
--retriever_output /cs/student/projects1/ml/2024/yihanli/retriever/output/evidence_medmcqa_cot_test_k_10.json \
--output_path /cs/student/projects1/ml/2024/yihanli/provence/output/provence_medmcqa_cot_output.json \
--model_name "naver/provence-reranker-debertav3-v1"

Command (convert):
jq 'to_entries | map({key: .key, value: {query: .value.query, filtered_docs: [.value.filtered_docs[]?.pruned_context]}}) | from_entries' /cs/student/projects1/ml/2024/yihanli/provence/output/provence_medmcqa_cot_output.json > /cs/student/projects1/ml/2024/yihanli/provence/output/provence_medmcqa_cot_output_new.json

Command (str to dict):
python3 -c 'import json; f = "/cs/student/projects1/ml/2024/yihanli/provence/output/provence_medmcqa_cot_output_new.json"; o = "/cs/student/projects1/ml/2024/yihanli/provence/output/provence_medmcqa_cot_output_new.json"; d = json.load(open(f)); [d[k].update({"filtered_docs": [{"text": s} for s in d[k]["filtered_docs"] if s.strip()]}) for k in d]; json.dump(d, open(o, "w"), indent=2)'

Command (generate):
python /cs/student/projects1/ml/2024/yihanli/provence/generator_medmcqa.py \
--input_questions /cs/student/projects1/ml/2024/yihanli/retriever/input/medmcqa/dev.json \
--filtered_evidence /cs/student/projects1/ml/2024/yihanli/provence/output/provence_medmcqa_cot_output_new.json \
--output_path /cs/student/projects1/ml/2024/yihanli/provence/generated_answers_medmcqa_cot.json

Command (eval):
python /cs/student/projects1/ml/2024/yihanli/uandreic/rag2-generation/metrics_new.py \
--model_json /cs/student/projects1/ml/2024/yihanli/provence/generated_answers_mmlu_cot.json \
--ground_truth_json /cs/student/projects1/ml/2024/yihanli/retriever/input/MMLUmed_test/mmlumed_test_final.json

"""    

"""   
# Optional: A small demonstration if you run `python provence.py` directly
if __name__ == "__main__":
    # Example query
    example_query = "A 67-year-old man with transitional cell carcinoma of the bladder comes to the physician because of a 2-day history of ringing sensation in his ear. He received this first course of neoadjuvant chemotherapy 1 week ago. Pure tone audiometry shows a sensorineural hearing loss of 45 dB. The expected beneficial effect of the drug that caused this patient's symptoms is most likely due to which of the following actions?"

    # Example "retrieved" data
    example_retrieved = [
      {
        "pmid": "7265913",
        "chunk_index": 7,
        "abstract": "",
        "confidence": 3.3670499324798584
      },
      {
        "pmid": "11613547",
        "chunk_index": 11,
        "abstract": "",
        "confidence": 3.274282217025757
      },
      {
        "pmid": "11961156",
        "chunk_index": 11,
        "abstract": "",
        "confidence": 3.2668306827545166
      },
      {
        "pmid": "3639365",
        "chunk_index": 3,
        "abstract": "",
        "confidence": 3.1540493965148926
      },
      {
        "text": "complications of include , , and . - Prognosis is generally , and the of patients with is approximately . # Diagnosis ## Diagnostic Criteria - The diagnosis of is made when at least of the following diagnostic criteria are met: ## Symptoms - is usually asymptomatic. - Symptoms of may include the following: ## Physical Examination - Patients with usually appear . - Physical examination may be remarkable for: ## Laboratory Findings - There are no specific laboratory findings associated with . - A is diagnostic of . - An concentration of is diagnostic of . - Other laboratory findings consistent with the diagnosis of include , , and . ## Imaging Findings - There are no findings associated with . - is the imaging modality of",
        "corpus": "CPG",
        "confidence": 0.8200969696044922
      },
      {
        "text": "be heterogeneous. Antisera are valuable for many biological purposes but they have certain inherent disadvantages that relate to the heterogeneity of the antibodies they contain. First, each antiserum is different from all other antisera, even if raised in a genetically identical animal by using the identical preparation of antigen and the same immunization protocol. Second, antisera can be produced in only limited volumes, and thus it is impossible to use the identical serological reagent in a long or complex series of experiments or clinical tests. Finally, even antibodies purified by affinity chromatography (see Section A-3) can include minor populations of antibodies that give unexpected cross-reactions, which confound the analysis of experiments. To avoid these problems, and to harness the full potential of antibodies, it was necessary to develop",
        "corpus": "Textbook",
        "confidence": -1.475448727607727
      },
      {
        "text": "is a reasonable surrogate for breast cancer prevention because these are second primaries not recurrences. In this regard, the aromatase inhibitors are all considerably more effective than tamoxifen; however, they are not approved for primary breast cancer prevention. It remains puzzling that agents with the safety 531 profile of raloxifene, which can reduce breast cancer risk by 50% with additional benefits in preventing osteoporotic fracture, are still so infrequently prescribed. They should be far more commonly offered to women than they are. Breast cancer develops as a series of molecular changes in the epithelial cells that lead to ever more malignant behavior. Increased use of mammography has led to more frequent diagnoses of noninvasive breast cancer. These lesions fall into two groups: ductal carcinoma in situ (DCIS) and",
        "corpus": "Textbook",
        "confidence": -3.3893496990203857
      },
      {
        "text": "around 5 kilo-base pairs(kbp) in length. Thirteen splice variants are supported by ACEVIEW analysis but only two have been experimentally identified. Mostly, different variants seem to vary based on differing truncation of the 3' and 5' ends (especially due to the presence of an upstream stop codon in the exonic region). # Protein Protein TMEM131L is an integral membrane protein and is also known as OTTHUMP00000205136. The protein TMEM131L Isoform I is 1,610 amino acids in length and its primary structure weighs 179.209 kilo-Dalton (unit) (kDa). Twelve different isoforms of this protein have been predicted (one partial, six COOH complete, and five complete) however there have been only 5 experimentally observed. ## Expression The protein TMEM131L shows highest levels of expression in immune related cells and tissues such",
        "corpus": "CPG",
        "confidence": -3.786402702331543
      },
      {
        "text": "illnesses, inflammatory conditions, pregnancy, and medications affect levels of many coagulation factors and their inhibitors. Antithrombin is decreased by heparin and in the setting of acute thrombosis. Protein C and S levels may be increased in the setting of acute thrombosis and are decreased by warfarin. Antiphospholipid antibodies are frequently transiently positive in acute illness. Testing for genetic thrombophilias should, in general, only be performed when there is a strong family history of thrombosis and results would affect clinical decision making. Because thrombophilia evaluations are usually performed to assess the need to extend anticoagulation, testing should be performed in a steady state, remote from the acute event. In most instances, warfarin anticoagulation can be stopped after the initial 3\u20136 months of treatment, and testing can be performed at",
        "corpus": "Textbook",
        "confidence": -4.2807793617248535
      },
      {
        "text": "membrane. FIguRE 80-3 Neutrophil band with D\u00f6hle body. The neutrophil with a sausage-shaped nucleus in the center of the field is a band form. D\u00f6hle bodies are discrete, blue-staining, nongranular areas found in the periphery of the cytoplasm of the neutrophil in infec-tions and other toxic states. They represent aggregates of rough endoplasmic reticulum. FIguRE 80-4 Normal granulocyte. The normal granulocyte has a segmented nucleus with heavy, clumped chromatin; fine neutrophilic granules are dispersed throughout the cytoplasm. Neutrophils are heterogeneous in function. Monoclonal antibodies have been developed that recognize only a subset of mature neutrophils. The meaning of neutrophil heterogeneity is not known. The morphology of eosinophils and basophils is shown in Fig. 80-6. Specific signals, including IL-1, tumor necrosis factor \u03b1 (TNF-\u03b1), the CSFs, complement fragments, and",
        "corpus": "Textbook",
        "confidence": -4.354891300201416
      }
    ]

    # Run filtering
    pruned_docs = filter_provence_on_retriever_output(
        query=example_query,
        retrieved_list=example_retrieved,
        model_name="naver/provence-reranker-debertav3-v1",
        always_select_title=True,
        threshold=0
    )

    # Print the pruned contexts
    for doc in pruned_docs:
        print(f"ID: {doc['id']} (corpus={doc['corpus']})")
        print(f"Score: {doc['score']}")
        print(f"Pruned Context: {doc['pruned_context']}")
        print("-"*80)
"""

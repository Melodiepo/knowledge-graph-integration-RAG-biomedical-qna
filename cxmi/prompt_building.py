import os
import json
import re
import torch
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import gc

def build_prompt(question_text, docs, options_dict):
    """
    Creates a full prompt, including:
      - Some context from docs
      - The question
      - The available multiple-choice options
      - A directive to produce a thorough explanation + pick an option
    """
    doc_context = "\n".join(doc for doc in docs) if docs else "No relevant documents."

    # Convert the options dict into lines: "(A) answer text", etc.
    if options_dict:
        options_str = "\n".join(f"({k}) {v}" for k, v in options_dict.items())
    else:
        options_str = "No multiple-choice options provided."

    prompt = (
        "You are a helpful medical assistant.\n\n"
        "Below is some context, followed by a medical question **and** a list of possible answers. "
        "Please provide a thorough explanation, then select the best answer.\n\n"
        f"Context:\n{doc_context}\n\n"
        f"Question:\n{question_text}\n\n"
        f"Options:\n{options_str}\n\n"
        "Answer (explain your reasoning and pick one option):"
    )
    return prompt

def load_file(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    return data


def main(filtered_docs_dir, answers_dir, output_path):
    filtered_data = load_file(filtered_docs_dir)
    answers_data = load_file(answers_dir)

    all_prompts = []
    # seen_queries = set()
    for idx, content in tqdm(enumerate(answers_data), total=len(answers_data), desc="Prompt Generating"):
        query = content.get("question", "")
        if not query:
            raise ValueError(f"query is empty for idx {idx}")
        
        # if query in seen_queries:
        #     continue
        # else:
        #     seen_queries.add(query)
        retrieved_dict = filtered_data.get(str(idx), {})
        # if query != retrieved_dict.get("query", ""):
        #     raise ValueError(f"Mismatched question at idx {idx}:\nExpected: {retrieved_dict.get('query', '')}\nGot: {query}")


        retrieved_docs = retrieved_dict.get("retrieved_docs", [])
        options_dict = content.get("options", {})
        answer = content.get("answer", "")

        prompt = build_prompt(query, retrieved_docs, options_dict)
        all_prompts.append({
            "question": query,
            "options": options_dict,
            "prompt": prompt
        })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_prompts, f, indent=4)
    
    print(f"Labeled data saved to {output_path}")

if __name__=="__main__":

    ANSWER_PATH = "/cs/student/projects1/ml/2024/yihanli/retriever/input/MMLUmed_test/mmlumed_test_final.json"

    sent_list = ["passage", "sent"]
    flant5_list = ["base", "large"]
    for sent_pas in sent_list:
        for flant5 in flant5_list:
            FILTERED_DOC_PATH = f"/cs/student/projects1/ml/2024/yihanli/filco/output/mmlu_cot/cxmi_{sent_pas}_pruning_{flant5}_k_10.json"
            OUTPUT_PATH = f"/cs/student/projects1/ml/2024/yihanli/filco/output/mmlu_cot/cxmi_{sent_pas}_prompt_{flant5}_k_10.json"
            main(FILTERED_DOC_PATH, ANSWER_PATH, OUTPUT_PATH)
            gc.collect()

    
    FILTERED_DOC_PATH = "/cs/student/projects1/ml/2024/yihanli/similarity_pruning/output/mmlu_cot/similarity_pruning_test_k_10.json"
    OUTPUT_PATH = "/cs/student/projects1/ml/2024/yihanli/similarity_pruning/output/mmlu_cot/similarity_prompt_test_k_10.json"
    main(FILTERED_DOC_PATH, ANSWER_PATH, OUTPUT_PATH)
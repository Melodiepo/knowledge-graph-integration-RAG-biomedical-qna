import os
import json
import re
import torch
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_filtered_evidence(filtered_path):
    """Load the filtered evidence JSON produced by your filtering step."""
    with open(filtered_path, "r") as f:
        return json.load(f)

def load_questions(path):
    """Load the multiple-choice questions from a JSON file."""
    with open(path, "r") as f:
        return json.load(f)

def build_prompt(question_text, docs, options_dict):
    """
    Creates a full prompt, including:
      - Some context from docs
      - The question
      - The available multiple-choice options
      - A directive to produce a thorough explanation + pick an option
    """
    doc_context = "\n".join(doc["text"] for doc in docs) if docs else "No relevant documents."

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_questions", 
        default="/cs/student/projects1/ml/2024/yihanli/retriever/input/MMLUmed_test/mmlumed_test_final.json",
        help="Path to the JSON file of multiple-choice questions"
    )
    parser.add_argument("--filtered_evidence", 
        default="/cs/student/projects1/ml/2024/yihanli/uandreic/rag2-filtering/output/filtered_evidence_mmlumed_test_k_10.json",
        help="Path to the JSON file of filtered evidence from the pipeline"
    )
    parser.add_argument("--model_name", 
        default="BioMistral/BioMistral-7B",
        #default="meta-llama/Meta-Llama-3-8B",
        help="Model identifier for your huggingface model"
    )
    parser.add_argument("--output_path", 
        default="/cs/student/projects1/ml/2024/yihanli/uandreic/rag2-generation/output/generated_answers.json",
        help="Where to store the final answers"
    )
    parser.add_argument("--max_new_tokens", type=int, default=512,
        help="Maximum tokens to generate for a longer answer."
    )
    parser.add_argument("--min_new_tokens", type=int, default=50,
        help="Minimum tokens so the model doesn't stop immediately."
    )
    parser.add_argument("--do_sample", action="store_true",
        help="If set, use sampling. Otherwise do deterministic generation."
    )
    args = parser.parse_args()

    # 1) Load data
    questions = load_questions(args.input_questions)
    filtered_data = load_filtered_evidence(args.filtered_evidence)

    # 2) Initialize the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)  
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    # Ensure we have a pad token ID if the model doesn't
    if model.config.pad_token_id is None:
        model.config.pad_token_id = model.config.eos_token_id

    model.eval()

    # 3) Generate answers
    results = []
    for idx, q in tqdm(enumerate(questions), total=len(questions), desc="Generating"):
        question_text = q["question"]
        options_dict = q.get("options", {})

        # Retrieve docs from your filtering step
        docs_entry = filtered_data.get(str(idx), {})
        docs = docs_entry.get("filtered_docs", [])

        # Build the prompt
        prompt = build_prompt(question_text, docs, options_dict)

        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)

        # Generation arguments
        generation_args = dict(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            min_new_tokens=args.min_new_tokens,
            repetition_penalty=1.1
        )

        # If user wants sampling, set do_sample + top_p + temperature
        if args.do_sample:
            generation_args.update(
                do_sample=True,
                top_p=0.9,
                temperature=0.8
            )
        else:
            # Deterministic
            generation_args.update(
                do_sample=False,
                num_beams=1
            )

        # 4) Generate output
        with torch.no_grad():
            output_ids = model.generate(**generation_args)

        # Only decode newly generated tokens
        new_tokens_start = inputs["input_ids"].shape[1]
        output_text = tokenizer.decode(
            output_ids[0][new_tokens_start:], 
            skip_special_tokens=True
        )

        # Store the entire generation
        results.append({
            "question_idx": idx,
            "question": question_text,
            "options": options_dict,
            "model_output": output_text
        })

    # 5) Write final answers
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved final answers to {args.output_path}")

if __name__ == "__main__":
    main()
import os
import json
import torch
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# Map numeric answer index (cop) to letter
INDEX_TO_LETTER = {1: "A", 2: "B", 3: "C", 4: "D"}

def load_ground_truth_questions(path):
    """Load MMLUmed-style JSON with ground-truth and convert to usable format."""
    with open(path, "r") as f:
        raw_data = json.load(f)

    questions = []
    for entry in raw_data:
        options = {
            "A": entry.get("opa", ""),
            "B": entry.get("opb", ""),
            "C": entry.get("opc", ""),
            "D": entry.get("opd", "")
        }
        questions.append({
            "id": entry.get("id"),
            "question": entry["question"],
            "options": options,
            "ground_truth": INDEX_TO_LETTER.get(entry.get("cop"))
        })
    return questions

def load_filtered_evidence(filtered_path):
    with open(filtered_path, "r") as f:
        return json.load(f)

def build_prompt(question_text, docs, options_dict):
    doc_context = "\n".join(doc["text"] for doc in docs) if docs else "No relevant documents."
    options_str = "\n".join(f"({k}) {v}" for k, v in options_dict.items()) if options_dict else "No multiple-choice options provided."

    prompt = (
        "You are a helpful medical assistant.\n\n"
        "Below is some context, followed by a medical question and possible answers. "
        "Please provide a thorough explanation, then select the best answer.\n\n"
        f"Context:\n{doc_context}\n\n"
        f"Question:\n{question_text}\n\n"
        f"Options:\n{options_str}\n\n"
        "Answer (explain your reasoning and pick one option):"
    )
    return prompt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_questions", default = "/cs/student/projects1/ml/2024/yihanli/retriever/input/medmcqa/dev.json")
    parser.add_argument("--filtered_evidence", default = "/cs/student/projects1/ml/2024/yihanli/uandreic/rag2-filtering/output/filtered_evidence_medmcqa_test_k_10.json")
    parser.add_argument("--model_name", default="BioMistral/BioMistral-7B")
    parser.add_argument("--output_path", default="/cs/student/projects1/ml/2024/yihanli/uandreic/rag2-generation/output/generated_answers.json")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--min_new_tokens", type=int, default=50)
    parser.add_argument("--do_sample", action="store_true")
    args = parser.parse_args()

    questions = load_ground_truth_questions(args.input_questions)
    filtered_data = load_filtered_evidence(args.filtered_evidence)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    if model.config.pad_token_id is None:
        model.config.pad_token_id = model.config.eos_token_id
    model.eval()

    results = []
    for idx, q in tqdm(enumerate(questions), total=len(questions), desc="Generating"):
        question_text = q["question"]
        options_dict = q["options"]
        ground_truth = q["ground_truth"]
        question_id = q["id"]

        docs_entry = filtered_data.get(str(idx), {})
        docs = docs_entry.get("filtered_docs", [])

        prompt = build_prompt(question_text, docs, options_dict)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)

        gen_args = {
            **inputs,
            "max_new_tokens": args.max_new_tokens,
            "min_new_tokens": args.min_new_tokens,
            "repetition_penalty": 1.1
        }

        if args.do_sample:
            gen_args.update({"do_sample": True, "top_p": 0.9, "temperature": 0.8})
        else:
            gen_args.update({"do_sample": False, "num_beams": 1})

        with torch.no_grad():
            output_ids = model.generate(**gen_args)

        new_tokens_start = inputs["input_ids"].shape[1]
        output_text = tokenizer.decode(output_ids[0][new_tokens_start:], skip_special_tokens=True)

        results.append({
            "question_idx": idx,
            "id": question_id,
            "question": question_text,
            "options": options_dict,
            "ground_truth": ground_truth,
            "model_output": output_text
        })

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved final answers to {args.output_path}")

if __name__ == "__main__":
    main()
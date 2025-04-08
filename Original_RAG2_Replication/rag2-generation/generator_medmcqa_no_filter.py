import os
import json
import torch
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# Mapping from integer answer index (cop) to letter
INDEX_TO_LETTER = {1: "A", 2: "B", 3: "C", 4: "D"}

def load_questions(path):
    """Load MMLUmed-style questions and reformat for generation."""
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

def get_document_text(doc):
    """Extract plain text from either 'text' or 'abstract' field."""
    if "text" in doc and doc["text"].strip():
        return doc["text"].strip()
    elif "abstract" in doc and doc["abstract"].strip():
        return doc["abstract"].strip()
    return ""

def build_prompt(question_text, docs, options_dict):
    doc_context = "\n".join(get_document_text(doc) for doc in docs if get_document_text(doc)) or "No relevant documents."
    options_str = "\n".join(f"({k}) {v}" for k, v in options_dict.items()) if options_dict else "No options provided."

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
    parser.add_argument("--retrieved_evidence", default = "/cs/student/projects1/ml/2024/yihanli/retriever/output/evidence_medmcqa_test_k_10.json")
    parser.add_argument("--model_name", default="BioMistral/BioMistral-7B", help="HF model name or local path")
    parser.add_argument("--output_path", default="/cs/student/projects1/ml/2024/yihanli/uandreic/rag2-generation/output/generated_answers_no_filter.json")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--min_new_tokens", type=int, default=50)
    parser.add_argument("--do_sample", action="store_true")
    args = parser.parse_args()

    # Load questions and evidence
    questions = load_questions(args.input_questions)
    with open(args.retrieved_evidence, "r") as f:
        retrieved_data = json.load(f)

    # Load model + tokenizer
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

    # Generate answers
    results = []
    for idx, q in tqdm(enumerate(questions), total=len(questions), desc="Generating"):
        question_text = q["question"]
        options_dict = q["options"]
        ground_truth = q["ground_truth"]
        question_id = q["id"]

        # Top-k retrieved docs
        docs = retrieved_data[idx]["retrieved"][:10]
        prompt = build_prompt(question_text, docs, options_dict)

        # Tokenize + inference
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
            torch.cuda.empty_cache()
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
import os
import json
import torch
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_questions(path):
    with open(path, "r") as f:
        return json.load(f)

def get_document_text(doc):
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
    parser.add_argument("--input_questions", 
        default="/cs/student/projects1/ml/2024/yihanli/retriever/input/MMLUmed_test/mmlumed_test_final.json")
    parser.add_argument("--retrieved_evidence", 
        default="/cs/student/projects1/ml/2024/yihanli/retriever/output/evidence_mmlumed_test_k_10.json")
    parser.add_argument("--model_name", 
        default="BioMistral/BioMistral-7B")
    parser.add_argument("--output_path", 
        default="/cs/student/projects1/ml/2024/yihanli/uandreic/rag2-generation/output/generated_answers_no_filter.json")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--min_new_tokens", type=int, default=50)
    parser.add_argument("--do_sample", action="store_true")
    args = parser.parse_args()

    questions = load_questions(args.input_questions)
    with open(args.retrieved_evidence, "r") as f:
        retrieved_data = json.load(f)

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
        options_dict = q.get("options", {})
        docs = retrieved_data[idx]["retrieved"][:10]  # Limit to top 10
        prompt = build_prompt(question_text, docs, options_dict)

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)

        generation_args = dict(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            min_new_tokens=args.min_new_tokens,
            repetition_penalty=1.1
        )

        if args.do_sample:
            generation_args.update(do_sample=True, top_p=0.9, temperature=0.8)
        else:
            generation_args.update(do_sample=False, num_beams=1)

        with torch.no_grad():
            torch.cuda.empty_cache()
            output_ids = model.generate(**generation_args)

        new_tokens_start = inputs["input_ids"].shape[1]
        output_text = tokenizer.decode(output_ids[0][new_tokens_start:], skip_special_tokens=True)

        results.append({
            "question_idx": idx,
            "question": question_text,
            "options": options_dict,
            "model_output": output_text
        })

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved final answers to {args.output_path}")

if __name__ == "__main__":
    main()

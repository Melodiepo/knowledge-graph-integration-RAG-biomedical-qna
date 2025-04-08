import os
import json
import torch
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

"""
Command: python cot_generation.py --input_questions /cs/student/projects1/ml/2024/yihanli/retriever/input/medmcqa/dev.json --model_name "dmis-lab/meerkat-7b-v1.0" --output_path /cs/student/projects1/ml/2024/yihanli/retriever/input/input_cot/medmcqa_cot.json --do_sample --max_new_tokens 512 --min_new_tokens 50

"""

def load_questions(path):
    """Load MCQ queries from a JSON file."""
    with open(path, "r") as f:
        return json.load(f)

def build_cot_prompt(initial_query):
    """
    Builds a Chain-of-Thought (CoT) prompt using the provided format.
    """
    prompt = (
        "The following are multiple choice questions about medical knowledge.\n"
        "Solve them in a step-by-step fashion. First, summarize the question. "
        "Then, reason step-by-step through the possible answers. Finally, "
        "explicitly return the correct answer in the format:\n\n"
        "**Final Answer: (X)**\n\n"
        f"Here is the question: {initial_query}\n"
    )
    return prompt

def load_existing_results(output_path):
    """Load existing results to resume from last processed query."""
    if os.path.exists(output_path): 
        with open(output_path, "r") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return []  # If file is corrupted, start fresh
    return []

def main():
    parser = argparse.ArgumentParser(description="Generate Chain-of-Thought expansions for MCQ questions.")
    # Remove or simplify defaults so you can pass them in from command line
    parser.add_argument("--input_questions", required=True,
        help="Path to the JSON file of multiple-choice questions"
    )
    parser.add_argument("--model_name", 
        default="dmis-lab/meerkat-7b-v1.0", # first trial: BioMistral
        help="Hugging Face model identifier for CoT generation"
    )
    parser.add_argument("--output_path", required=True,
        help="Where to store the final CoT-augmented JSON"
    )
    parser.add_argument("--max_new_tokens", type=int, default=512,
        help="Maximum tokens to generate for a longer explanation."
    )
    parser.add_argument("--min_new_tokens", type=int, default=50,
        help="Minimum tokens so the model doesn't stop too soon."
    )
    parser.add_argument("--do_sample", action="store_true",
        help="If set, use sampling. Otherwise, use greedy or beam search."
    )
    args = parser.parse_args()

    # 1) Load data
    questions = load_questions(args.input_questions)
    
    # 2️⃣ Load existing results and resume from last processed index
    existing_results = load_existing_results(args.output_path)
    processed_indices = {entry["question_idx"] for entry in existing_results}
    start_index = max(processed_indices, default=-1) + 1  # Start from next query

    if start_index >= len(questions):
        print(f"✅ All {len(questions)} queries are already processed. No work needed.")
        return

    print(f"🔄 Resuming from query {start_index}...")

    # 3) Initialize the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Bistral: tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False) 
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, 
        use_fast=True, 
        trust_remote_code=True 
    )   
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
        # BioMistral: offload_folder="offload_weights"
    )

    if model.config.pad_token_id is None:
        model.config.pad_token_id = model.config.eos_token_id

    model.eval()

    # 3) Generate Chain-of-Thought expansions
    results = existing_results  # Append to existing results
    for idx, q in tqdm(enumerate(questions), total=len(questions), desc="Generating CoT Responses"):
        question_text = q["question"]

        # Build the CoT-style prompt
        prompt = build_cot_prompt(question_text)

        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)

        # Set generation arguments
        generation_args = dict(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            min_new_tokens=args.min_new_tokens,
            repetition_penalty=1.1
        )

        if args.do_sample:
            generation_args.update(
                do_sample=True,
                top_p=0.9,
                temperature=0.8
            )
        else:
            generation_args.update(
                do_sample=False,
                num_beams=1
            )

        # Generate response
        with torch.no_grad():
            output_ids = model.generate(**generation_args)

        # Extract generated text
        new_tokens_start = inputs["input_ids"].shape[1]
        output_text = tokenizer.decode(
            output_ids[0][new_tokens_start:], 
            skip_special_tokens=True
        )

        # Store results
        results.append({
            "question_idx": idx,
            "question": question_text,
            "model_output": output_text
        })
        
        # Save partial progress every 10 queries
        if (idx + 1) % 10 == 0: 
            with open(args.output_path, "w") as f:
                json.dump(results, f, indent=2)

    # 4) Save final CoT expansions
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"✅ Saved Chain-of-Thought expansions to {args.output_path}")

if __name__ == "__main__":
    main()

import os
import json
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

def load_filtered_evidence(filtered_evidence_path):
    """Load the filtered evidence JSON."""
    with open(filtered_evidence_path, 'r') as file:
        return json.load(file)

def load_questions(input_questions_path):
    """Load the multiple-choice questions."""
    with open(input_questions_path, 'r') as file:
        return json.load(file)

def initialize_llm(model_name):
    """Initialize the language model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return tokenizer, model

def format_prompt(question, options, retrieved_docs):
    """Format the prompt for the LLM."""
    doc_texts = "\n".join([doc["text"] for doc in retrieved_docs]) if retrieved_docs else "No relevant documents found."
    options_text = " ".join([f"({key}) {value}" for key, value in options.items()])
    prompt = f"""
    These are the relevant information:
    {doc_texts}

    Using these, answer this question:
    {question}

    Your options for the answer are:
    {options_text}

    Your answer should only be in the form of (A) or (B) or (C) or (D)."""
    return prompt.strip()

def generate_answer(model, tokenizer, prompt):
    """Generate an answer using the LLM."""
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=10)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer.strip()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-q', '--input_questions', default = '/cs/student/projects1/ml/2024/yihanli/retriever/input/medqa/medqa_llama_cot.json',  help='Path to the input questions JSON file')
    parser.add_argument('-f', '--filtered_evidence', default = '/cs/student/projects1/ml/2024/yihanli/uandreic/rag2-filtering/output/filtered_evidence_medqa_llama_cot.json',  help='Path to the filtered evidence JSON file')
    parser.add_argument('-m', '--model_name', default='facebook/opt-1.3b', help='Pretrained LLM model')
    parser.add_argument('-o', '--output_path', default='/cs/student/projects1/ml/2024/yihanli/uandreic/rag2-generation/output/generated_answers.json', help='Path to store generated answers')
    args = parser.parse_args()
    
    # Load data
    questions = load_questions(args.input_questions)
    filtered_evidence = load_filtered_evidence(args.filtered_evidence)
    tokenizer, model = initialize_llm(args.model_name)
    
    generated_answers = []
    
    for idx, q in tqdm(enumerate(questions), total=len(questions)):
        query = q["question"]
        options = q["options"]
        retrieved_docs = filtered_evidence.get(str(idx), {}).get("filtered_docs", [])
        prompt = format_prompt(query, options, retrieved_docs)
        answer = generate_answer(model, tokenizer, prompt)
        generated_answers.append({
            "question": query,
            "predicted_answer": answer,
            "answer_idx": q.get("answer_idx"),
            "options": options,
        })
    
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, 'w') as file:
        json.dump(generated_answers, file, indent=4)
    
    print(f"Generated answers saved to {args.output_path}")

if __name__ == "__main__":
    main()
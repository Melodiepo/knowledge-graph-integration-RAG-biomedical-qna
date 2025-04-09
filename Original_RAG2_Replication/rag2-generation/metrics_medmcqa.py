import re
import json
import argparse

# Convert integer label to corresponding letter
INDEX_TO_LETTER = {1: "A", 2: "B", 3: "C", 4: "D", 5: "E"}

def extract_choice(text):
    """
    Attempt to extract the selected multiple-choice option (A-E) from model output.
    Returns the choice letter, or None if not found.
    """
    text = text.strip().upper()

    # Try to find patterns like "Option A", "(A)", "A.", "Answer: A"
    match = re.search(r"(OPTION\s*)?[\(\[]?([A-E])[\)\].: ]", text)
    if match:
        return match.group(2)
    
    # Fallback: check if first character is a valid choice
    if text and text[0] in "ABCDE":
        return text[0]

    return None

def evaluate_mmlumed(model_json_path, ground_truth_json_path):
    """
    Evaluate accuracy for MMLUmed-style data where ground truth is stored as "cop": 1..4
    and model outputs are decoded free-text answers.
    
    Both files are expected to be lists of entries with aligned indices.
    """
    with open(model_json_path, 'r', encoding='utf-8') as f1, open(ground_truth_json_path, 'r', encoding='utf-8') as f2:
        model_data = json.load(f1)
        ground_truth_data = json.load(f2)

    if len(model_data) != len(ground_truth_data):
        print(f"Warning: Length mismatch — {len(model_data)} predictions vs {len(ground_truth_data)} ground truth entries.")

    correct = 0
    total = len(model_data)

    for idx, model_item in enumerate(model_data):
        model_answer = extract_choice(model_item["model_output"])
        true_index = ground_truth_data[idx].get("cop")  # 1 = A, 2 = B, etc.
        true_answer = INDEX_TO_LETTER.get(true_index)

        if model_answer and true_answer and model_answer == true_answer:
            correct += 1

    accuracy = correct / total if total > 0 else 0.0
    print(f"Model Accuracy: {accuracy:.2%} ({correct}/{total})")
    return accuracy

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate MMLUmed-style Model Accuracy")
    parser.add_argument("--model_json", default = " /cs/student/projects1/ml/2024/yihanli/uandreic/rag2-generation/output/generated_answers.json")
    parser.add_argument("--ground_truth_json", default = "/cs/student/projects1/ml/2024/yihanli/retriever/input/medmcqa/dev.json")

    args = parser.parse_args()
    evaluate_mmlumed(args.model_json, args.ground_truth_json)
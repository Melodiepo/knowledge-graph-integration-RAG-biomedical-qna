import re
import json
import argparse
import gc


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


def evaluate_medqa(model_json, ground_truth_json):
    """
    Evaluate MedQA model accuracy by comparing model output to ground truth.
    Both inputs are JSON files with the following format:
    [
        {
            "question_idx": int,
            "model_output": str
        },
        ...
    ]
    """
    with open(model_json, 'r', encoding='utf-8') as predicted, open(ground_truth_json, 'r', encoding='utf-8') as gt:
        model_data, ground_truth_data = json.load(predicted), json.load(gt)

    ground_truth_dict = {idx: item["answer_idx"] for idx, item in
                         enumerate(ground_truth_data)}  # Map question_idx to answer_idx

    correct = 0
    total = len(model_data)

    for item in model_data:
        # Extract choice from model output and compare to ground truth
        model_answer = extract_choice(item["model_output"])
        true_answer = ground_truth_dict.get(item["question_idx"])
        if model_answer and model_answer == true_answer:
            correct += 1

    accuracy = correct / total if total > 0 else 0.0
    print(f"Model Accuracy: {accuracy:.2%} ({correct}/{total})")
    return accuracy


if __name__ == "__main__":

    sent_list = ["sent", "passage"]
    flant5_list = ["large", "base"]
    ground_truth_json = "/cs/student/projects1/ml/2024/yihanli/retriever/input/MMLUmed_test/mmlumed_test_final.json"
    for sent in sent_list:
        for flant5 in flant5_list:
            print(f"Evaluating with sent: {sent}, flant5: {flant5}")
            model_json = f"/cs/student/projects1/ml/2024/yihanli/filco/output/mmlu_cot/cxmi_{sent}_generation_{flant5}_k_10.json"

            evaluate_medqa(model_json, ground_truth_json)
            gc.collect()

    print("Evaluating similarity pruning")
    model_json = "/cs/student/projects1/ml/2024/yihanli/similarity_pruning/output/mmlu_cot/similarity_generation_test_k_10.json"
    evaluate_medqa(model_json, ground_truth_json)
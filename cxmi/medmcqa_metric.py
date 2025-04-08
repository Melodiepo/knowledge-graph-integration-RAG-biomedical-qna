import re
import json
import argparse


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


def evaluate_medqa(model_json):
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
    with open(model_json, 'r', encoding='utf-8') as predicted:
        model_data = json.load(predicted)

    ground_truth_dict = {idx: item["answer"] for idx, item in enumerate(model_data)}  # Map question_idx to answer_idx

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

    sent_list = ["passage", "sent"]
    flant5_list = ["base", "large"]
    for sent_pas in sent_list:
        for flant5 in flant5_list:
            model_json = f"/cs/student/projects1/ml/2024/yihanli/filco/output/medmcqa_cot/cxmi_{sent_pas}_generation_{flant5}_k_10.json"
            print(f"Evaluating with sent: {sent_pas}, flant5: {flant5}")
            evaluate_medqa(model_json)

    model_json = "/cs/student/projects1/ml/2024/yihanli/similarity_pruning/output/medmcqa_cot/similarity_generation_test_k_10.json"
    evaluate_medqa(model_json)


import os
import json
from generate_labels import PerplexityLabelGenerator
from fine_tune_flan_t5 import fine_tune_flan_t5
from filtering_module import FilteringModule

def main():
    # Define file paths
    MODEL_NAME = "google/flan-t5-base"
    ARTICLES_PATH = "/cs/student/projects1/ml/2024/yihanli/retriever/articles/pubmed/PubMed_Articles_0.json"
    INPUT_EVIDENCE_PATH = "/cs/student/projects1/ml/2024/yihanli/retriever/output/evidence_medqa_llama_cot.json"
    LABELED_DATA_PATH = "/cs/student/projects1/ml/2024/yihanli/uandreic/rag2-filtering/output/labeled_perplexity_data.json"
    OUTPUT_MODEL_DIR = "/cs/student/projects1/ml/2024/yihanli/uandreic/rag2-filtering/fine_tuned_flan_t5"
    OUTPUT_FILTERED_PATH = "/cs/student/projects1/ml/2024/yihanli/uandreic/rag2-filtering/output/filtered_evidence_medqa_llama_cot.json"

    # Step 1: Generate Labels
    print("Step 1: Generating labels...")
    label_generator = PerplexityLabelGenerator(
        model_name_or_path=MODEL_NAME,
        articles_path=ARTICLES_PATH,
        threshold=0.25
    )
    label_generator.generate_labels(INPUT_EVIDENCE_PATH, LABELED_DATA_PATH)
    print(f"Labels saved to {LABELED_DATA_PATH}")

    # Step 2: Fine-tune the Model
    print("Step 2: Fine-tuning FLAN-T5 model...")
    fine_tune_flan_t5(MODEL_NAME, LABELED_DATA_PATH, ARTICLES_PATH, OUTPUT_MODEL_DIR)
    print(f"Fine-tuned model saved to {OUTPUT_MODEL_DIR}")

    # Step 3: Filter Documents using the Fine-tuned Model
    print("Step 3: Running filtering module...")
    filtering_module = FilteringModule(
        model_name_or_path=OUTPUT_MODEL_DIR,
        articles_path=ARTICLES_PATH,
        threshold=0.25
    )
    filtering_module.run_filtering(INPUT_EVIDENCE_PATH, OUTPUT_FILTERED_PATH)
    print(f"Filtering complete. Filtered evidence saved to {OUTPUT_FILTERED_PATH}")

if __name__ == "__main__":
    main()
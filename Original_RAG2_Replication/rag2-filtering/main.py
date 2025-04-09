import os
import json
from tqdm import tqdm

# Import pipeline components from their respective modules
from generate_labels import PerplexityLabelGenerator
from fine_tune_flan_t5 import train_classifier
from filtering_module import FilteringModule
from preprocess_labeled_data import preprocess_labeled_data

def main():
    """
    Main pipeline to process evidence, generate perplexity-based labels, preprocess the labeled data,
    fine-tune the FLAN-T5 model, and finally filter the evidence using the fine-tuned model.
    
    The steps are:
        1. Generate labels for the retrieved evidence.
        1.5. Preprocess the labeled data to produce a training dataset.
        2. Fine-tune the FLAN-T5 model using the processed training data.
        3. Run the filtering module with the fine-tuned model to filter candidate documents.
    """
    # ============================
    # Define file and directory paths
    # ============================
    MODEL_NAME = "google/flan-t5-large"
    ARTICLES_DIR = "/cs/student/projects1/ml/2024/yihanli/retriever/articles/pubmed/"  # Directory containing PubMed JSON files
    INPUT_EVIDENCE_PATH = "/cs/student/projects1/ml/2024/yihanli/retriever/output/evidence_mmlumed_test_k_10.json"
    LABELED_DATA_PATH = "/cs/student/projects1/ml/2024/yihanli/uandreic/rag2-filtering/output/labeled_perplexity_data.json"
    PROCESSED_DATA_JSON = "/cs/student/projects1/ml/2024/yihanli/uandreic/rag2-filtering/output/training_data.json"
    OUTPUT_MODEL_DIR = "/cs/student/projects1/ml/2024/yihanli/uandreic/rag2-filtering/fine_tuned_flan_t5_large"
    OUTPUT_FILTERED_PATH = "/cs/student/projects1/ml/2024/yihanli/uandreic/rag2-filtering/output/filtered_evidence_medqa_llama_cot.json"

    # ============================
    # Step 1: Generate Labels
    # ============================
    print("Step 1: Generating labels...")
    label_generator = PerplexityLabelGenerator(
        model_name_or_path=MODEL_NAME,
        articles_dir=ARTICLES_DIR,
        threshold=0.25  # Adjust threshold if needed (e.g., to avoid OOM)
    )
    label_generator.generate_labels(INPUT_EVIDENCE_PATH, LABELED_DATA_PATH)
    print(f"Labeled data saved to {LABELED_DATA_PATH}")

    # ============================
    # Step 1.5: Preprocess Labeled Data
    # ============================
    print("Step 1.5: Preprocessing labeled data...")
    # Optional: Limit the number of queries and documents per query
    MAX_QUERIES = 1271
    MAX_DOCS_PER_QUERY = 5
    preprocess_labeled_data(
        labeled_data_path=LABELED_DATA_PATH,
        articles_dir=ARTICLES_DIR,
        output_path=PROCESSED_DATA_JSON,
        max_queries=MAX_QUERIES,
        max_docs_per_query=MAX_DOCS_PER_QUERY
    )
    print(f"Processed training data saved to {PROCESSED_DATA_JSON}")

    # ============================
    # Step 2: Fine-tune the Model
    # ============================

    print("Step 2: Fine-tuning FLAN-T5 model...")
    train_classifier(
        model_name=MODEL_NAME,
        data_path=PROCESSED_DATA_JSON,
        output_dir=OUTPUT_MODEL_DIR
    )
    print(f"Fine-tuned model saved to {OUTPUT_MODEL_DIR}")
    """
    if not os.path.exists(OUTPUT_MODEL_DIR):
        print("Step 2: Fine-tuning FLAN-T5 model...")
        train_classifier(
            model_name=MODEL_NAME,
            data_path=PROCESSED_DATA_JSON,
            output_dir=OUTPUT_MODEL_DIR
        )
        print(f"Fine-tuned model saved to {OUTPUT_MODEL_DIR}")
    else:
        print(f"Fine-tuned model already exists at {OUTPUT_MODEL_DIR}. Skipping fine-tuning.")
    """

    # ============================
    # Step 3: Filter Documents
    # ============================
    print("Step 3: Running filtering module...")
    # Use the fine-tuned model for filtering by providing OUTPUT_MODEL_DIR as the model path
    filtering_module = FilteringModule(
        model_name_or_path=OUTPUT_MODEL_DIR,
        articles_dir=ARTICLES_DIR,
        threshold=0.25  # You may adjust this threshold if needed
    )
    filtering_module.run_filtering(INPUT_EVIDENCE_PATH, OUTPUT_FILTERED_PATH)
    print(f"Filtering complete. Filtered evidence saved to {OUTPUT_FILTERED_PATH}")

if __name__ == "__main__":
    main()
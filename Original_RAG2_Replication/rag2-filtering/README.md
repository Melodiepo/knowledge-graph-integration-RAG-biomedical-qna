# RAG2 Filtering Pipeline

## Overview
This pipeline processes retrieved documents using a **perplexity-based label generation approach**, fine-tunes a **FLAN-T5 model** on the labeled data, and then applies the fine-tuned model to filter the retrieved documents. The pipeline consists of the following components:

1. **Label Generation (`generate_labels.py`):**
   - Computes perplexity scores for query-document pairs.
   - Labels documents based on their impact on perplexity reduction.
   - Saves the labeled data for training.

2. **Data Preprocessing (`preprocess_labeled_data.py`):**
   - Converts labeled evidence into a training dataset with text and binary labels.
   - Optionally limits the number of queries and documents per query.
   - Saves the processed training data to a JSON file.

3. **Fine-Tuning FLAN-T5 (`fine_tune_flan_t5.py`):**
   - Fine-tunes a FLAN-T5 model on the processed training data.
   - Splits the dataset into training and evaluation sets.
   - Saves the trained model for later filtering.

4. **Document Filtering (`filtering_module.py`):**
   - Uses the fine-tuned model to compute relevance scores for retrieved documents.
   - Retains only documents that surpass a specified relevance threshold.
   - Saves the filtered results.

5. **Pipeline Integration (`main.py`):**
   - Orchestrates the complete pipeline: label generation, data preprocessing, model fine-tuning, and document filtering.

---

## Running the Pipeline

Execute python main.py

Before running the pipeline, ensure that the required dataset paths exist and your environment is set up properly.

### Environment Setup

- **GPU/CPU:** The scripts prefer running on a GPU (CUDA), but will fall back to CPU if necessary.
- **Model Cache Directory:** To manage disk quota and avoid re-downloading models, set the `HF_HOME` environment variable to use a custom cache directory:


Important to set the env variable to tell the system where to look for the model: 


```bash
# For csh or tcsh
setenv HF_HOME "/cs/student/projects1/ml/2024/yihanli/models"

# For bash/zsh
export HF_HOME="/cs/student/projects1/ml/2024/yihanli/models".py
```

This script will:
1. Generate labeled data.
2. Fine-tune the model on labeled data.
3. Apply the fine-tuned model to filter the documents.



## Notes
- The scripts expect a GPU (`cuda`), but will run on CPU if no GPU is available.
- Make sure that the dataset paths exist before running the script.
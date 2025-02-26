# RAG2 Filtering Pipeline

## Overview
This pipeline processes retrieved documents using a perplexity-based label generation approach, fine-tunes a FLAN-T5 model on the labeled data, and then applies the fine-tuned model to filter the retrieved documents.

## Pipeline Steps
1. **Generate Labels (`generate_labels.py`)**
   - Computes perplexity scores for query-document pairs.
   - Labels documents based on their impact on perplexity reduction.
   - Saves labeled data for training the model.

2. **Fine-Tune FLAN-T5 (`fine_tune_flan_t5.py`)**
   - Uses labeled data to fine-tune FLAN-T5 for classification.
   - Splits the dataset into training and evaluation sets.
   - Saves the trained model for document filtering.

3. **Filter Documents (`filtering_module.py`)**
   - Uses the fine-tuned model to compute relevance scores for retrieved documents.
   - Retains only documents that surpass a certain relevance threshold.
   - Saves the filtered results.

## Running the Pipeline
Instead of running each script manually, you can execute the entire pipeline using:

```bash
python main.py
```

This script will:
1. Generate labeled data.
2. Fine-tune the model on labeled data.
3. Apply the fine-tuned model to filter the documents.

## File Paths
- **Articles Path:** `/cs/student/projects1/ml/2024/yihanli/retriever/articles/pubmed/PubMed_Articles_0.json`
- **Input Evidence Path:** `/cs/student/projects1/ml/2024/yihanli/retriever/output/evidence_medqa_llama_cot.json`
- **Labeled Data Path:** `rag2-filtering/output/labeled_perplexity_data.json`
- **Fine-tuned Model Directory:** `rag2-filtering/fine_tuned_flan_t5`
- **Filtered Output Path:** `rag2-filtering/output/filtered_evidence_medqa_llama_cot.json`


## Notes
- The scripts expect a GPU (`cuda`), but will run on CPU if no GPU is available.
- Make sure that the dataset paths exist before running the script.
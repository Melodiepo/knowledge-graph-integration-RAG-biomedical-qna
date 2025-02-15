# Rationale-Guided RAG2 Pipeline for Medical Question Answering

## **1. Overview**
The **Rationale-Guided Retrieval-Augmented Generation (RAG2)** pipeline integrates:

1. **Retriever** - Identifies relevant documents from large medical corpora.
2. **Filtering Module** - Eliminates unhelpful retrieved documents using a perplexity-based approach.
3. **Classifier** - Predicts the most probable answer (A, B, or C) using a fine-tuned transformer model based on the filtered evidence.

---

## **2. Data Input**
The pipeline requires **two primary datasets**: **medical question answering dataset** and **precomputed corpus embeddings**.

### **(A) Corpus Precomputed Embeddings**
- **Source**: PubMed, PMC, Clinical Practice Guidelines (CPG), Textboook.
- **Format**: `.npy` files stored in `retriever/embeddings/` and `.json` files stored in `retriever/articles/` .
- **Download**: Available from below linkes (by MedCPT)
  - [PubMed Embeddings](https://ftp.ncbi.nlm.nih.gov/pub/lu/MedCPT/pubmed_embeddings/)
  - [Textbooks](https://myuva-my.sharepoint.com/:u:/g/personal/hhu4zu_virginia_edu/EQ8uXe4RiqJJm0Tmnx7fUUkBKKvTwhu9AqecPA3ULUxUqQ?download=1)
  - PMC (TBD)
  - CPG (TBD)


### **(B) Medical QA Dataset (Multiple-Choice Format)**
- **Format**: JSON (`input/medqa_llama_cot.json`).
- **Structure**:
  ```json
  [
    {
      "question": "What are the common symptoms of myocarditis?",
      "options": { "A": "Chest pain", "B": "High blood pressure", "C": "Vision loss" },
      "answer": "A"
    }
  ]
  ```
- **Download**: RAG2 include 3 public QA datasets: MedQA, MedMCQA, MMLU-Med.

---

## **3. Pipeline Architecture**
### **3.1. Workflow Overview**
The pipeline follows these **stages**:

1. **Retriever** (`retriever/main.py`)
   - Performs **FAISS-based nearest-neighbor search** to find **top-k relevant documents** for each question.

2. **Filtering Module** (`filtering/filtering_module.py`)
   - Uses **perplexity differential** to remove irrelevant retrieved documents.

3. **Classifier** (`classifier/run_classifier.py`)
   - Applies a **fine-tuned Seq2Seq model** to classify the best answer.

---

## **4. Pipeline Components**

### **4.1. Retriever (`retriever/main.py`)**
Retrieve **top-k relevant medical documents** for a given question using **FAISS-based retrieval**.

#### **Input**
- `--embeddings_dir`: Directory containing document embeddings.
- `--articles_dir`: Directory containing the actual text of articles.
- `--input_path`: JSON file with input medical queries (default: `/retriever/input/medqa/medqa_llama_cot.json`).
- `--corpus`: List of corpora to search in (`cpg`, `textbook`, `pmc`, `pubmed`).
- `--top_k`: Number of top documents to retrieve (default: `100`).
- `--instruction_preprocess`: Whether to preprocess queries for instruction set retrieval.
- `--output_path`: JSON file path to store retrieved and reranked results.
- `--use_spacy`: Whether to use SciSpacy for tokenizing sentences.
- `--pubmed_group_num`: Number of chunks of PubMed articles to concatenate.


#### **Processing**
1. **Query Encoding**
   - Converts text queries into vector embeddings (`qe.query_encode()`).
2. **FAISS Search**
   - Performs **Maximum Inner Product Search (MIPS)** across corpus embeddings (`rt.pubmed_index_create()`).
3. **Document Retrieval**
   - Extracts **top-k nearest documents** for each query.
4. **Re-ranking**
   - Combines retrieved documents from multiple corpora (`rr.combine_query_evidence()`) and performs ranking of retrieved documents (`rr.rerank()`).

#### **Output**
- **Retrieved Evidence JSON** (`retriever/output/evidence_medqa_llama_cot.json`)
  - Contains **reranked top-k retrieved documents** for each query.


---

### **4.2. Filtering Module (`filtering/filtering_module.py`)**
Remove **unhelpful retrieved documents** using a **perplexity-based filtering approach**.

#### **Input**
- **Medical Query**.
- **Retrieved evidence** (`retriever/output/evidence_medqa_llama_cot.json`).
- **Flan-T5 Model** (pretrained).

#### **Processing**
1. **Compute Perplexity of the Query Alone** (`compute_perplexity()`).
2. **Compute Perplexity of Query + Document**.
3. **Calculate Perplexity Differential**:
   \[
   \Delta PPL = PPL(\text{query}) - PPL(\text{query} + \text{document})
   \]
4. **Filter Based on Threshold** (`threshold = 0.25`):
   - Retains only **documents where \(\Delta PPL \geq \text{threshold}\)**.

#### **Output**
- **Filtered Evidence JSON** (`filtering/output/filtered_evidence_medqa_llama_cot.json`)
  - Contains only the **most relevant retrieved documents**.


---

### **4.3. Classifier (`classifier/run_classifier.py`)**
#### **Objective**
Predict the **correct answer choice (A, B, or C)** using a **fine-tuned transformer model**.

#### **Input**
- **Filtered Evidence** (`filtering/output/filtered_evidence_medqa_llama_cot.json`).
- **Pretrained Transformer Model** (e.g., `t5-base`, `Flan-T5`).
- **Medical QA Dataset** (`data/medqa_llama_cot.json`).

#### **Processing**
1. **Data Loading & Tokenization**  
   - Loads raw **training/validation** data from a local file or Hugging Face dataset.  
   - Converts questions and answers to **tokenized** sequences (respecting `max_seq_length`), using a **seq2seq** tokenizer.

2. **Seq2Seq Fine-Tuning**  
   - Trains a **Transformer-based** model (e.g., T5) to predict correct answers.  
   - Uses `DataCollatorForSeq2Seq` for batching, then applies **backpropagation** and optimizations (e.g., AdamW, LR scheduling).

3. **Probability Computation & Accuracy Calculation**  
   - Performs **inference** on validation data via `model.generate(...)`.  
   - Extracts **softmax** scores for each answer option, selects the predicted class, and **computes accuracy** to gauge model performance.

4. **Outputs & Logging**  
   - Logs training progress (e.g., losses, steps) and saves **model checkpoints**.  
   - Writes **evaluation metrics** (accuracy, per-class metrics) to JSON files and optionally **pushes** results to the Hugging Face Hub.


#### **Output**
- **Final Predictions JSON** (`classifier/output/final_eval_results.json`)
  - Stores the **predicted answers** for each question.

---

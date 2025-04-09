# Enhancing Context Relevance and Efficiency in Biomedical Retrieval-Augmented Generation Through Lightweight Filtering Methods

## Abstract
Large Language Models (LLMs) have shown promising potential in biomedical question answering (QA), yet remain susceptible to \textit{hallucinations}--plausible but factually incorrect responses. Retrieval-Augmented Generation (RAG) addresses this by grounding answers in external knowledge, though it often retrieves irrelevant or noisy context, particularly in specialized domains like biomedicine. 

In this work, we replicate RAG2, a rationale-guided variant designed to improve context relevance, on the MedQA dataset and identify substantial performance gaps compared to the original paper, indicating potential reproducibility issues and sensitivity to implementation details. To address these limitations, we explore three lightweight, supervision-free context filtering methods based on pretrained models and similarity metrics. Our approaches achieve performance comparable to the RAG2 replication while reducing runtime by up to 75\%, though they remain below the originally reported results. Nonetheless, our empirical and qualitative analyses indicate their potential to enhance context relevance and computational efficiency.



## Repository Structure
### Original_RAG2_Replication
- **rag2-retriever:** Contains the implementation of document retrieval using MedCPT encoders and FAISS indexing, including query preprocessing, retrieval, and reranking logic.
- **rag2-filtering:** Implements the rationale-guided filtering mechanism using perplexity-based FLAN-T5 classifier to prune irrelevant context.
- **rag2-generation:** Includes scripts for generating final answers using a causal language model (BioMistral-7B) with retrieved and filtered contexts.

### cxmi
- Contains scripts to perform Conditional Cross-Mutual Information (CXMI) based filtering at passage and sentence levels using FLAN-T5 models (base and large).
- Provides utilities for computing CXMI scores, prompt building, and evaluating model outputs specifically tailored to biomedical datasets.

### provence
- Provides the implementation of the Provence model for adaptive token-level pruning and reranking using pretrained DeBERTa-v3 cross-encoders.
- The `output` folder includes generated outputs and sample results to demonstrate the effectiveness of the Provence filtering method.

### similarity_pruning
- Implements embedding-based filtering leveraging cosine similarity to retain only the most semantically relevant context passages based on adaptive thresholding strategies.

---

## Getting Started

To run the experiments, first set up the required Python environment:

```bash
pip install -r requirements.txt

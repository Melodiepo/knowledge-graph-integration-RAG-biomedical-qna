# knowledge-graph-integration-RAG-biomedical-qna

**Goal**: A minimal Retrieval-Augmented Generation (RAG) pipeline for biomedical question answering, using:
- A domain-specific model (BioBERT / PubMedBERT) as the retriever
- A generative language model (BioGPT / T5 / GPT) as the generator
- [LangChain](https://github.com/hwchase17/langchain) to orchestrate retrieval + generation

## Features
- Easily swap the **retriever model** (e.g., BioBERT, PubMedBERT, or other Sentence-Transformer-based embeddings).
- Easily swap the **generator model** (BioGPT, T5, or even GPT via OpenAI).
- Minimal code to illustrate the RAG pattern with a small sample dataset.
- Extendable to more complex pipelines (multi-hop reasoning, knowledge graph queries, etc.).

## Setup Instructions

1. **Clone this repository**:

    ```bash
    git clone https://github.com/Melodiepo/knowledge-graph-integration-RAG-biomedical-qna.git
    ```

2. **Create a Python environment** (using conda or venv):

    ```bash
    conda create -n bio-rag python=3.9
    conda activate bio-rag
    ```
    or
    ```bash
    python -m venv bio-rag
    source bio-rag/bin/activate  # Linux/Mac
    # or
    bio-rag\Scripts\activate     # Windows
    ```

3. **Install requirements**:

    ```bash
    pip install --upgrade pip
    pip install -r requirements.txt
    ```

4. **Run the pipeline**:

    ```bash
    cd similarity_pruning
    python pipeline.py
    ```
    This script will:
    - Load a small biomedical text corpus
    - Build a vector store from embeddings generated by BioBERT/PubMedBERT
    - Run a simple question query (hard-coded) through the pipeline
    - Output an answer generated by the chosen model (BioGPT or T5)

## Usage & Customization

- **Swapping Retrieval Model**: In `pipeline.py`, change the `retriever_model_name` to another [Hugging Face model](https://huggingface.co/models) that supports embeddings, e.g. `pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb` or `nreimers/MiniLM-L6-H384-uncased` for quick experimentation.
- **Swapping Generator Model**: In `pipeline.py`, change `generator_model_name` from e.g. `microsoft/BioGPT-Large` to `google/flan-t5-base` or an OpenAI GPT model (via `OpenAI` in LangChain).
- **Dataset**: Replace `sample_corpus.txt` with your own biomedical documents. For a larger corpus, you may want a chunking strategy or a more robust vector store (e.g. FAISS, Chroma).
- **Evaluation**: A minimal example is shown in the code. Replace with domain-specific metrics (exact match, entity-level correctness, etc.).



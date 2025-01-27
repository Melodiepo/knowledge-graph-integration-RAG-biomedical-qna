"""
pipeline.py

A minimal RAG pipeline prototype for biomedical QA using:
- BioBERT or PubMedBERT embeddings as a retriever
- BioGPT or T5 (or other GPT) for generation
- LangChain for orchestration

"""

import os
import sys
import logging
from typing import List

import torch
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate

# Vector store (Here, we use Chroma for example. Swap to FAISS if you prefer.)
from langchain_community.vectorstores import Chroma

# Hugging Face-based embeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

# LangChain prompt template (optional) / chain
from langchain.chains import RetrievalQA

# For a local huggingface generative model
from langchain_community.llms import HuggingFacePipeline

# OR if you want an OpenAI GPT model, uncomment below
# from langchain.llms import OpenAI

# To handle text generation pipeline
from transformers import (AutoTokenizer,
                          AutoModelForCausalLM,
                          AutoModelForSeq2SeqLM,
                          pipeline as hf_pipeline)

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------
# 1) RETRIEVER MODEL CHOICE
# -------------------------
# Example domain-specific embedding model for retrieval
# You can swap to "dmis-lab/biobert-base-cased-v1.1", "microsoft/BioGPT-Large", etc.
retriever_model_name = "BioBERT-mnli-snli-scinli-scitail-mednli-stsb"

# --------------------------
# 2) GENERATIVE MODEL CHOICE
# --------------------------
# Example: local BioGPT
# generator_model_name = "microsoft/BioGPT-Large"

# Example: T5
generator_model_name = "google/flan-t5-base"

# If using a *Causal* model (e.g., GPT-style), set pipeline='text-generation'
# If using a *Seq2Seq* model (e.g., T5, BART), set pipeline='text2text-generation'

IS_SEQ2SEQ = "t5" in generator_model_name.lower() or "bart" in generator_model_name.lower()

# ---------------------
# 3) LOAD OUR DATASET
# ---------------------
# In a real project, you'd load multiple documents or a large corpus, then chunk them.
# For demonstration, let's assume we have a single text file "sample_corpus.txt".

CORPUS_FILE = os.path.join(os.path.dirname(__file__), "sample_corpus.txt")

# If this doesn't exist, create a small sample to illustrate.
if not os.path.exists(CORPUS_FILE):
    text_data = """\
Breast cancer type 1 susceptibility protein (BRCA1) is crucial for DNA repair via homologous recombination.
Vancomycin is commonly used to treat MRSA infections.
Gene TP53 is a tumor suppressor and involved in multiple cancers.
"""
    with open(CORPUS_FILE, "w", encoding="utf-8") as f:
        f.write(text_data)

with open(CORPUS_FILE, "r", encoding="utf-8") as f:
    raw_text = f.read().strip()

# Split into smaller chunks if needed
text_splitter = CharacterTextSplitter(
    chunk_size=300, chunk_overlap=50, separator="\n"
)
docs = text_splitter.split_text(raw_text)
documents = [Document(page_content=chunk) for chunk in docs]

logger.info(f"Loaded {len(documents)} documents/chunks from {CORPUS_FILE}.")

# ---------------------
# 4) Make a custom prompt
# ---------------------

custom_prompt = PromptTemplate(
    template="""You are a helpful assistant for biomedical questions.
Use the following context to answer the question in 2-3 sentences. 
If you don't know, say "I am not sure" and do not fabricate content.

Context:
{context}

Question: {question}
Answer (please be concise yet complete):""",
    input_variables=["context", "question"]
)

# -----------------------------------
# 5) BUILD RETRIEVER (EMBEDDINGS + DB)
# -----------------------------------
logger.info("Loading retriever embedding model...")
embedding_function = HuggingFaceEmbeddings(
    model_name=retriever_model_name
    # model_kwargs={"device": "cpu"}  # or "cuda" if GPU available
)

# Create a vector store from these docs
logger.info("Creating vector store...")
vectorstore = Chroma.from_documents(documents, embedding_function)

retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# -----------------------------
# 6) LOAD/WRAP GENERATIVE MODEL
# -----------------------------
logger.info("Loading generator model/pipeline...")

if IS_SEQ2SEQ:
    # T5 / BART, etc. pipeline
    generator_tokenizer = AutoTokenizer.from_pretrained(generator_model_name)
    generator_model = AutoModelForSeq2SeqLM.from_pretrained(generator_model_name)
    pipe = hf_pipeline(
        "text2text-generation",
        model=generator_model,
        tokenizer=generator_tokenizer,
        max_length=256,
        min_length=40,   
        do_sample=True,
        top_p=0.9,
        temperature=0.7,
        num_beams=3,
        device=-1
    )
    llm = HuggingFacePipeline(pipeline=pipe)
else:
    # GPT-style (BioGPT or GPT2-like)
    generator_tokenizer = AutoTokenizer.from_pretrained(generator_model_name)
    generator_model = AutoModelForCausalLM.from_pretrained(generator_model_name)
    pipe = hf_pipeline(
        "text-generation",
        model=generator_model,
        tokenizer=generator_tokenizer,
        min_length=40,   
        do_sample=True,
        top_p=0.9,
        temperature=0.7,
        num_beams=3,
        device=-1
    )
    llm = HuggingFacePipeline(pipeline=pipe)

# If you wanted to use OpenAI GPT instead, you could do:
# llm = OpenAI(model_name="gpt-3.5-turbo", temperature=0.0)

# ----------------------
# 7) BUILD THE RAG CHAIN
# ----------------------
logger.info("Building RetrievalQA chain...")
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",  # or "refine", "map_reduce"
    retriever=retriever,
    chain_type_kwargs={"prompt": custom_prompt},
    return_source_documents=True,
)

# ---------------------------
# 8) TEST THE PIPELINE (DEMO)
# ---------------------------
# A sample question. You can replace with your domain-specific query.
sample_question = "What is BRCA1's role in DNA repair?"

logger.info(f"Running sample question: {sample_question}")
result = qa_chain({"query": sample_question})

print("=== Question ===")
print(sample_question)
print("\n=== Generated Answer ===")
print(result["result"])

# Optionally view the retrieved source docs
print("\n--- Source Documents ---")
for i, doc in enumerate(result["source_documents"]):
    snippet = doc.page_content.replace("\n", " ")
    print(f"[Doc {i+1}] {snippet[:200]}...")

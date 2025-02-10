#!/usr/bin/env python
"""
Main Script for the RAG2 Pipeline
----------------------------------
This script demonstrates the use of the FilteringModule and GenerationModule.
It simulates an end-to-end pipeline where a query and a set of candidate documents
are provided, the filtering module selects the helpful documents, and the generation
module produces the final answer.
"""

from filtering_module import FilteringModule
from generation_module import GenerationModule

def main():
    # Example query and candidate documents (these would normally be retrieved via FAISS/retriever)
    query = (
        "A 5-year-old boy presents with bilateral conjunctivitis and pharyngitis. "
        "What is the most likely cause?"
    )
    candidate_docs = [
        "Evidence suggests that adenovirus is a common cause of conjunctivitis in children.",
        "Influenza virus typically causes high fever and cough but less often conjunctivitis.",
        "Human metapneumovirus is associated with respiratory infections but rarely conjunctivitis.",
    ]

    # Initialize the filtering module (using Flan-T5 as the filtering model)
    filtering_model_name = "google/flan-t5-large"  # Adjust if using a fine-tuned version
    filter_module = FilteringModule(filtering_model_name, threshold=0.25)
    
    print("=== Running Filtering Module ===")
    filtered_docs = filter_module.filter_documents(query, candidate_docs)
    
    print("\nFiltered Documents:")
    for idx, doc in enumerate(filtered_docs, start=1):
        print(f"{idx}. {doc}")
    
    # Initialize the generation module (this can be the same or a different model)
    generation_model_name = "google/flan-t5-large"  # Replace with your chosen generation model
    gen_module = GenerationModule(generation_model_name, max_length=64, num_beams=5)
    
    print("\n=== Running Generation Module ===")
    answer = gen_module.generate_answer(query, filtered_docs)
    print("\nGenerated Answer:")
    print(answer)

if __name__ == "__main__":
    main()

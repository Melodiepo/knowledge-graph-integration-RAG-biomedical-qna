#!/usr/bin/env python
"""
Filtering Module
----------------
This module defines a FilteringModule class that uses a Flan‑T5 model
to filter candidate documents based on the change in perplexity when the
document is appended to a query. Documents for which the perplexity
differential exceeds a set threshold are considered helpful.
"""

import math
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

class FilteringModule:
    def __init__(self, model_name_or_path: str, threshold: float = 0.25):
        """
        Initializes the filtering module.
        
        Args:
            model_name_or_path (str): Path or identifier for the Flan-T5 model.
            threshold (float): The minimum perplexity differential required
                               to consider a document helpful.
        """
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.threshold = threshold
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def compute_perplexity(self, text: str) -> float:
        """
        Compute the perplexity of a given text using the model's loss.
        
        Args:
            text (str): The text to evaluate.
            
        Returns:
            float: The perplexity computed as exp(loss).
        """
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512
        ).to(self.device)
        with torch.no_grad():
            # Use the text as both input and target.
            outputs = self.model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss.item()
        perplexity = math.exp(loss)
        return perplexity

    def filter_documents(self, query: str, candidate_docs: list) -> list:
        """
        For each candidate document, compute the perplexity differential:
            ΔPPL = PPL(query) - PPL(query + document)
        Only documents with ΔPPL >= threshold are returned.
        
        Args:
            query (str): The original query.
            candidate_docs (list): A list of candidate document strings.
        
        Returns:
            list: A list of documents deemed helpful.
        """
        ppl_query = self.compute_perplexity(query)
        filtered_docs = []
        for doc in candidate_docs:
            # Combine the query and the candidate document
            combined_text = f"{query} {doc}"
            ppl_with_doc = self.compute_perplexity(combined_text)
            delta_ppl = ppl_query - ppl_with_doc
            print(f"Document delta perplexity: {delta_ppl:.4f}")
            if delta_ppl >= self.threshold:
                filtered_docs.append(doc)
        return filtered_docs

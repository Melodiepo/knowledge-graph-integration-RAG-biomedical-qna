#!/usr/bin/env python
"""
Generation Module
-----------------
This module defines a GenerationModule class that uses an LLM to generate an answer.
It concatenates the original query with the filtered document contexts and then uses
beam search to generate a final answer.
"""

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

class GenerationModule:
    def __init__(self, model_name_or_path: str, max_length: int = 64, num_beams: int = 5):
        """
        Initializes the generation module.
        
        Args:
            model_name_or_path (str): Path or identifier for the generation model.
            max_length (int): The maximum length for the generated answer.
            num_beams (int): The number of beams for beam search.
        """
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.max_length = max_length
        self.num_beams = num_beams
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def generate_answer(self, query: str, filtered_docs: list) -> str:
        """
        Generates an answer by concatenating the query with the filtered documents
        and then performing generation with beam search.
        
        Args:
            query (str): The original query.
            filtered_docs (list): A list of filtered document strings.
        
        Returns:
            str: The generated answer.
        """
        # Concatenate filtered documents into one context string
        context = " ".join(filtered_docs)
        input_text = f"{query} {context}"
        inputs = self.tokenizer(
            input_text, return_tensors="pt", truncation=True, max_length=512
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=self.max_length,
                num_beams=self.num_beams,
                early_stopping=True,
            )
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return answer

import os
import json
import math
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm 

'''
EXPECTS THE FOLLOWING STYLE OUTPUR FROM RETRIEVER:

[
    {
        "query": "A junior orthopaedic surgery resident is completing a carpal tunnel repair with the department chairman as the attending physician. During the case, the resident inadvertently cuts a flexor tendon. The tendon is repaired without complication. The attending tells the resident that the patient will do fine, and there is no need to report this minor complication that will not harm the patient, as he does not want to make the patient worry unnecessarily. He tells the resident to leave this complication out of the operative report. Which of the following is the correct next action for the resident to take?",
        "retrieved_docs": [
            "36719412",
            "36895610",
            "36339070",
            "36821739",
            "36212594",
            "36728037",
            "36576536",
            "36534362",
            "36727955",
            "36107116"
        ]
    },
    {
        "query": "A 67-year-old man with transitional cell carcinoma of the bladder comes to the physician because of a 2-day history of ringing sensation in his ear. He received this first course of neoadjuvant chemotherapy 1 week ago. Pure tone audiometry shows a sensorineural hearing loss of 45 dB. The expected beneficial effect of the drug that caused this patient's symptoms is most likely due to which of the following actions?",
        "retrieved_docs": [
            "36567514",
            "36051418",
            "36051415",
            "36742606",
            "36699929",
            "36874770",
            "36032922",
            "36089725",
            "36054054",
            "36019254"
        ]
    },
]
'''
class PerplexityLabelGenerator:
    def __init__(self, model_name_or_path: str, threshold: float = 0.25):
        """
        Initializes the label generator using a pre-trained Flan-T5 model.
        Args:
            model_name_or_path (str): Path to the pretrained Flan-T5 model.
            threshold (float): Threshold for determining if a document is helpful.
        """
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.threshold = threshold
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def compute_perplexity(self, text: str) -> float:
        """
        Compute the perplexity of a given text.
        """
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss.item()
        return math.exp(loss)

    def generate_labels(self, input_path: str, output_path: str):
        """
        Reads retrieved evidence, computes perplexity differentials, labels them, and saves the output.
        """
        with open(input_path, "r") as f:
            retrieved_data = json.load(f)

        labeled_data = {}
        for idx, content in tqdm(enumerate(retrieved_data), total=len(retrieved_data), desc="Generating Labels"):
            query = content["query"]
            docs = content["retrieved_docs"]
            labels = []

            # Compute perplexity for the base query
            base_perplexity = self.compute_perplexity(query)

            for doc in docs:
                combined_text = f"{query} {doc}"
                doc_perplexity = self.compute_perplexity(combined_text)
                delta_ppl = base_perplexity - doc_perplexity
                label = 1 if delta_ppl >= self.threshold else 0  # Helpful if perplexity reduction exceeds threshold
                labels.append(label)

            labeled_data[idx] = {
                "query": query,
                "retrieved_docs": docs,
                "labels": labels
            }

        # Save the labeled data
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(labeled_data, f, indent=4)

        print(f"Labeled data saved to {output_path}")


if __name__ == "__main__":
    MODEL_NAME = "google/flan-t5-base"  # Pretrained Flan-T5 model
    INPUT_EVIDENCE_PATH = "/Original_RAG2_Replication/rag2-retriever/output/evidence_medqa_llama_cot.json"  # Retrieved evidence JSON
    OUTPUT_LABELED_PATH = "/Original_RAG2_Replication/rag2-filtering/output/labeled_perplexity_data.json"  # Output path for labeled data

    label_generator = PerplexityLabelGenerator(model_name_or_path=MODEL_NAME, threshold=0.25)
    label_generator.generate_labels(INPUT_EVIDENCE_PATH, OUTPUT_LABELED_PATH)
    print("Label generation completed.")

    
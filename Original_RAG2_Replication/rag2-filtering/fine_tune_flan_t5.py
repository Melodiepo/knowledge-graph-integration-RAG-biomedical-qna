import os
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import json
import random
from tqdm import tqdm

# Define a custom dataset for classification fine-tuning
class PerplexityLabeledDataset(Dataset):
    def __init__(self, data_path, articles_path, tokenizer, max_length=512, split="train", train_ratio=0.8, seed=42):
        with open(data_path, 'r') as f:
            self.data = json.load(f)
        with open(articles_path, 'r') as f:
            self.articles_data = json.load(f)  # Load full document texts

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = self.prepare_samples()

        # Shuffle and split dataset
        random.seed(seed)
        random.shuffle(self.samples)
        split_idx = int(len(self.samples) * train_ratio)
        self.samples = self.samples[:split_idx] if split == "train" else self.samples[split_idx:]

    def prepare_samples(self):
        samples = []
        for entry in self.data.values():
            query = entry["query"]
            docs = entry["retrieved_docs"]
            labels = entry["labels"]

            assert len(docs) == len(labels), f"Mismatch: {len(docs)} docs vs {len(labels)} labels"

            for doc, label in zip(docs, labels):
                doc_content = self.get_document_text(doc)
                if not doc_content:
                    continue

                text = f"Query: {query} Document: {doc_content}"
                samples.append((text, label))
        return samples

    def get_document_text(self, pmid):
        doc_data = self.articles_data.get(str(pmid))
        if doc_data:
            title = doc_data.get("t", "")
            abstract = doc_data.get("a", "")
            return f"{title}. {abstract}"
        return ""

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text, label = self.samples[idx]
        inputs = self.tokenizer(
            text,
            padding="longest",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        label = torch.tensor(label, dtype=torch.long)
        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "labels": label
        }

# Fine-tuning function for classification
def fine_tune_flan_t5(model_name: str, dataset_path: str, articles_path: str, output_dir: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_dataset = PerplexityLabeledDataset(dataset_path, articles_path, tokenizer, split="train")
    eval_dataset = PerplexityLabeledDataset(dataset_path, articles_path, tokenizer, split="eval")

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,
        learning_rate=3e-5,
        weight_decay=0.01,
        warmup_steps=50,
        logging_dir=os.path.join(output_dir, "logs"),
        logging_steps=50,
        gradient_checkpointing=True,
        fp16=False,
        max_grad_norm=1.0,
        optim="adamw_torch",
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer
    )

    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Fine-tuning complete. Model saved to {output_dir}")

# Main execution
if __name__ == "__main__":
    MODEL_NAME = "google/flan-t5-base"
    LABELED_DATA_PATH = "/cs/student/projects1/ml/2024/yihanli/uandreic/rag2-filtering/output/labeled_perplexity_data.json"
    ARTICLES_PATH = "/cs/student/projects1/ml/2024/yihanli/retriever/articles/pubmed/PubMed_Articles_0.json"
    OUTPUT_DIR = "./fine_tuned_flan_t5"

    fine_tune_flan_t5(MODEL_NAME, LABELED_DATA_PATH, ARTICLES_PATH, OUTPUT_DIR)
import os
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import json
from tqdm import tqdm

# Define a custom dataset for classification fine-tuning
class PerplexityLabeledDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        with open(data_path, 'r') as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = self.prepare_samples()

    def prepare_samples(self):
        samples = []
        for entry in self.data.values():
            query = entry['query']
            docs = entry['retrieved_docs']
            labels = entry['labels']
            for doc, label in zip(docs, labels):
                text = f"{query} {doc}"
                samples.append((text, label))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text, label = self.samples[idx]
        inputs = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        label = torch.tensor(label, dtype=torch.long)
        return {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'labels': label
        }

# Fine-tuning function for classification
def fine_tune_flan_t5(model_name: str, dataset: Dataset, output_dir: str):
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)  # Binary classification
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=2,  # Small batch size for MPS
        evaluation_strategy="no",  # No evaluation during training
        save_strategy="epoch",
        learning_rate=5e-5,
        weight_decay=0.01,
        logging_dir=os.path.join(output_dir, "logs"),
        save_total_limit=2,
        gradient_checkpointing=True,  # Helps manage memory on limited GPU setups
        fp16=False  # Disable FP16 for MPS backend compatibility
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer
    )

    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

# Main execution
if __name__ == "__main__":
    MODEL_NAME = "google/flan-t5-base"  # Pretrained model for classification
    LABELED_DATA_PATH = "rag2-filtering/output/labeled_perplexity_data.json"  # Labeled data JSON path
    OUTPUT_DIR = "./fine_tuned_flan_t5"  # Output directory for the fine-tuned model

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    dataset = PerplexityLabeledDataset(LABELED_DATA_PATH, tokenizer, max_length=256)  # Shorter input length for MPS memory management

    fine_tune_flan_t5(MODEL_NAME, dataset, OUTPUT_DIR)
    print(f"Fine-tuning complete. Model saved to {OUTPUT_DIR}")
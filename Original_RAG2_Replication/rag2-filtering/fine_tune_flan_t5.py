import json
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)


class SimpleClassificationDataset(Dataset):
    def __init__(
        self,
        json_path,
        tokenizer,
        max_length=512,
        split="train",
        train_ratio=0.8,
        seed=42,
    ):
        """
        Initialize a simple classification dataset for training a sequence classification model. The dataset is loaded
        from a JSON file containing samples with text and labels. The dataset is split into training and evaluation
        sets based on the train_ratio.

        Args:
            json_path (str): Path to the JSON file containing the dataset.
            tokenizer (transformers.PreTrainedTokenizer): Tokenizer for encoding text samples.
            max_length (int): Maximum length of the tokenized input sequences.
            split (str): Split to use ("train" or "eval").
            train_ratio (float): Ratio of samples to use for training.
            seed (int): Random seed for shuffling the dataset.
        """
        self.tokenizer = tokenizer
        self.max_length = max_length

        with open(json_path, "r", encoding="utf-8") as f:
            all_samples = json.load(f)

        # Shuffle and split
        torch.manual_seed(seed)
        perm = torch.randperm(len(all_samples)).tolist()
        split_index = int(len(all_samples) * train_ratio)
        if split == "train":
            self.samples = [all_samples[i] for i in perm[:split_index]]
        else:
            self.samples = [all_samples[i] for i in perm[split_index:]]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset by index. The sample is tokenized and returned as a dictionary with input_ids,
        attention_mask, and labels.

        Args:
            idx (int): Index of the sample to retrieve.
        """
        item = self.samples[idx]
        text = item["text"]
        label = item["label"]
        enc = self.tokenizer(
            text,
            padding="longest",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)
        label_tensor = torch.tensor(label, dtype=torch.long)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": label_tensor,
        }


def train_classifier(model_name, data_path, output_dir):
    """
    Fine-tune a sequence classification model on a simple dataset. The model is trained using the Trainer API from
    Hugging Face Transformers. The dataset is loaded from a JSON file containing samples with text and labels. The
    model and tokenizer are saved to the output directory after training.

    Args:
        model_name (str): Pretrained model name or path.
        data_path (str): Path to the JSON file containing the training data.
        output_dir (str): Directory to save the fine-tuned model and tokenizer.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_dataset = SimpleClassificationDataset(
        data_path, tokenizer, split="train", train_ratio=0.8
    )
    eval_dataset = SimpleClassificationDataset(
        data_path, tokenizer, split="eval", train_ratio=0.8
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2
    ).to(device)

    training_args = TrainingArguments(
        output_dir=output_dir,
        fp16=False,
        num_train_epochs=5,
        per_device_train_batch_size=2, 
        per_device_eval_batch_size=2, 
        evaluation_strategy="no", # No evaluation during training
        save_strategy="no", # No checkpoints saved during training
        save_total_limit=1, # Only keep one checkpoint
        learning_rate=3e-5, 
        weight_decay=0.01, 
        warmup_steps=50,
        logging_steps=50,
        logging_dir=f"{output_dir}/logs",
        gradient_checkpointing=False,
        max_grad_norm=1.0,
        optim="adamw_torch",
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        dataloader_num_workers=0,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )

    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Training complete. Model + tokenizer saved to: {output_dir}")


if __name__ == "__main__":
    MODEL_NAME = "google/flan-t5-large"
    PROCESSED_DATA_JSON = "/cs/student/projects1/ml/2024/yihanli/uandreic/rag2-filtering/output/training_data.json"
    OUTPUT_DIR = "/cs/student/projects1/ml/2024/yihanli/uandreic/rag2-filtering/fine_tuned_flan_t5_large"

    train_classifier(MODEL_NAME, PROCESSED_DATA_JSON, OUTPUT_DIR)
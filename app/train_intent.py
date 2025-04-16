import random
import torch
from torch.utils.data import Dataset
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import os

# Define Intents
INTENTS = [
    "SEMANTIC",
    "KEYWORD",
    "HYBRID",
    "STRUCTURED",
    "HYBRID_STRUCTURED",
    "AGGREGATE",
    "COMPARISON",
    "TEMPORAL",
    "EXPLANATORY",
    "MULTI_INTENT",
    "ENTITY_SPECIFIC",
    "DOCUMENT_FETCH",
]
INTENT2ID = {intent: idx for idx, intent in enumerate(INTENTS)}


# Generate Synthetic Data
def generate_synthea_names(n=1000):
    first_names = ["Julian", "Emma", "Liam", "Olivia", "Noah", "Ava"]
    last_names = ["Stamm", "Turner", "Smith", "Johnson", "Brown"]
    return [
        f"{random.choice(first_names)}{random.randint(100, 999)} {random.choice(last_names)}{random.randint(100, 999)}"
        for _ in range(n)
    ]


def generate_intent_data(n_samples=2000):
    names = generate_synthea_names(1000)
    conditions = ["migraine", "sinusitis", "hypertension", "diabetes"]
    codes = ["I21", "99213", "12345-6"]
    templates = {
        "SEMANTIC": ["Search for {} treatment options.", "What are the causes of {}?"],
        "KEYWORD": ["Look up code {}.", "Find {} in records."],
        "HYBRID": ["Find patients with {}.", "List patients with {} and {}."],
        "STRUCTURED": [
            "List procedures with code {} for {}.",
            "Find conditions with code {}.",
        ],
        "HYBRID_STRUCTURED": [
            "Find patients with {} and code {}.",
            "List {} patients with {}.",
        ],
        "AGGREGATE": ["How many patients have {}?", "Count patients with {}."],
        "COMPARISON": ["Compare {} vs {} outcomes.", "Show {} vs {} for {}."],
        "TEMPORAL": ["Show trends for {}'s {}.", "Track {} for patient {}."],
        "EXPLANATORY": ["Explain {}.", "What is {} in medical terms?"],
        "MULTI_INTENT": [
            "Explain {} and list patients with it.",
            "Fetch {} records and trends.",
        ],
        "ENTITY_SPECIFIC": [
            "Get details for patient {}.",
            "Show info about {}.",
            "Records for {}.",
        ],
        "DOCUMENT_FETCH": ["Fetch document for {}.", "Get record for patient {}."],
    }
    data = []
    for _ in range(n_samples):
        intent = random.choice(INTENTS)
        template = random.choice(templates[intent])
        num_placeholders = template.count("{}")
        if num_placeholders == 2:
            if (
                " and " in template or " vs " in template
            ):  # e.g., "List patients with {} and {}" or "Compare {} vs {}"
                val1 = random.choice(conditions)
                val2 = random.choice(conditions + codes)
                text = template.format(val1, val2)
            elif "code {} for {}" in template or "{} patients with {}" in template:
                name = random.choice(names)
                code = random.choice(codes)
                text = (
                    template.format(code, name)
                    if "code {} for {}" in template
                    else template.format(name, code)
                )
            elif (
                "trend" in template or "track" in template
            ):  # e.g., "Show trends for {}'s {}"
                name = random.choice(names)
                metric = random.choice(["blood pressure", "weight"])
                text = template.format(name, metric)
            else:  # Fallback
                name = random.choice(names)
                cond = random.choice(conditions)
                text = template.format(name, cond)
        elif num_placeholders == 1:
            if "code {}" in template:
                code = random.choice(codes)
                text = template.format(code)
            elif intent in ["ENTITY_SPECIFIC", "DOCUMENT_FETCH"]:
                name = random.choice(names)
                text = template.format(name)
            else:
                cond = random.choice(conditions)
                text = template.format(cond)
        else:  # No placeholders
            text = template
        data.append({"text": text, "intent": intent})
    return data


# Custom Dataset
class IntentDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item["text"]
        label = INTENT2ID[item["intent"]]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(label, dtype=torch.long),
        }


# Compute Metrics
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="weighted")
    return {"accuracy": acc, "f1": f1}


# Custom Trainer for Single Checkpoint
class CustomIntentTrainer(Trainer):
    def training_step(self, model, inputs):
        step_result = super().training_step(model, inputs)
        self.global_step = self.global_step + 1 if hasattr(self, "global_step") else 1
        if self.global_step % 25 == 0:  # Save every k=25 samples
            checkpoint_dir = "./intent_model/checkpoint"
            self.model.save_pretrained(checkpoint_dir)
            self.tokenizer.save_pretrained(checkpoint_dir)
            print(
                f"Overwrote checkpoint at {checkpoint_dir} at step {self.global_step}"
            )
        return step_result


# Main Training
def main():
    # Initialize
    model_name = "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(
        model_name, num_labels=len(INTENTS)
    )

    # Generate data
    data = generate_intent_data(2000)
    train_data = data[:1600]
    eval_data = data[1600:]

    # Create datasets
    train_dataset = IntentDataset(train_data, tokenizer)
    eval_dataset = IntentDataset(eval_data, tokenizer)

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./intent_model",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
        num_train_epochs=5,
        weight_decay=0.01,
        logging_dir="./logs_intent",
        logging_steps=10,
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
    )

    # Trainer
    trainer = CustomIntentTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    # Train
    trainer.train()

    # Save final model
    model.save_pretrained("./intent_model/final")
    tokenizer.save_pretrained("./intent_model/final")

    # Test
    model.eval()
    test_queries = [
        "Get details for patient Julian140 Stamm395.",
        "Find patients with migraine.",
        "Fetch document for Emma567 Brown123.",
    ]
    for query in test_queries:
        inputs = tokenizer(query, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
        pred = outputs.logits.argmax(-1).item()
        print(f"\nQuery: {query}")
        print(f"Intent: {INTENTS[pred]}")


if __name__ == "__main__":
    main()

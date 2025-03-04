# Import required libraries
import os
import json
import torch
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForTokenClassification
from torch.optim import AdamW
from transformers import get_scheduler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from torch.amp import autocast, GradScaler

# Prevent issues related to parallelism in tokenizers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Set Random Seed for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

# Dataset preparation
class spanned_dataset(Dataset):
    def __init__(self, metadata_path, tokenizer, max_length=512):
        self.examples = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.labels_distribution = []  

        with open(metadata_path, "r") as meta_file:
            for line in meta_file:
                entry = json.loads(line.strip())
                labeled_path = entry["labeled_path"]

                with open(labeled_path, "r") as labeled_file:
                    labeled_data = json.load(labeled_file)
                    text = labeled_data["text"]
                    spans = labeled_data["spans"]

                    tokenized = tokenizer(
                        text,
                        return_offsets_mapping=True,
                        truncation=True,
                        padding="max_length",
                        max_length=self.max_length,
                    )
                    offset_mapping = tokenized.pop("offset_mapping")
                    labels = [0] * len(tokenized["input_ids"])

                    for span in spans:
                        if span["label"] == "SYMPTOM":
                            start, end = span["start"], span["end"]
                            token_start = next((i for i, offset in enumerate(offset_mapping) if offset[0] <= start < offset[1]), None)
                            token_end = next((i for i, offset in enumerate(offset_mapping) if offset[0] < end <= offset[1]), None)

                            if token_start is not None and token_end is not None:
                                for i in range(token_start, token_end + 1):
                                    labels[i] = 1

                    self.examples.append({
                        "input_ids": tokenized["input_ids"],
                        "attention_mask": tokenized["attention_mask"],
                        "labels": labels[: self.max_length],
                    })

                    self.labels_distribution.append(sum(labels) / len(labels))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return {key: torch.tensor(val) for key, val in self.examples[idx].items()}

    def get_stratification_labels(self):

        return np.digitize(self.labels_distribution, bins=np.linspace(0, 1, 10))

# Model initialization function
def init_model():
    model = AutoModelForTokenClassification.from_pretrained("SpanBERT/spanbert-large-cased", num_labels=2)
    device0 = torch.device("cuda:0")
    device1 = torch.device("cuda:1")

    model.bert.embeddings.to(device0)
    model.bert.encoder.layer[:12].to(device0)
    model.bert.encoder.layer[12:].to(device1)
    model.classifier.to(device1)
    return model

# Forward pass for model parallelism
def forward_model_parallel(model, input_ids, attention_mask, labels=None):
    device0 = torch.device("cuda:0")
    device1 = torch.device("cuda:1")
    
    attention_mask = attention_mask.to(dtype=torch.float).unsqueeze(1).unsqueeze(2)
    input_ids = input_ids.to(device0)
    attention_mask = attention_mask.to(device0)

    embeddings = model.bert.embeddings(input_ids)
    encoder_outputs = embeddings
    for layer in model.bert.encoder.layer[:12]:
        encoder_outputs = layer(encoder_outputs, attention_mask)[0]

    encoder_outputs = encoder_outputs.to(device1)
    attention_mask = attention_mask.to(device1)

    for layer in model.bert.encoder.layer[12:]:
        encoder_outputs = layer(encoder_outputs, attention_mask)[0]

    logits = model.classifier(encoder_outputs)

    if labels is not None:
        labels = labels.to(device1)
        loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
        return logits, loss

    return logits

def epoch_cycle(model, loader, optimizer=None, scheduler=None, train=False, accumulation_steps=4):
    mode = "Train" if train else "Validation"
    print(f"\n{mode}")
    model.train() if train else model.eval()
    scaler = GradScaler()
    total_loss = 0
    all_predictions, all_labels = [], []
    running_loss = 0

    if train:
        optimizer.zero_grad()

    for batch_idx, batch in enumerate(loader):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"].long()

        with autocast(device_type="cuda"):
            logits, loss = forward_model_parallel(model, input_ids, attention_mask, labels)
            if train:
                loss = loss / accumulation_steps

        if train:
            scaler.scale(loss).backward()
            if ((batch_idx + 1) % accumulation_steps == 0) or (batch_idx + 1 == len(loader)):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

        running_loss += loss.item() * (accumulation_steps if train else 1)
        predictions = logits.argmax(dim=-1).view(-1).cpu().numpy()
        true_labels = labels.view(-1).cpu().numpy()
        all_predictions.extend(predictions)
        all_labels.extend(true_labels)

        if (batch_idx + 1) % 50 == 0:
            avg_loss = running_loss / (batch_idx + 1)
            print(f"Batch {batch_idx + 1}/{len(loader)} - Average Loss: {avg_loss:.4f}")

    metrics = {
        'precision': precision_score(all_labels, all_predictions, average="weighted", zero_division=0),
        'recall': recall_score(all_labels, all_predictions, average="weighted", zero_division=0),
        'f1': f1_score(all_labels, all_predictions, average="weighted", zero_division=0),
        'accuracy': accuracy_score(all_labels, all_predictions),
        'loss': running_loss / len(loader)
    }

    print(f"{mode} Metrics:\nPrecision: {metrics['precision']:.4f}, "
          f"Recall: {metrics['recall']:.4f}, F1-Score: {metrics['f1']:.4f}, "
          f"Accuracy: {metrics['accuracy']:.4f}")
    return metrics

# Load Dataset
metadata_path = "clinical_notes/metadata.jsonl"
tokenizer = AutoTokenizer.from_pretrained("SpanBERT/spanbert-large-cased")
dataset = spanned_dataset(metadata_path, tokenizer)

# Separate test set
indices = list(range(len(dataset)))
stratify_labels = dataset.get_stratification_labels()
train_val_indices, test_indices = train_test_split(
    indices, test_size=0.15, random_state=42,
    stratify=stratify_labels
)

# Create test dataset
test_dataset = torch.utils.data.Subset(dataset, test_indices)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=2)

# Initialize cross-validation
n_splits = 3
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
train_val_stratify_labels = stratify_labels[train_val_indices]

# Store results for each fold
fold_results = []
best_f1 = 0
best_model = None
best_fold = None

# Cross-validation loop
for fold, (train_idx, val_idx) in enumerate(skf.split(train_val_indices, train_val_stratify_labels)):
    print(f"\nFold {fold + 1}/{n_splits}")
    
    # Get actual indices from train_val_indices
    train_indices = [train_val_indices[i] for i in train_idx]
    val_indices = [train_val_indices[i] for i in val_idx]
    
    # Create datasets for this fold
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2)
    
    # Initialize model and optimization components
    model = init_model()
    optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    accumulation_steps = 4
    epochs = 3
    
    num_training_steps = (len(train_loader) // accumulation_steps) * epochs
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )
    
    # Training loop
    best_val_f1 = 0
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        train_metrics = epoch_cycle(
            model, train_loader,
            optimizer=optimizer,
            train=True,
            accumulation_steps=accumulation_steps
        )
        lr_scheduler.step()
        
        with torch.no_grad():
            val_metrics = epoch_cycle(model, val_loader)
            
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
    
    # Evaluate on test set
    print(f"\nEvaluating Fold {fold + 1} on Test Set")
    with torch.no_grad():
        test_metrics = epoch_cycle(model, test_loader)
    
    fold_results.append(test_metrics)
    
    # Save best model based on test F1 score
    if test_metrics['f1'] > best_f1:
        best_f1 = test_metrics['f1']
        best_model = model
        best_fold = fold + 1

# Calculate and print aggregate results
avg_metrics = {
    'precision': np.mean([result['precision'] for result in fold_results]),
    'recall': np.mean([result['recall'] for result in fold_results]),
    'f1': np.mean([result['f1'] for result in fold_results]),
    'accuracy': np.mean([result['accuracy'] for result in fold_results])
}

print("\nAggregate Results Across All Folds (Test Set):")
print(f"Average Precision: {avg_metrics['precision']:.4f}")
print(f"Average Recall: {avg_metrics['recall']:.4f}")
print(f"Average F1-Score: {avg_metrics['f1']:.4f}")
print(f"Average Accuracy: {avg_metrics['accuracy']:.4f}")

print(f"\nBest performing model was from Fold {best_fold} with F1-Score: {best_f1:.4f}")

# Save the best model
print(f"Saving best model from Fold {best_fold}")
best_model.save_pretrained("./SpanBERT-SCM-Large")
tokenizer.save_pretrained("./SpanBERT-SCM-Large")

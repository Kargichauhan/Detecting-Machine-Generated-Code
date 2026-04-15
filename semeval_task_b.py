"""
SemEval-2026 Task 13 — Subtask B
"""

#  1. Install 
# !pip install -q torch transformers datasets scikit-learn pandas numpy

# 2. Imports 
import os
import torch
import pandas as pd
import numpy as np

from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from sklearn.metrics import f1_score, accuracy_score

# 3. GPU check
print(f"GPU Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# 4. Load data 
print("Loading from Google Drive...")
train_df = pd.read_parquet('/content/drive/My Drive/Semeval/task_b_training_set.parquet')
val_df   = pd.read_parquet('/content/drive/My Drive/Semeval/task_b_validation_set.parquet')
test_df  = pd.read_parquet('/content/drive/My Drive/Semeval/task_b_test_set_sample.parquet')

print(f"Train: {len(train_df)} rows")
print(f"Val: {len(val_df)} rows")
print(f"Test: {len(test_df)} rows")
print("\nSuccess! Real data loaded!")

# 5. Load model 
print("Loading CodeBERT model...")

model_name = "microsoft/codebert-base"
tokenizer  = AutoTokenizer.from_pretrained(model_name)
model      = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=11,
    ignore_mismatched_sizes=True
)

device = torch.device("cuda:0")
model  = model.to(device)

print(f"Model loaded: {model_name}")
print(f"Number of labels: 11 (Human + 10 LLM families)")
print(f"Device: {device}")

# 6. Tokenize 
print("Tokenizing training data...")

dataset = DatasetDict({
    'train':      Dataset.from_pandas(train_df),
    'validation': Dataset.from_pandas(val_df)
})

def tokenize_fn(examples):
    return tokenizer(
        examples['code'],
        padding=True,
        truncation=True,
        max_length=512
    )

tokenized_train = dataset['train'].map(
    tokenize_fn,
    batched=True,
    batch_size=1000,
    remove_columns=['code']
)

tokenized_val = dataset['validation'].map(
    tokenize_fn,
    batched=True,
    batch_size=1000,
    remove_columns=['code']
)

print(f"Tokenized train: {len(tokenized_train)} samples")
print(f"Tokenized validation: {len(tokenized_val)} samples")
print("Tokenization complete!")

# 7. Trainer 
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='macro', zero_division=0)
    return {'accuracy': accuracy, 'f1': f1}

training_args = TrainingArguments(
    output_dir="results_task_b",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    logging_steps=100,
    save_steps=1000,
    eval_strategy="steps",
    save_strategy="steps",
    eval_steps=1000,
    fp16=True,
    gradient_checkpointing=True,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    compute_metrics=compute_metrics,
)

print("Trainer created and ready for training!")

# 8. Train 
print("Starting training on Task B...")
trainer.train()
print("\nTraining complete!")

# 9. Predict on test set 
print("Making predictions on test set...")

test_inputs = tokenizer(
    test_df['code'].tolist(),
    padding=True,
    truncation=True,
    max_length=512,
    return_tensors="pt"
)

test_inputs = {k: v.to('cuda:0') for k, v in test_inputs.items()}

model.eval()
with torch.no_grad():
    outputs     = model(**test_inputs)
    logits      = outputs.logits
    predictions = torch.argmax(logits, dim=1).cpu().numpy()

print(f"Predictions made on {len(predictions)} samples")
print(f"Unique classes predicted: {sorted(set(predictions))}")
print(f"First 10 predictions: {predictions[:10]}")
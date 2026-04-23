"""
SemEval-2026 Task 13 — Subtask B: Class-Weighted CodeBERT (Kaggle)
Model : microsoft/codebert-base with inverse-frequency class weighting
Goal  : 11-class attribution (Human + 10 LLM families)
"""

# 1. Imports 
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from collections import Counter

from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
)
from sklearn.metrics import f1_score, accuracy_score, classification_report

# 2. GPU check 
print(f"GPU available : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU name      : {torch.cuda.get_device_name(0)}")
    print(f"GPU memory    : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 3. Load data
train_df = pd.read_parquet('/kaggle/input/competitions/sem-eval-2026-task-13-subtask-b/Task_B/train.parquet')
val_df   = pd.read_parquet('/kaggle/input/competitions/sem-eval-2026-task-13-subtask-b/Task_B/validation.parquet')
test_df  = pd.read_parquet('/kaggle/input/competitions/sem-eval-2026-task-13-subtask-b/Task_B/test.parquet')

print(f"Train : {len(train_df):,} rows  |  columns: {train_df.columns.tolist()}")
print(f"Val   : {len(val_df):,} rows")
print(f"Test  : {len(test_df):,} rows")
print(f"\nLabel distribution (train):\n{train_df['label'].value_counts().sort_index()}")

# 4. Compute inverse-frequency class weights 
label_counts = Counter(train_df['label'].tolist())
n_classes    = 11
total        = sum(label_counts.values())

weights = torch.zeros(n_classes)
for label_id, count in label_counts.items():
    weights[label_id] = total / (n_classes * count)

class_weights = weights.to(device)

print("Class weights (higher = rarer class):")
for i, w in enumerate(weights):
    print(f"  class {i:2d} : weight={w:.4f}  (n={label_counts.get(i, 0):>7,})")

# 5. Load tokenizer and model 
MODEL_NAME = "microsoft/codebert-base"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Loading model...")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=n_classes,
    ignore_mismatched_sizes=True,
).to(device)

print(f"Model loaded on {device}")

# 6. Tokenize 
dataset = DatasetDict({
    'train'     : Dataset.from_pandas(train_df),
    'validation': Dataset.from_pandas(val_df),
})

def tokenize_fn(examples):
    return tokenizer(
        examples['code'],
        padding=False,
        truncation=True,
        max_length=512,
    )

print("Tokenizing train...")
tokenized_train = dataset['train'].map(
    tokenize_fn, batched=True, batch_size=2000,
    remove_columns=['code'], num_proc=2,
)
print("Tokenizing validation...")
tokenized_val = dataset['validation'].map(
    tokenize_fn, batched=True, batch_size=2000,
    remove_columns=['code'], num_proc=2,
)

print(f"Tokenized train : {len(tokenized_train):,} samples")
print(f"Tokenized val   : {len(tokenized_val):,} samples")

# 7. WeightedTrainer 
class WeightedTrainer(Trainer):
    """Trainer subclass that applies inverse-frequency class weights to CE loss."""

    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels  = inputs.pop("labels")
        outputs = model(**inputs)
        logits  = outputs.logits
        loss_fn = nn.CrossEntropyLoss(weight=self.class_weights)
        loss    = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {
        'accuracy' : accuracy_score(labels, predictions),
        'macro_f1' : f1_score(labels, predictions, average='macro',   zero_division=0),
        'wtd_f1'   : f1_score(labels, predictions, average='weighted', zero_division=0),
    }


training_args = TrainingArguments(
    output_dir                  = "/kaggle/working/results_task_b_weighted",
    num_train_epochs            = 1,
    per_device_train_batch_size = 32,
    per_device_eval_batch_size  = 64,
    gradient_accumulation_steps = 2,
    learning_rate               = 2e-5,
    weight_decay                = 0.01,
    warmup_steps                = 250,
    fp16                        = True,
    gradient_checkpointing      = True,
    dataloader_num_workers      = 2,
    dataloader_pin_memory       = True,
    eval_strategy               = "steps",
    save_strategy               = "steps",
    eval_steps                  = 2000,
    save_steps                  = 2000,
    save_total_limit            = 1,
    logging_steps               = 200,
    load_best_model_at_end      = True,
    metric_for_best_model       = "macro_f1",
    greater_is_better           = True,
    report_to                   = "none",
)

trainer = WeightedTrainer(
    model           = model,
    args            = training_args,
    train_dataset   = tokenized_train,
    eval_dataset    = tokenized_val,
    compute_metrics = compute_metrics,
    data_collator   = DataCollatorWithPadding(tokenizer),
    class_weights   = class_weights,
    callbacks       = [EarlyStoppingCallback(early_stopping_patience=2)],
)

print("WeightedTrainer ready.")

# 8. Train 
print("Starting training...")
train_result = trainer.train()
print("\nTraining complete!")
print(train_result.metrics)

# 9. Evaluate on validation set 
val_pred   = trainer.predict(tokenized_val)
val_preds  = np.argmax(val_pred.predictions, axis=1)
val_labels = val_pred.label_ids

print("=" * 60)
print("VALIDATION RESULTS")
print("=" * 60)
print(f"Accuracy : {accuracy_score(val_labels, val_preds):.4f}")
print(f"Macro F1 : {f1_score(val_labels, val_preds, average='macro',   zero_division=0):.4f}")
print(f"Wtd F1   : {f1_score(val_labels, val_preds, average='weighted', zero_division=0):.4f}")
print()
print("Per-class report:")
print(classification_report(val_labels, val_preds, zero_division=0))

# 10. Save model 
SAVE_DIR = "/kaggle/working/codebert_weighted_best"
trainer.save_model(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)
print(f"Model saved to {SAVE_DIR}")

# 11. Inference on test set (batched) 
print("Running inference on test set...")

model.eval()
all_preds = []
BATCH     = 64

codes = test_df['code'].tolist()
for i in range(0, len(codes), BATCH):
    batch  = codes[i : i + BATCH]
    inputs = tokenizer(
        batch,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits
    all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy().tolist())
    if (i // BATCH) % 10 == 0:
        print(f"  {i + len(batch)}/{len(codes)} done")

print(f"\nPredictions made: {len(all_preds)}")
print(f"Unique classes predicted: {sorted(set(all_preds))}")
print(f"Prediction distribution: {Counter(all_preds)}")

# 12. Save submission CSV 
id_col = 'id' if 'id' in test_df.columns else test_df.columns[0]

submission = pd.DataFrame({
    id_col  : test_df[id_col].values,
    'label' : all_preds,
})

OUT = "/kaggle/working/submission_taskb_weighted.csv"
submission.to_csv(OUT, index=False)

print(f"Submission saved → {OUT}")
print(f"Shape: {submission.shape}")
print(submission['label'].value_counts().sort_index())

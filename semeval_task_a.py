"""
SemEval-2026 Task 13 — Subtask A: Multi-View UniXcoder Training
================================================================
Binary classification: Human (0) vs Machine-generated (1) code.

Architecture:
  - Backbone   : microsoft/unixcoder-base
  - Tokenizer  : First+Last truncation (first 256 + last 256 tokens)
  - Views       : original | delexicalized | mixed-content augmented
  - Loss        : CE + symmetric KL (original ↔ delex) + soft KL (original ↔ mixed)
  - Inference  : 3-view logit ensemble

Usage (Colab / Kaggle):
  1. Set DATA_DIR and CHECKPOINT_DIR below.
  2. python semeval_task_a.py
"""

# 1. Install (uncomment if running as script in a fresh env) 
# import subprocess, sys
# subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
#     "transformers", "accelerate", "datasets", "evaluate", "scikit-learn"])

#  2. Imports 
import os
import re
import random
import gc
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    DataCollatorWithPadding,
)
from transformers.modeling_outputs import SequenceClassifierOutput
import evaluate
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, f1_score

# 3. Config — edit these paths 
# Kaggle: DATA_DIR = "/kaggle/input/sem-eval-2026-task-13-subtask-a/Task_A"
# Colab:  DATA_DIR = "/root/.cache/kagglehub/competitions/sem-eval-2026-task-13-subtask-a/Task_A"
DATA_DIR       = "/root/.cache/kagglehub/competitions/sem-eval-2026-task-13-subtask-a/Task_A"
CHECKPOINT_DIR = "/content/drive/MyDrive/semeval_taskA"
SUBMISSION_OUT = "/content/submission.csv"

MODEL_NAME  = "microsoft/unixcoder-base"
SEED        = 42
HALF_LEN    = 256        # first 256 + last 256 = 512 effective tokens
MAX_LENGTH  = HALF_LEN * 2
TOKEN_DROPOUT_RATE = 0.15

# 4. Reproducibility
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
print(f"Checkpoints → {CHECKPOINT_DIR}")

# 5. Load data 
print("\nLoading data...")
train_df = pd.read_parquet(os.path.join(DATA_DIR, "train.parquet"))
val_df   = pd.read_parquet(os.path.join(DATA_DIR, "validation.parquet"))
test_df  = pd.read_parquet(os.path.join(DATA_DIR, "test.parquet"))

print(f"Train : {train_df.shape}  |  Val : {val_df.shape}  |  Test : {test_df.shape}")

assert "code"  in train_df.columns and "label" in train_df.columns
assert "code"  in val_df.columns   and "label" in val_df.columns
assert "code"  in test_df.columns  and "ID"    in test_df.columns

print(f"Label distribution:\n{train_df['label'].value_counts()}")

# 6. 3-Domain Structural Prefix 
# Classifies each snippet into clean / mixed / fragment and prepends a
# structured prefix. Allows the model to learn separate decision boundaries
# per content type — critical given 33.4% mixed-content in the test set.

_code_pattern = re.compile(
    r'[{}\[\]();]|==|!=|<=|>=|->|::|=>|'
    r'\b(if|else|for|while|return|def|class|'
    r'function|func|import|include|var|let|const|int|void)\b'
)
_text_pattern = re.compile(
    r'^#+\s|^//\s*[A-Z]|^/\*|^\s*\*|'
    r'[.!?]\s+[A-Z]|'
    r'\b(the|this|that|you|can|will|should|must|note|example)\b',
    re.IGNORECASE
)

def classify_and_prefix(code: str) -> str:
    if not code or len(code.strip()) < 10:
        return f"[TYPE:fragment LEN:short CLASS:0 LOOP:0 FUNC:0] {code}"

    lines      = [l for l in code.strip().splitlines() if l.strip()]
    n          = max(len(lines), 1)
    code_ratio = sum(1 for l in lines if _code_pattern.search(l)) / n
    text_ratio = sum(1 for l in lines if _text_pattern.search(l)) / n

    if code_ratio >= 0.5:
        content_type = "clean"
    elif text_ratio >= 0.4 or code_ratio < 0.2:
        content_type = "fragment"
    else:
        content_type = "mixed"

    total_lines = len(code.strip().splitlines())
    complexity  = "short" if total_lines < 20 else ("medium" if total_lines < 60 else "long")
    has_class   = int(bool(re.search(r'\bclass\b', code)))
    has_loop    = int(bool(re.search(r'\bfor\b|\bwhile\b', code)))
    has_func    = int(bool(re.search(
        r'\bdef\b|\bfunction\b|\bfunc\b|\bvoid\b|\bint\b\s+\w+\s*\(', code)))

    prefix = (f"[TYPE:{content_type} LEN:{complexity} "
              f"CLASS:{has_class} LOOP:{has_loop} FUNC:{has_func}]")
    return f"{prefix} {code}"

for df in [train_df, val_df, test_df]:
    df["code"] = [classify_and_prefix(c) if c else "" for c in df["code"]]

print(f"\nSample prefix: {train_df['code'].iloc[0][:100]}")
test_types = pd.Series(test_df["code"]).str.extract(r'\[TYPE:(\w+)')[0]
print(f"Test content type distribution:\n{test_types.value_counts()}")

#  7. Delexicalization 
# Replaces identifiers / strings / numbers with generic tokens.
# Forces the model to rely on structural patterns rather than surface lexical
# cues that vary across languages — used as a second view during training.

_re_block = re.compile(r"/\*.*?\*/", re.S)
_re_line  = re.compile(r"//.*?$|#.*?$", re.M)
_re_str   = re.compile(r"\"(?:\\.|[^\"\\])*\"|'(?:\\.|[^'\\])*'", re.S)
_re_num   = re.compile(r"\b\d+(\.\d+)?\b")
_re_ident = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")

def delexicalize(code: str) -> str:
    if not code:
        return ""
    s = _re_block.sub(" ", code)
    s = _re_line.sub(" ", s)
    s = _re_str.sub(" STR ", s)
    s = _re_num.sub(" NUM ", s)
    s = _re_ident.sub(" ID ", s)
    return re.sub(r"[ \t]+", " ", s).strip()

# 8. Mixed-Content Augmentation 
# Injects generic text fragments into training snippets at 40% probability.
# Bridges the gap between training (no mixed samples) and test (33.4% mixed).

FILLER_TEXTS = [
    "function description", "helper method", "main logic", "initialization",
    "return value", "loop body", "base case", "edge case",
    "input handling", "output format",
]

def mix_code_with_text(code: str, mix_prob: float = 0.4) -> str:
    if random.random() > mix_prob or not code:
        return code
    lines = code.splitlines()
    if len(lines) < 3:
        return code
    for _ in range(random.randint(1, 3)):
        pos  = random.randint(0, len(lines))
        lines.insert(pos, f"# {random.choice(FILLER_TEXTS)}")
    return "\n".join(lines)

# 9. Token Dropout 
# Randomly masks 15% of non-special tokens during training.
# Prevents over-reliance on language-specific keywords that don't generalise.

def apply_token_dropout(
    input_ids: torch.Tensor,
    mask_token_id: int,
    pad_token_id: int,
    rate: float = TOKEN_DROPOUT_RATE,
) -> torch.Tensor:
    result       = input_ids.clone()
    special_mask = (result <= 3) | (result == pad_token_id)
    drop_mask    = (torch.rand_like(result, dtype=torch.float) < rate) & ~special_mask
    result[drop_mask] = mask_token_id
    return result

#  10. Tokenizer & First+Last Encoding
# UniXcoder trained on CodeSearchNet across 6 languages → best cross-lingual
# transfer for unseen language generalisation.
# First+Last truncation: preserves opening signatures AND closing logic.

print("\nLoading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

def first_last_encode(code: str) -> Dict[str, List[int]]:
    tokens = tokenizer.encode(code, truncation=False, add_special_tokens=False)
    if len(tokens) <= MAX_LENGTH - 2:
        enc = tokenizer(code, truncation=True, max_length=MAX_LENGTH, padding=False)
        return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}
    half           = HALF_LEN - 1
    combined       = tokens[:half] + tokens[-half:]
    input_ids      = [tokenizer.cls_token_id] + combined + [tokenizer.sep_token_id]
    attention_mask = [1] * len(input_ids)
    return {"input_ids": input_ids, "attention_mask": attention_mask}

def tokenize_with_aug(batch):
    """Tokenize original + delex + mixed views for training/validation."""
    results = {
        "input_ids": [], "attention_mask": [],
        "input_ids_aug": [], "attention_mask_aug": [],
        "input_ids_mix": [], "attention_mask_mix": [],
    }
    for code in batch["code"]:
        code  = code if code else ""
        enc_main = first_last_encode(code)
        enc_aug  = first_last_encode(delexicalize(code))
        enc_mix  = first_last_encode(mix_code_with_text(code, mix_prob=0.4))
        results["input_ids"].append(enc_main["input_ids"])
        results["attention_mask"].append(enc_main["attention_mask"])
        results["input_ids_aug"].append(enc_aug["input_ids"])
        results["attention_mask_aug"].append(enc_aug["attention_mask"])
        results["input_ids_mix"].append(enc_mix["input_ids"])
        results["attention_mask_mix"].append(enc_mix["attention_mask"])
    return results

def tokenize_test_fn(batch):
    """Tokenize original + delex views for test-time ensembling."""
    results = {
        "input_ids": [], "attention_mask": [],
        "input_ids_delex": [], "attention_mask_delex": [],
    }
    for code in batch["code"]:
        code  = code if code else ""
        enc_main  = first_last_encode(code)
        enc_delex = first_last_encode(delexicalize(code))
        results["input_ids"].append(enc_main["input_ids"])
        results["attention_mask"].append(enc_main["attention_mask"])
        results["input_ids_delex"].append(enc_delex["input_ids"])
        results["attention_mask_delex"].append(enc_delex["attention_mask"])
    return results

# Token length stats
sample_lengths = [
    len(tokenizer.encode(c, truncation=False, add_special_tokens=False))
    for c in train_df["code"].sample(min(2000, len(train_df)), random_state=SEED)
]
print(f"Token length — p50={np.percentile(sample_lengths,50):.0f}  "
      f"p90={np.percentile(sample_lengths,90):.0f}  "
      f"max={max(sample_lengths)}")

# Build & tokenize HuggingFace datasets
ds = DatasetDict({
    "train":      Dataset.from_pandas(train_df, preserve_index=False),
    "validation": Dataset.from_pandas(val_df,   preserve_index=False),
    "test":       Dataset.from_pandas(test_df,  preserve_index=False),
})

remove_train = [c for c in ds["train"].column_names      if c not in ["ID", "code", "label"]]
remove_val   = [c for c in ds["validation"].column_names if c not in ["ID", "code", "label"]]
remove_test  = [c for c in ds["test"].column_names       if c not in ["ID", "code"]]

print("\nTokenizing train...")
train_tok = ds["train"].map(tokenize_with_aug, batched=True, batch_size=256,
                             remove_columns=remove_train, num_proc=1)
train_tok = train_tok.rename_column("label", "labels")

print("Tokenizing validation...")
val_tok = ds["validation"].map(tokenize_with_aug, batched=True, batch_size=256,
                                remove_columns=remove_val, num_proc=1)
val_tok = val_tok.rename_column("label", "labels")

print("Tokenizing test...")
test_tok = ds["test"].map(tokenize_test_fn, batched=True, batch_size=256,
                           remove_columns=remove_test, num_proc=1)

tokenized_ds = DatasetDict({"train": train_tok, "validation": val_tok, "test": test_tok})
print(tokenized_ds)

# 11. Dual-View Collator

@dataclass
class DualViewCollator:
    """
    Pads original, delex, and mixed views independently.
    Falls back gracefully when aug columns are absent (test time).
    """
    tokenizer: Any

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        has_aug = "input_ids_aug" in features[0]
        has_mix = "input_ids_mix" in features[0]
        main, aug, mix = [], [], []
        for f in features:
            main.append({
                "input_ids":      f["input_ids"],
                "attention_mask": f["attention_mask"],
                **( {"labels": f["labels"]} if "labels" in f else {} ),
            })
            if has_aug:
                aug.append({"input_ids": f["input_ids_aug"],
                            "attention_mask": f["attention_mask_aug"]})
            if has_mix:
                mix.append({"input_ids": f["input_ids_mix"],
                            "attention_mask": f["attention_mask_mix"]})
        batch = self.tokenizer.pad(main, return_tensors="pt")
        if has_aug:
            ab = self.tokenizer.pad(aug, return_tensors="pt")
            batch["input_ids_aug"]      = ab["input_ids"]
            batch["attention_mask_aug"] = ab["attention_mask"]
        if has_mix:
            mb = self.tokenizer.pad(mix, return_tensors="pt")
            batch["input_ids_mix"]      = mb["input_ids"]
            batch["attention_mask_mix"] = mb["attention_mask"]
        return batch

data_collator = DualViewCollator(tokenizer=tokenizer)

# 12. Model 
print("\nLoading model...")
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
model = model.to(device)

# 13. Metrics
f1_metric  = evaluate.load("f1")
acc_metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "macro_f1": f1_metric.compute(predictions=preds, references=labels, average="macro")["f1"],
        "acc":      acc_metric.compute(predictions=preds, references=labels)["accuracy"],
    }

# 14. ConsistencyTrainer (CE + Symmetric KL) 
# L = CE(original) + λ·SymKL(original ∥ delex) + (λ/2)·SymKL(original ∥ mixed)
# KL enforces prediction consistency across views rather than correctness of aug
# labels, forcing the model to rely on structure not surface cues.

class ConsistencyTrainer(Trainer):
    """
    Fused forward pass over [original; delex; mixed] concatenated batch.
    CE loss on original logits + symmetric KL divergence between views.
    """
    def __init__(self, *args, lambda_kl: float = 0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.lambda_kl      = lambda_kl
        self.mask_token_id  = tokenizer.mask_token_id or tokenizer.unk_token_id
        self.pad_token_id   = tokenizer.pad_token_id

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels             = inputs.pop("labels",             None)
        input_ids_aug      = inputs.pop("input_ids_aug",      None)
        attention_mask_aug = inputs.pop("attention_mask_aug", None)
        input_ids_mix      = inputs.pop("input_ids_mix",      None)
        attention_mask_mix = inputs.pop("attention_mask_mix", None)

        # Eval / test — plain forward pass
        if input_ids_aug is None or labels is None:
            logits = model(**inputs).logits
            loss   = F.cross_entropy(logits, labels) if labels is not None else None
            out    = SequenceClassifierOutput(loss=loss, logits=logits)
            return (loss, out) if return_outputs else out.loss

        # Token dropout on original view
        dropped_ids = apply_token_dropout(
            inputs["input_ids"],
            mask_token_id=self.mask_token_id,
            pad_token_id=self.pad_token_id,
        )

        def pad_to(tensor, length, val):
            if tensor.shape[1] < length:
                tensor = F.pad(tensor, (0, length - tensor.shape[1]), value=val)
            return tensor

        lengths = [dropped_ids.shape[1], input_ids_aug.shape[1]]
        if input_ids_mix is not None:
            lengths.append(input_ids_mix.shape[1])
        max_len = max(lengths)

        dropped_ids        = pad_to(dropped_ids,              max_len, self.pad_token_id)
        input_ids_aug      = pad_to(input_ids_aug,            max_len, self.pad_token_id)
        attn_main          = pad_to(inputs["attention_mask"], max_len, 0)
        attention_mask_aug = pad_to(attention_mask_aug,       max_len, 0)

        all_ids   = [dropped_ids, input_ids_aug]
        all_masks = [attn_main,   attention_mask_aug]

        if input_ids_mix is not None:
            input_ids_mix      = pad_to(input_ids_mix,      max_len, self.pad_token_id)
            attention_mask_mix = pad_to(attention_mask_mix, max_len, 0)
            all_ids.append(input_ids_mix)
            all_masks.append(attention_mask_mix)

        combined_out = model(
            input_ids=torch.cat(all_ids,   dim=0),
            attention_mask=torch.cat(all_masks, dim=0),
        )

        bsz        = dropped_ids.size(0)
        logits     = combined_out.logits[:bsz]
        logits_aug = combined_out.logits[bsz : 2 * bsz]

        loss_ce = F.cross_entropy(logits, labels, label_smoothing=0.1)

        p = F.log_softmax(logits,     dim=-1)
        q = F.log_softmax(logits_aug, dim=-1)
        loss_kl_delex = (
            F.kl_div(p, q.detach().exp(), reduction="batchmean") +
            F.kl_div(q, p.detach().exp(), reduction="batchmean")
        ) / 2.0

        loss = loss_ce + self.lambda_kl * loss_kl_delex

        if input_ids_mix is not None:
            logits_mix  = combined_out.logits[2 * bsz :]
            r           = F.log_softmax(logits_mix, dim=-1)
            loss_kl_mix = (
                F.kl_div(p, r.detach().exp(), reduction="batchmean") +
                F.kl_div(r, p.detach().exp(), reduction="batchmean")
            ) / 2.0
            loss = loss + (self.lambda_kl * 0.5) * loss_kl_mix

        out = SequenceClassifierOutput(loss=loss_ce, logits=logits)
        return (loss, out) if return_outputs else loss

# 15. Training Arguments 
training_args = TrainingArguments(
    output_dir                  = CHECKPOINT_DIR,
    remove_unused_columns       = False,
    learning_rate               = 2e-5,
    per_device_train_batch_size = 16,
    per_device_eval_batch_size  = 32,
    gradient_accumulation_steps = 4,
    num_train_epochs            = 1,
    weight_decay                = 0.01,
    warmup_steps                = 470,
    fp16                        = True,
    dataloader_num_workers      = 2,
    eval_strategy               = "steps",
    eval_steps                  = 1000,
    save_steps                  = 1000,
    save_total_limit            = 2,
    logging_steps               = 200,
    load_best_model_at_end      = True,
    metric_for_best_model       = "macro_f1",
    greater_is_better           = True,
    report_to                   = "none",
    seed                        = SEED,
)

trainer = ConsistencyTrainer(
    model            = model,
    args             = training_args,
    train_dataset    = tokenized_ds["train"],
    eval_dataset     = tokenized_ds["validation"],
    processing_class = tokenizer,
    data_collator    = data_collator,
    compute_metrics  = compute_metrics,
    callbacks        = [EarlyStoppingCallback(early_stopping_patience=2)],
    lambda_kl        = 0.5,
)

# 16. Train 
print("\nStarting training...")
train_result = trainer.train(resume_from_checkpoint=True)
print("Train metrics:", train_result.metrics)

trainer.save_model(os.path.join(CHECKPOINT_DIR, "best_model"))
tokenizer.save_pretrained(os.path.join(CHECKPOINT_DIR, "best_model"))
print(f"Model saved → {CHECKPOINT_DIR}/best_model")

eval_result = trainer.evaluate()
print("Validation:", eval_result)

# 17. Inference — 3-view ensemble 
# Averages logits from: original | delexicalized | mixed-content views.
# Zero-cost ensemble at test time.

print("\nTokenizing mixed test view...")

def tokenize_mixed_test(batch):
    results = {"input_ids": [], "attention_mask": []}
    for code in batch["code"]:
        enc = first_last_encode(mix_code_with_text(code, mix_prob=1.0))
        results["input_ids"].append(enc["input_ids"])
        results["attention_mask"].append(enc["attention_mask"])
    return results

test_mixed_ds  = Dataset.from_pandas(test_df.copy(), preserve_index=False)
remove_cols    = [c for c in test_mixed_ds.column_names if c not in ["ID", "code"]]
test_mixed_tok = test_mixed_ds.map(
    tokenize_mixed_test, batched=True, batch_size=64,
    remove_columns=remove_cols, num_proc=1
)

print("Predicting — original view...")
orig_logits = trainer.predict(tokenized_ds["test"]).predictions

print("Predicting — delex view...")
test_delex_ds = (
    tokenized_ds["test"]
    .remove_columns([
        c for c in tokenized_ds["test"].column_names
        if c not in ["input_ids_delex", "attention_mask_delex"]
    ])
    .rename_column("input_ids_delex",      "input_ids")
    .rename_column("attention_mask_delex", "attention_mask")
)
delex_logits = trainer.predict(test_delex_ds).predictions

print("Predicting — mixed view...")
mixed_logits = trainer.predict(test_mixed_tok).predictions

# Equal-weight 3-way ensemble
ensemble_logits = (orig_logits + delex_logits + mixed_logits) / 3.0
test_preds      = np.argmax(ensemble_logits, axis=-1).astype(int)

print(f"\nTest prediction distribution:")
print(pd.Series(test_preds).value_counts(normalize=True).mul(100).round(1))

# 18. Save submission 
submission = pd.DataFrame({
    "ID":    tokenized_ds["test"]["ID"],
    "label": test_preds,
})
submission.to_csv(SUBMISSION_OUT, index=False)
submission.to_csv(os.path.join(CHECKPOINT_DIR, "submission.csv"), index=False)
print(f"\nSubmission saved → {SUBMISSION_OUT}")
print(submission["label"].value_counts())
print(submission["label"].value_counts(normalize=True).mul(100).round(1))

# 19. Validation report & confusion matrix 
print("\nValidation classification report:")
val_pred   = trainer.predict(tokenized_ds["validation"])
val_preds  = np.argmax(val_pred.predictions, axis=-1)
val_labels = val_pred.label_ids

print(classification_report(val_labels, val_preds, digits=4))

for normalize, fmt, title in [
    (None,   "d",   "Confusion Matrix (Validation)"),
    ("true", ".2f", "Confusion Matrix (Validation, normalized)"),
]:
    cm   = confusion_matrix(val_labels, val_preds, labels=[0, 1], normalize=normalize)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Human(0)", "Machine(1)"])
    plt.figure(figsize=(5, 5))
    disp.plot(values_format=fmt)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(CHECKPOINT_DIR, f"confusion_matrix_{'norm' if normalize else 'raw'}.png"),
                dpi=150, bbox_inches="tight")
    plt.show()

print("\nDone.")
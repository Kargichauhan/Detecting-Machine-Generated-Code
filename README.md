# UCSC-NLP at SemEval-2026 Task 13: Multi-Lingual Machine-Generated Code Detection

> **Paper:** *Multi-View Generalization and Diagnostic Analysis of Machine-Generated Code Detection*  
> **Authors:** Kargi Chauhan\*, Sadiba Nusrat Nur\* (University of California, Santa Cruz)  
> **Task:** SemEval-2026 Task 13 — Subtask A (binary detection) & Subtask B (multi-class attribution)  
> \* Equal contribution

---

## Overview

This repository contains the code for our SemEval-2026 Task 13 system, which addresses the problem of detecting and attributing machine-generated code. We participate in two subtasks:

- **Subtask A:** Binary classification — human-written vs. machine-generated code, with evaluation under cross-lingual and cross-domain distribution shift.
- **Subtask B:** Multi-class attribution — assigning code to one of 11 classes (human or one of 10 LLM families), framed as a diagnostic baseline to isolate the effect of severe class imbalance.

---

## Results

| Subtask | Split | Accuracy | Macro F1 | AUC |
|---|---|---|---|---|
| A (Binary) | Validation | 0.993 | 0.993 | 0.999 |
| A (Binary) | Test | 0.892 | 0.845 | 0.843 |
| B (Multi-class) | Validation | 0.884 | 0.086 | — |
| B (Multi-class) | Test | 0.884 | 0.086 | — |


## Repository Structure

```
.
├── Semeval_A.ipynb   # Subtask A: UniXcoder multi-view training pipeline
├── Semeval_B.ipynb                # Subtask B: CodeBERT diagnostic baseline
```

---

## Subtask A: Multi-View Generalization

**Notebook:** `Semeval_A.ipynb`  
**Model:** `microsoft/unixcoder-base`  


## Subtask B: Diagnostic Baseline

**Notebook:** `Semeval_B.ipynb`  
**Model:** `microsoft/codebert-base`  

## Data

Data is sourced from the [SemEval-2026 Task 13 competition on Kaggle](https://www.kaggle.com/competitions/sem-eval-2026-task-13-subtask-a) and the DROID resource suite (Orel et al., 2025).

- Subtask A data is loaded via `kagglehub` from the Kaggle competition.
- Subtask B data (`.parquet` files) is loaded from Google Drive.

---

## Setup

### Requirements

```bash
pip install torch transformers datasets scikit-learn pandas numpy evaluate accelerate
```

### Running

1. **Subtask A:** Open `Semeval_A.ipynb` in Kaggle or Colab. Mount Google Drive for checkpoint saving. Run cells in order.
2. **Subtask B:** Open `Semeval_B.ipynb` in Colab with an A100 GPU. Mount Google Drive with the Task B parquet files at `My Drive/Semeval/`. Run cells in order.

---

## Citation

If you use this work, please cite:

```bibtex
@inproceedings{chauhan-nur-2026-ucscnlp,
  title     = {{UCSC-NLP} at {SemEval}-2026 Task 13: Multi-View Generalization and
               Diagnostic Analysis of Machine-Generated Code Detection},
  author    = {Chauhan, Kargi and Nur, Sadiba Nusrat},
  booktitle = {Proceedings of the 20th International Workshop on Semantic Evaluation (SemEval-2026)},
  year      = {2026},
}
```

---

## License

This code is released for research purposes. See individual model licenses for UniXcoder and CodeBERT at the [Hugging Face model hub](https://huggingface.co/microsoft).

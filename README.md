# UCSC-NLP at SemEval-2026 Task 13: Multilingual Machine-Generated Code Detection

> **Paper:** *Multi-View Generalization and Diagnostic Analysis of Machine-Generated Code Detection*  
> **Authors:** Kargi Chauhan\*, Sadiba Nusrat Nur\* — University of California, Santa Cruz  
> **Task:** SemEval-2026 Task 13 · Subtask A (binary) · Subtask B (multi-class)  
> \* Equal contribution

---

## Results

| Subtask | System | Split | Accuracy | Macro F1 | Wtd. F1 |
|---|---|---|---|---|---|
| A (Binary) | Multi-view UniXcoder | Val | 0.993 | 0.993 | — |
| A (Binary) | Multi-view UniXcoder | Test | 0.892 | 0.845 | — |
| B (Multi-class) | Diagnostic baseline | Val | 0.884 | 0.086 | 0.876 |
| B (Multi-class) | Class-weighted CodeBERT | Val | 0.787 | 0.345 | 0.836 |

---

## Repository Structure

```
.
├── semeval_task_a.py      # Subtask A — multi-view UniXcoder training (Colab/Kaggle)
├── semeval_task_b.py      # Subtask B — class-weighted CodeBERT training (Kaggle)
└── README.md
```

Each `.py` file is a direct conversion of the original notebook and can be run as a script or pasted cell-by-cell into a Colab/Kaggle notebook.

---

## Data

Data is sourced from the SemEval-2026 Task 13 competition on Kaggle:

- **Subtask A:** [sem-eval-2026-task-13-subtask-a](https://www.kaggle.com/competitions/sem-eval-2026-task-13-subtask-a)
- **Subtask B:** [sem-eval-2026-task-13-subtask-b](https://www.kaggle.com/competitions/sem-eval-2026-task-13-subtask-b)

Both datasets are in `.parquet` format with `code`, `label`, and `ID` columns.

---

## Requirements

```bash
pip install transformers accelerate datasets evaluate scikit-learn \
            peft torch pandas numpy matplotlib
```

Tested with:
- Python 3.12
- PyTorch 2.x
- transformers ≥ 4.40
- GPU: NVIDIA T4 or A100 (recommended)

---

## Subtask A — Multi-View UniXcoder

**File:** `semeval_task_a.py`

### Running on Kaggle

1. Add the Subtask A competition dataset to your notebook
2. Update `DATA_DIR` at the top of the file:
   ```python
   DATA_DIR = "/kaggle/input/sem-eval-2026-task-13-subtask-a/Task_A"
   ```
3. Set `CHECKPOINT_DIR` to wherever you want checkpoints saved:
   ```python
   CHECKPOINT_DIR = "/kaggle/working/semeval_taskA"
   ```
4. Run:
   ```bash
   python semeval_task_a.py
   ```

### Running on Colab

1. Mount Google Drive in a cell:
   ```python
   from google.colab import drive
   drive.mount("/content/drive")
   ```
2. Download data via `kagglehub`:
   ```python
   import kagglehub
   kagglehub.login()
   kagglehub.competition_download("sem-eval-2026-task-13-subtask-a")
   ```
3. Leave `DATA_DIR` as the default (it points to the kagglehub cache) or update to your path
4. Run the script or paste cells into Colab

### Running as a notebook

Paste each numbered section (separated by `# ── N.` comments) into its own notebook cell. The sections map 1:1 to the original notebook cells.


## Subtask B — Class-Weighted CodeBERT

**File:** `semeval_task_b.py`

### Running on Kaggle

1. Add the Subtask B competition dataset to your notebook
2. The data paths are already set for Kaggle — no changes needed:
   ```python
   '/kaggle/input/competitions/sem-eval-2026-task-13-subtask-b/Task_B/train.parquet'
   ```
3. Run:
   ```bash
   python semeval_task_b.py
   ```

### Running on Colab

Update the three data paths in section 3:
```python
train_df = pd.read_parquet('/content/drive/My Drive/Semeval/task_b_training_set.parquet')
val_df   = pd.read_parquet('/content/drive/My Drive/Semeval/task_b_validation_set.parquet')
test_df  = pd.read_parquet('/content/drive/My Drive/Semeval/task_b_test_set_sample.parquet')
```
And update the output paths in sections 10 and 12 from `/kaggle/working/` to `/content/`.

### Running as a notebook

Paste each numbered section into its own notebook cell. The 12 sections map directly to the original 12 notebook cells.


### Expected outputs

- `results_task_b_weighted/` — training checkpoints
- `codebert_weighted_best/` — best model checkpoint
- `submission_taskb_weighted.csv` — test set predictions

---

## Reproducing the Paper Results

### Subtask A (0.845 test macro F1)

```bash
# On Kaggle with T4 GPU 
python semeval_task_a.py
```

The 3-view ensemble at inference is automatic. Download `submission.csv` from the working directory and submit to the Kaggle competition.

### Subtask B (0.345 val macro F1)

```bash
# On Kaggle with T4 GPU 
python semeval_task_b.py
```

---

## Citation

```bibtex
@inproceedings{chauhan-nur-2026-ucscnlp,
  title     = {{UCSC-NLP} at {SemEval}-2026 Task 13: Multi-View Generalization and
               Diagnostic Analysis of Multilingual Machine-Generated Code Detection},
  author    = {Chauhan, Kargi and Nur, Sadiba Nusrat},
  booktitle = {Proceedings of the 20th International Workshop on Semantic Evaluation (SemEval-2026)},
  year      = {2026},
}
```

---

## License

Released for research purposes. Model weights are subject to their respective licenses on Hugging Face:
- [microsoft/unixcoder-base](https://huggingface.co/microsoft/unixcoder-base)
- [microsoft/codebert-base](https://huggingface.co/microsoft/codebert-base)

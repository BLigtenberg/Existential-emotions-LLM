import pandas as pd
import numpy as np
import os
from datasets import Dataset
from setfit import SetFitModel, Trainer, TrainingArguments
from setfit.model_card import SetFitModelCardData
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    confusion_matrix, classification_report
)
from sklearn.model_selection import StratifiedKFold
# pyrefly: ignore [missing-import]
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression

# Patch SetFitModelCardData to only pass parameters that have init=True
original_init = SetFitModelCardData.__init__
def patched_init(self, *args, **kwargs):
    valid_fields = [k for k, field in self.__dataclass_fields__.items() if field.init]
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_fields}
    original_init(self, *args, **filtered_kwargs)

SetFitModelCardData.__init__ = patched_init


# ─────────────────────────────────────────────
# Helper: print per-label F-score table
# ─────────────────────────────────────────────
def print_per_label_scores(y_true, y_pred, label_columns, title="Per-Label Scores"):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    header = f"{'Label':<25} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}"
    print(header)
    print("-" * 65)
    for i, label in enumerate(label_columns):
        col_true = np.array(y_true)[:, i]
        col_pred = np.array(y_pred)[:, i]
        p  = precision_score(col_true, col_pred, zero_division=0)
        r  = recall_score(col_true, col_pred, zero_division=0)
        f1 = f1_score(col_true, col_pred, zero_division=0)
        support = int(col_true.sum())
        print(f"{label:<25} {p:>10.4f} {r:>10.4f} {f1:>10.4f} {support:>10}")

    # Macro / micro averages
    print("-" * 65)
    macro_f1 = f1_score(y_true, y_pred, average="macro",  zero_division=0)
    micro_f1 = f1_score(y_true, y_pred, average="micro",  zero_division=0)
    print(f"{'Macro avg':<25} {'':>10} {'':>10} {macro_f1:>10.4f}")
    print(f"{'Micro avg':<25} {'':>10} {'':>10} {micro_f1:>10.4f}")
    print(f"{'='*60}\n")


# ─────────────────────────────────────────────
# Helper: print per-label confusion matrices
# ─────────────────────────────────────────────
def print_per_label_confusion_matrices(y_true, y_pred, label_columns):
    print(f"\n{'='*72}")
    print("  Per-Label Confusion Matrices (binary: 1 vs 0)")
    print(f"{'='*72}")
    header = f"{'Label':<25} {'TN':>8} {'FP':>8} {'FN':>8} {'TP':>8}"
    print(header)
    print("-" * 72)
    for i, label in enumerate(label_columns):
        col_true = np.array(y_true)[:, i]
        col_pred = np.array(y_pred)[:, i]
        cm = confusion_matrix(col_true, col_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (cm[0, 0], 0, 0, 0)
        print(f"{label:<25} {tn:>8} {fp:>8} {fn:>8} {tp:>8}")
    print(f"{'='*72}\n")


# ─────────────────────────────────────────────
# Helper: build a fresh model with balanced head
# ─────────────────────────────────────────────
def make_model():
    """Returns a SetFitModel with a class_weight='balanced' logistic head."""
    m = SetFitModel.from_pretrained(
        "sentence-transformers/all-MiniLM-L6-v2",
        multi_target_strategy="one-vs-rest",
    )
    # Replace the default head with a balanced one
    m.model_head = OneVsRestClassifier(
        LogisticRegression(class_weight="balanced", max_iter=1000)
    )
    return m


# ─────────────────────────────────────────────
# Helper: multi-label stratified K-Fold CV
# ─────────────────────────────────────────────
def run_stratified_kfold_cv(full_df, label_columns, n_splits=5):
    """
    Uses MultilabelStratifiedKFold (iterative stratification) to ensure
    each fold preserves the label distribution of every label independently.
    This is the standard approach for multi-label classification.
    """
    print(f"\n{'='*60}")
    print(f"  Multi-Label Stratified {n_splits}-Fold Cross-Validation")
    print(f"  (iterative stratification — per-label balanced folds)")
    print(f"{'='*60}")

    texts      = full_df["Text"].tolist()
    labels     = full_df[label_columns].values.tolist()
    label_mat  = full_df[label_columns].values  # shape (n_samples, n_labels)

    mskf   = MultilabelStratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    splits = list(mskf.split(texts, label_mat))

    fold_macro_f1s     = []
    fold_micro_f1s     = []
    fold_weighted_f1s  = []
    # per-label F1 across folds: shape (n_folds, n_labels)
    fold_label_f1s = []

    for fold, (train_idx, val_idx) in enumerate(splits, start=1):
        print(f"\n  --- Fold {fold}/{n_splits} ---")

        fold_train_texts  = [texts[i]  for i in train_idx]
        fold_train_labels = [labels[i] for i in train_idx]
        fold_val_texts    = [texts[i]  for i in val_idx]
        fold_val_labels   = [labels[i] for i in val_idx]

        fold_train_ds = Dataset.from_dict({"text": fold_train_texts, "label": fold_train_labels})
        fold_val_ds   = Dataset.from_dict({"text": fold_val_texts,   "label": fold_val_labels})

        fold_model = make_model()

        fold_args = TrainingArguments(
            batch_size=8,
            num_epochs=5,
            num_iterations=2,
        )

        fold_trainer = Trainer(
            model=fold_model,
            args=fold_args,
            train_dataset=fold_train_ds,
            eval_dataset=fold_val_ds,
        )
        fold_trainer.train()

        # Predict on validation fold
        preds = fold_model.predict(fold_val_texts)
        if hasattr(preds, "tolist"):
            preds = preds.tolist()
        y_true = fold_val_labels
        y_pred = preds

        macro_f1    = f1_score(y_true, y_pred, average="macro",    zero_division=0)
        micro_f1    = f1_score(y_true, y_pred, average="micro",    zero_division=0)
        weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        fold_macro_f1s.append(macro_f1)
        fold_micro_f1s.append(micro_f1)
        fold_weighted_f1s.append(weighted_f1)

        # Collect per-label F1 for this fold
        label_f1s = [
            f1_score(np.array(y_true)[:, i], np.array(y_pred)[:, i], zero_division=0)
            for i in range(len(label_columns))
        ]
        fold_label_f1s.append(label_f1s)

        print(f"  Fold {fold} — Macro F1: {macro_f1:.4f} | Micro F1: {micro_f1:.4f} | Weighted F1: {weighted_f1:.4f}")
        print_per_label_scores(y_true, y_pred, label_columns,
                               title=f"Fold {fold} Per-Label Scores")
        print_per_label_confusion_matrices(y_true, y_pred, label_columns)

    # ── Table 1: Mean & SD across folds per metric ───────────────
    fold_label_f1s = np.array(fold_label_f1s)   # shape (n_folds, n_labels)

    print(f"\n{'='*50}")
    print(f"  TABLE 1 — Cross-Validation F1 Summary ({n_splits} Folds)")
    print(f"{'='*50}")
    print(f"  {'Metric':<20} {'Mean':>8} {'SD':>8}")
    print(f"  {'-'*38}")
    print(f"  {'Macro F1':<20} {np.mean(fold_macro_f1s):>8.2f} {np.std(fold_macro_f1s):>8.2f}")
    print(f"  {'Micro F1':<20} {np.mean(fold_micro_f1s):>8.2f} {np.std(fold_micro_f1s):>8.2f}")
    print(f"  {'Weighted F1':<20} {np.mean(fold_weighted_f1s):>8.2f} {np.std(fold_weighted_f1s):>8.2f}")
    print(f"{'='*50}\n")

    # ── Table 2: Average F1 per label across folds ────────────────
    print(f"{'='*62}")
    print(f"  TABLE 2 — Average F1 per Label Across {n_splits} Folds")
    print(f"{'='*62}")
    print(f"  {'Label':<25} {'Mean F1':>10} {'SD':>10}")
    print(f"  {'-'*47}")
    for i, label in enumerate(label_columns):
        col = fold_label_f1s[:, i]
        print(f"  {label:<25} {np.mean(col):>10.4f} {np.std(col):>10.4f}")
    print(f"  {'-'*47}")
    overall_mean = np.mean(fold_label_f1s)
    overall_sd   = np.std(np.mean(fold_label_f1s, axis=1))  # SD across fold-level macro F1s
    print(f"  {'Macro avg':<25} {overall_mean:>10.4f} {overall_sd:>10.4f}")
    print(f"{'='*62}\n")

    # ── Save CV results to CSV ────────────────────────────────────
    fold_rows = []
    for i in range(n_splits):
        row = {
            "fold":        i + 1,
            "macro_f1":    fold_macro_f1s[i],
            "micro_f1":    fold_micro_f1s[i],
            "weighted_f1": fold_weighted_f1s[i],
        }
        for j, label in enumerate(label_columns):
            row[label] = fold_label_f1s[i, j]
        fold_rows.append(row)
    # Mean and SD summary rows
    mean_row = {"fold": "Mean", "macro_f1": np.mean(fold_macro_f1s),
                "micro_f1": np.mean(fold_micro_f1s), "weighted_f1": np.mean(fold_weighted_f1s)}
    sd_row   = {"fold": "SD",   "macro_f1": np.std(fold_macro_f1s),
                "micro_f1": np.std(fold_micro_f1s),  "weighted_f1": np.std(fold_weighted_f1s)}
    for j, label in enumerate(label_columns):
        mean_row[label] = np.mean(fold_label_f1s[:, j])
        sd_row[label]   = np.std(fold_label_f1s[:, j])
    fold_rows.extend([mean_row, sd_row])
    cv_df = pd.DataFrame(fold_rows)
    cv_df.to_csv("cv_results.csv", index=False, sep=";")
    print("  CV results saved to 'cv_results.csv'\n")

    return fold_macro_f1s, fold_micro_f1s


# ══════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════

print("--- Step 1: Loading Filtered Datasets ---")
print("(All-zero rows removed; run 'data 25-06-26.py' first if files are missing)")

label_columns = [
    "Death Anxiety",
    "Death Acceptance",
    "Loneliness",
    "Solitude",
    "Identity Confusion",
    "Identity Synthesis",
    "Freedom Paralysis",
    "Freedom Responsibility",
    "Meaninglessness",
    "Engagement"
]

train_df = pd.read_csv("training_filtered.csv",   sep=";")
val_df   = pd.read_csv("validation_filtered.csv", sep=";")
test_df  = pd.read_csv("test_filtered.csv",       sep=";")

train_dataset = Dataset.from_dict({
    "text":  train_df["Text"].tolist(),
    "label": train_df[label_columns].values.tolist()
})

val_dataset = Dataset.from_dict({
    "text":  val_df["Text"].tolist(),
    "label": val_df[label_columns].values.tolist()
})

test_dataset = Dataset.from_dict({
    "text":  test_df["Text"].tolist(),
    "label": test_df[label_columns].values.tolist()
})

print(f"Loaded {len(train_dataset)} training, {len(val_dataset)} validation, "
      f"and {len(test_dataset)} test examples.")

print("\n--- Step 2: Loading SetFit Model (all-MiniLM-L6-v2) ---")
print("Using class_weight='balanced' on the logistic classifier head.")
model = make_model()

print("\n--- Step 3: Setting Up Trainer ---")
args = TrainingArguments(
    batch_size=8,
    num_epochs=5,
    num_iterations=2,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

print("\n--- Step 4: Starting Full SetFit Training ---")
print("Fine-tuning the sentence embedding space (Stage 1) and training the "
      "classifier head with class_weight='balanced' (Stage 2)...")
trainer.train()

print("\n--- Step 5: Evaluating Model on Test Set ---")
eval_results = trainer.evaluate(test_dataset)
print("Evaluation results on test dataset:", eval_results)

# Get raw predictions for detailed metrics
print("\nGenerating predictions for detailed metrics...")
test_texts  = test_df["Text"].tolist()
test_labels = test_df[label_columns].values.tolist()

preds = model.predict(test_texts)
if hasattr(preds, "tolist"):
    preds = preds.tolist()

# --- F-score per label ---
print_per_label_scores(test_labels, preds, label_columns,
                       title="Test Set: Per-Label F-Scores")

# --- Confusion matrix per label ---
print_per_label_confusion_matrices(test_labels, preds, label_columns)

# --- Save test results to CSV ---
test_rows = []
for i, label in enumerate(label_columns):
    col_true = np.array(test_labels)[:, i]
    col_pred = np.array(preds)[:, i]
    test_rows.append({
        "label":     label,
        "precision": precision_score(col_true, col_pred, zero_division=0),
        "recall":    recall_score(col_true, col_pred, zero_division=0),
        "f1":        f1_score(col_true, col_pred, zero_division=0),
        "support":   int(col_true.sum()),
    })
test_rows.append({"label": "Macro avg", "precision": "", "recall": "",
                  "f1": f1_score(test_labels, preds, average="macro", zero_division=0),
                  "support": ""})
test_rows.append({"label": "Micro avg", "precision": "", "recall": "",
                  "f1": f1_score(test_labels, preds, average="micro", zero_division=0),
                  "support": ""})
test_rows.append({"label": "Weighted avg", "precision": "", "recall": "",
                  "f1": f1_score(test_labels, preds, average="weighted", zero_division=0),
                  "support": ""})
pd.DataFrame(test_rows).to_csv("test_results.csv", index=False, sep=";")
print("  Test results saved to 'test_results.csv'")

print("\n--- Step 6: Stratified 5-Fold Cross-Validation ---")
# Combine train + validation for cross-validation
full_df = pd.concat([train_df, val_df], ignore_index=True)
run_stratified_kfold_cv(full_df, label_columns, n_splits=5)

print("\n--- Step 7: Saving Model ---")
output_dir = "setfit-minilm-filtered-balanced"
model.save_pretrained(output_dir)
print(f"Model saved successfully in '{output_dir}' directory!")

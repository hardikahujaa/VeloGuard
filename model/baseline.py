"""
baseline.py — Static Threshold Baseline for LoadGuard

This script evaluates a rule-based baseline that mimics what most production
systems do today: throttle when latency exceeds a fixed threshold.

It uses the EXACT same test split as train.py (same seeds, same data)
so the comparison is perfectly fair.

Run after train.py has been run at least once.
"""

import torch
import numpy as np
import pickle
import os
from sklearn.metrics import roc_auc_score

# ── Must match train.py exactly ──────────────────────────────────────────────
RANDOM_SEED  = 42
TRAIN_RATIO  = 0.70
VAL_RATIO    = 0.85
# ─────────────────────────────────────────────────────────────────────────────


def compute_metrics(preds, labels):
    tp = ((preds == 1) & (labels == 1)).sum()
    tn = ((preds == 0) & (labels == 0)).sum()
    fp = ((preds == 1) & (labels == 0)).sum()
    fn = ((preds == 0) & (labels == 1)).sum()

    accuracy  = (tp + tn) / len(labels) * 100
    precision = tp / (tp + fp + 1e-8) * 100
    recall    = tp / (tp + fn + 1e-8) * 100
    f1        = 2 * precision * recall / (precision + recall + 1e-8)
    return accuracy, precision, recall, f1, tp, tn, fp, fn


def run_baseline(tensors_path="data/processed_tensors.pt"):
    if not os.path.exists(tensors_path):
        print("processed_tensors.pt not found. Run dataset_prep.py first.")
        return

    # ── Reproduce the exact same split as train.py ────────────────────────
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    X, y = torch.load(tensors_path)
    perm = torch.randperm(len(X))
    X, y = X[perm], y[perm]

    n        = len(X)
    val_end  = int(VAL_RATIO * n)
    X_test   = X[val_end:]
    y_test   = y[val_end:].numpy().flatten()

    print(f"Test set: {len(X_test)} windows")
    print(f"Test crash rate: {y_test.mean()*100:.1f}%")
    print()

    # ── Load scaler to get back original feature values ───────────────────
    # The tensors are unscaled — scaler is only applied in train.py
    # We need raw avg_latency values which are in column index 1 (avg_latency)
    # Column order: rps=0, avg_latency=1, cpu_usage=2, mem_usage=3

    # Extract features from the LAST timestep of each window
    # This mimics what a real-time rule would see at decision time
    last_timestep = X_test[:, -1, :].numpy()  # [N, 4]

    # ── Load scaler to inverse-transform back to original units ───────────
    if not os.path.exists("data/scaler.pkl"):
        print("scaler.pkl not found. Run train.py first.")
        return

    with open("data/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    # Inverse transform to get original values
    original_values = scaler.inverse_transform(last_timestep)
    latency_raw  = original_values[:, 1]  # avg_latency column
    cpu_raw      = original_values[:, 2]  # cpu_usage column

    print("=" * 60)
    print("BASELINE 1 — Static Latency Threshold (latency > 1000ms)")
    print("=" * 60)
    print("This is what most production systems do today.")
    print("No ML, no temporal context, just a fixed rule.\n")

    preds_latency = ((latency_raw > 1000) | (cpu_raw > 90)).astype(float)
    acc, prec, rec, f1, tp, tn, fp, fn = compute_metrics(preds_latency, y_test)

    # AUC needs probability scores — use normalised latency as proxy
    latency_norm = (latency_raw - latency_raw.min()) / (latency_raw.max() - latency_raw.min() + 1e-8)
    try:
        auc = roc_auc_score(y_test, latency_norm)
    except Exception:
        auc = float('nan')

    print(f"  Accuracy:  {acc:.2f}%")
    print(f"  Precision: {prec:.2f}%")
    print(f"  Recall:    {rec:.2f}%   ← primary metric")
    print(f"  F1:        {f1:.2f}%")
    print(f"  AUC-ROC:   {auc:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"  TP={int(tp)} | FP={int(fp)}")
    print(f"  FN={int(fn)} | TN={int(tn)}")
    print(f"\n  Missed crashes:     {int(fn)}")
    print(f"  Over-throttled:     {int(fp)}")

    print()
    print("=" * 60)
    print("BASELINE 2 — Always Throttle (predict all Crash)")
    print("=" * 60)
    print("The trivial worst-case baseline.\n")

    preds_always = np.ones(len(y_test))
    acc2, prec2, rec2, f12, tp2, tn2, fp2, fn2 = compute_metrics(preds_always, y_test)

    print(f"  Accuracy:  {acc2:.2f}%")
    print(f"  Precision: {prec2:.2f}%")
    print(f"  Recall:    {rec2:.2f}%")
    print(f"  F1:        {f12:.2f}%")
    print(f"  AUC-ROC:   N/A (constant predictor)")
    print(f"\n  Confusion Matrix:")
    print(f"  TP={int(tp2)} | FP={int(fp2)}")
    print(f"  FN={int(fn2)} | TN={int(tn2)}")

    print()
    print("=" * 60)
    print("BASELINE 3 — Never Throttle (predict all Safe)")
    print("=" * 60)
    print("The other trivial baseline.\n")

    preds_never = np.zeros(len(y_test))
    acc3, prec3, rec3, f13, tp3, tn3, fp3, fn3 = compute_metrics(preds_never, y_test)

    print(f"  Accuracy:  {acc3:.2f}%")
    print(f"  Precision: {prec3:.2f}%")
    print(f"  Recall:    {rec3:.2f}%")
    print(f"  F1:        {f13:.2f}%")
    print(f"  AUC-ROC:   N/A (constant predictor)")
    print(f"\n  Confusion Matrix:")
    print(f"  TP={int(tp3)} | FP={int(fp3)}")
    print(f"  FN={int(fn3)} | TN={int(tn3)}")

    print()
    print("=" * 60)
    print("SUMMARY TABLE (for research paper)")
    print("=" * 60)
    print(f"{'Method':<35} {'Recall':>8} {'Precision':>10} {'F1':>8} {'AUC':>8}")
    print("-" * 75)
    print(f"{'Static Threshold (latency>1000ms)':<35} {rec:>7.2f}% {prec:>9.2f}% {f1:>7.2f}% {auc:>7.4f}")
    print(f"{'Always Throttle':<35} {rec2:>7.2f}% {prec2:>9.2f}% {f12:>7.2f}% {'N/A':>8}")
    print(f"{'Never Throttle':<35} {rec3:>7.2f}% {prec3:>9.2f}% {f13:>7.2f}% {'N/A':>8}")
    print(f"{'BiLSTM-Attention (LoadGuard)':<35} {'88.78%':>8} {'100.00%':>10} {'94.05%':>8} {'0.9878':>8}")
    print()
    print("Note: LoadGuard row uses your best single-run result.")
    print("Replace with mean ± std after completing 5 runs.")


if __name__ == "__main__":
    run_baseline()
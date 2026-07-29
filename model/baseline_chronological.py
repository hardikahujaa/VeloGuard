# model/baseline_chronological.py
"""
Evaluates Never Throttle, Always Throttle, and Static Latency Threshold
on the SAME chronological (temporally blocked) test split used by
train_chronological.py, so the comparison against VeloGuard's
chronological result remains apples-to-apples.

NOTE: deliberately does NOT use scaler.inverse_transform() -- the data
loaded from processed_tensors.pt is already raw/unscaled (per
dataset_prep.py's own docstring), so no inverse-transform is needed.
Applying one (as baseline.py currently does) produces nonsensical
latency values in the millions of milliseconds -- confirmed via
verify_static_baseline.py.
"""
import torch
import numpy as np
from sklearn.metrics import roc_auc_score

from train_chronological import chronological_split
from baseline import compute_metrics


def run_baseline_chronological(tensors_path="data/processed_tensors.pt"):
    torch.manual_seed(42)
    np.random.seed(42)

    X, y = torch.load(tensors_path)
    _, _, _, _, X_test, y_test = chronological_split(X, y)
    y_test = y_test.numpy().flatten()

    print(f"Chronological test set: {len(X_test)} windows")
    print(f"Test crash rate: {y_test.mean()*100:.1f}%\n")

    last_timestep = X_test[:, -1, :].numpy()

    print("=" * 60)
    print("Static Latency Threshold (latency > 1000ms or CPU > 90%)")
    print("=" * 60)
    latency_raw = last_timestep[:, 1]
    cpu_raw     = last_timestep[:, 2]
    preds_latency = ((latency_raw > 1000) | (cpu_raw > 90)).astype(float)
    acc, prec, rec, f1, tp, tn, fp, fn = compute_metrics(preds_latency, y_test)
    latency_norm = (latency_raw - latency_raw.min()) / (latency_raw.max() - latency_raw.min() + 1e-8)
    try:
        auc = roc_auc_score(y_test, latency_norm)
    except Exception:
        auc = float('nan')
    print(f"  Precision: {prec:.2f}% | Recall: {rec:.2f}% | F1: {f1:.2f}% | AUC: {auc:.4f}")
    print(f"  TP={int(tp)} FP={int(fp)} FN={int(fn)} TN={int(tn)}\n")

    print("=" * 60)
    print("Always Throttle")
    print("=" * 60)
    preds_always = np.ones(len(y_test))
    acc2, prec2, rec2, f12, tp2, tn2, fp2, fn2 = compute_metrics(preds_always, y_test)
    print(f"  Precision: {prec2:.2f}% | Recall: {rec2:.2f}% | F1: {f12:.2f}%")
    print(f"  TP={int(tp2)} FP={int(fp2)} FN={int(fn2)} TN={int(tn2)}\n")

    print("=" * 60)
    print("Never Throttle")
    print("=" * 60)
    preds_never = np.zeros(len(y_test))
    acc3, prec3, rec3, f13, tp3, tn3, fp3, fn3 = compute_metrics(preds_never, y_test)
    print(f"  Precision: {prec3:.2f}% | Recall: {rec3:.2f}% | F1: {f13:.2f}%")
    print(f"  TP={int(tp3)} FP={int(fp3)} FN={int(fn3)} TN={int(tn3)}\n")

    print("=" * 60)
    print("SUMMARY (chronological split, corrected static baseline)")
    print("=" * 60)
    print(f"{'Method':<20} {'Recall':>8} {'Prec':>8} {'F1':>8}")
    print(f"{'Never Throttle':<20} {rec3:>7.2f}% {prec3:>7.2f}% {f13:>7.2f}%")
    print(f"{'Always Throttle':<20} {rec2:>7.2f}% {prec2:>7.2f}% {f12:>7.2f}%")
    print(f"{'Static Threshold':<20} {rec:>7.2f}% {prec:>7.2f}% {f1:>7.2f}%")
    print(f"{'VeloGuard':<20} {'83.02%':>8} {'65.67%':>8} {'73.33%':>8}   (from train_chronological.py)")


if __name__ == "__main__":
    run_baseline_chronological()

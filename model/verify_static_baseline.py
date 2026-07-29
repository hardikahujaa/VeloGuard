# model/verify_static_baseline.py
import torch
import numpy as np
import pickle
from baseline import compute_metrics, RANDOM_SEED, VAL_RATIO

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

X, y = torch.load("data/processed_tensors.pt")
perm = torch.randperm(len(X))
X, y = X[perm], y[perm]
n = len(X)
val_end = int(VAL_RATIO * n)
X_test = X[val_end:]
y_test = y[val_end:].numpy().flatten()

last_timestep = X_test[:, -1, :].numpy()

with open("data/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Version A: baseline.py's current logic (inverse_transform on already-raw data)
inv = scaler.inverse_transform(last_timestep)
latency_A, cpu_A = inv[:, 1], inv[:, 2]
preds_A = ((latency_A > 1000) | (cpu_A > 90)).astype(float)
accA, precA, recA, f1A, tpA, tnA, fpA, fnA = compute_metrics(preds_A, y_test)

# Version B: use the values as-is (already raw, per dataset_prep.py's own docstring)
latency_B, cpu_B = last_timestep[:, 1], last_timestep[:, 2]
preds_B = ((latency_B > 1000) | (cpu_B > 90)).astype(float)
accB, precB, recB, f1B, tpB, tnB, fpB, fnB = compute_metrics(preds_B, y_test)

print("Version A - current baseline.py logic (inverse_transform applied):")
print(f"  Recall={recA:.2f}% Precision={precA:.2f}% TP={int(tpA)} FP={int(fpA)} FN={int(fnA)} TN={int(tnA)}")
print(f"  Sample latency_A values: {latency_A[:5]}")

print("\nVersion B - raw values used directly (no inverse_transform):")
print(f"  Recall={recB:.2f}% Precision={precB:.2f}% TP={int(tpB)} FP={int(fpB)} FN={int(fnB)} TN={int(tnB)}")
print(f"  Sample latency_B values: {latency_B[:5]}")

print("\nPublished paper Table II Static Threshold: Recall=100.00% Precision=64.90% FP=53 FN=0")

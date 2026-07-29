"""
Chronological (temporally blocked) train/val/test split, addressing
Reviewer 2's concern that a random split over overlapping 60-second
sliding windows can leak near-identical windows across splits.

Everything else (scaling, pos_weight, model architecture, optimizer,
schedule, threshold calibration) is IDENTICAL to train.py — only the
split method changes, so results are directly comparable.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pickle

from lstm_model import CrashPredictorLSTM
from train import compute_metrics, scale_splits, find_optimal_threshold, DEVICE, BATCH_SIZE, EPOCHS, LR, PATIENCE

LOOKBACK = 60  # must match dataset_prep.py's lookback — also the guard-gap size

def chronological_split(X, y, lookback=LOOKBACK, train_frac=0.70, val_frac=0.15):
    """
    Splits X, y in their ORIGINAL (chronological) order, with a gap of
    `lookback` windows removed at each boundary so no window in one
    split shares even a single timestep with a window in another.
    """
    n = len(X)
    gap = lookback
    usable = n - 2 * gap
    train_n = int(train_frac * usable)
    val_n   = int(val_frac * usable)
    test_n  = usable - train_n - val_n

    train_end  = train_n
    val_start  = train_end + gap
    val_end    = val_start + val_n
    test_start = val_end + gap
    test_end   = test_start + test_n

    X_train, y_train = X[:train_end],         y[:train_end]
    X_val,   y_val   = X[val_start:val_end],   y[val_start:val_end]
    X_test,  y_test  = X[test_start:test_end], y[test_start:test_end]

    print(f"Chronological split (gap={gap} windows at each boundary):")
    print(f"  Train: {len(X_train)} windows | Val: {len(X_val)} windows | Test: {len(X_test)} windows")
    print(f"  Dropped to gaps: {n - len(X_train) - len(X_val) - len(X_test)} windows")
    return X_train, y_train, X_val, y_val, X_test, y_test


def train_chronological(tensors_path="data/processed_tensors.pt",
                         model_path="data/model_chronological.pth",
                         threshold_path="data/threshold_chronological.txt"):
    torch.manual_seed(42)
    np.random.seed(42)

    print("Loading dataset...")
    X, y = torch.load(tensors_path)
    print(f"Loaded: X={X.shape} | y={y.shape}")

    X_train, y_train, X_val, y_val, X_test, y_test = chronological_split(X, y)
    print(f"  Train crash rate: {y_train.mean().item()*100:.1f}%")
    print(f"  Val crash rate:   {y_val.mean().item()*100:.1f}%")
    print(f"  Test crash rate:  {y_test.mean().item()*100:.1f}%")

    X_train, X_val, X_test, scaler = scale_splits(X_train, X_val, X_test)
    with open("data/scaler_chronological.pkl", "wb") as f:
        pickle.dump(scaler, f)

    n_neg = (y_train == 0).sum().float()
    n_pos = (y_train == 1).sum().float()
    pos_weight = (n_neg / n_pos).to(DEVICE)
    print(f"pos_weight: {pos_weight.item():.3f}")

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader   = DataLoader(TensorDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(TensorDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False)

    model = CrashPredictorLSTM(input_size=4).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    warmup_epochs = 5
    def lr_lambda(epoch):
        return (epoch + 1) / warmup_epochs if epoch < warmup_epochs else 1.0
    warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS - warmup_epochs, eta_min=1e-5)

    best_val_f2 = 0.0
    patience_counter = 0
    print(f"\n{'Epoch':>5} | {'Loss':>6} | {'Acc':>6} | {'Prec':>6} | {'Recall':>6} | {'F1':>6} | Note")
    print("-" * 75)
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        for bX, by in train_loader:
            bX, by = bX.to(DEVICE), by.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(bX), by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item()

        model.eval()
        val_probs, val_labels = [], []
        with torch.no_grad():
            for bX, by in val_loader:
                bX, by = bX.to(DEVICE), by.to(DEVICE)
                probs = torch.sigmoid(model(bX))
                val_probs.append(probs.cpu().numpy())
                val_labels.append(by.cpu().numpy())
        val_probs = np.concatenate(val_probs).flatten()
        val_labels = np.concatenate(val_labels).flatten()
        acc, prec, rec, f1, tp, tn, fp, fn = compute_metrics(val_probs, val_labels)
        f2 = (5 * prec * rec) / (4 * prec + rec + 1e-8)

        note = ""
        if f2 > best_val_f2:
            best_val_f2 = f2
            patience_counter = 0
            torch.save(model.state_dict(), model_path)
            note = f" <- best F2 ({f2:.2f}%)"
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\nEarly stopping at epoch {epoch}. Best val F2: {best_val_f2:.2f}%")
                break

        print(f"{epoch:5d} | {running_loss/len(train_loader):6.4f} | {acc:5.2f}% | {prec:5.2f}% | {rec:5.2f}% | {f1:5.2f}%{note}")
        if epoch <= warmup_epochs:
            warmup_scheduler.step()
        else:
            cosine_scheduler.step()

    print("\n--- Calibrating threshold on chronological validation set ---")
    model.load_state_dict(torch.load(model_path))
    best_threshold, auc, best_prec, best_rec = find_optimal_threshold(model, val_loader)
    print(f"AUC-ROC (val): {auc:.4f} | threshold: {best_threshold:.4f}")
    with open(threshold_path, "w") as f:
        f.write(str(best_threshold))

    print("\n--- Final evaluation on chronological held-out TEST set ---")
    model.eval()
    test_probs, test_labels = [], []
    with torch.no_grad():
        for bX, by in test_loader:
            bX, by = bX.to(DEVICE), by.to(DEVICE)
            probs = torch.sigmoid(model(bX))
            test_probs.append(probs.cpu().numpy())
            test_labels.append(by.cpu().numpy())
    test_probs = np.concatenate(test_probs).flatten()
    test_labels = np.concatenate(test_labels).flatten()
    acc, prec, rec, f1, tp, tn, fp, fn = compute_metrics(test_probs, test_labels, threshold=best_threshold)
    try:
        from sklearn.metrics import roc_auc_score
        test_auc = roc_auc_score(test_labels, test_probs)
    except Exception:
        test_auc = None

    print(f"\nChronological-split test results (threshold={best_threshold:.4f}):")
    print(f"  Accuracy:  {acc:.2f}%")
    print(f"  Precision: {prec:.2f}%")
    print(f"  Recall:    {rec:.2f}%")
    print(f"  F1:        {f1:.2f}%")
    if test_auc is not None:
        print(f"  AUC-ROC:   {test_auc:.4f}")
    print(f"  TP={int(tp)} FP={int(fp)} FN={int(fn)} TN={int(tn)}")


if __name__ == "__main__":
    train_chronological()

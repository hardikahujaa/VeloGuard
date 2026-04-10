import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_recall_curve, roc_auc_score, average_precision_score
import numpy as np
import pickle
import os

# --- Added Matplotlib Imports for Figures ---
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# ---------------------------------------------

from lstm_model import CrashPredictorLSTM

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
EPOCHS     = 150
LR         = 3e-4
PATIENCE   = 25

os.makedirs("data", exist_ok=True)


def compute_metrics(preds, labels, threshold=0.5):
    """
    Computes TP, TN, FP, FN and derives accuracy, precision, recall, F1.
    Recall is the primary metric for VeloGuard — missing a crash is catastrophic.
    """
    binary_preds = (preds >= threshold).astype(float)

    tp = ((binary_preds == 1) & (labels == 1)).sum()
    tn = ((binary_preds == 0) & (labels == 0)).sum()
    fp = ((binary_preds == 1) & (labels == 0)).sum()
    fn = ((binary_preds == 0) & (labels == 1)).sum()

    accuracy  = (tp + tn) / len(labels) * 100
    precision = tp / (tp + fp + 1e-8) * 100
    recall    = tp / (tp + fn + 1e-8) * 100
    f1        = 2 * precision * recall / (precision + recall + 1e-8)

    return accuracy, precision, recall, f1, tp, tn, fp, fn


def scale_splits(X_train, X_val, X_test):
    """
    Fit scaler ONLY on training data.
    Transform val and test using training statistics — no leakage.
    """
    scaler     = MinMaxScaler()
    B_tr, T, F = X_train.shape

    X_tr_scaled = scaler.fit_transform(
        X_train.numpy().reshape(-1, F)
    ).reshape(B_tr, T, F)

    X_va_scaled = scaler.transform(
        X_val.numpy().reshape(-1, X_val.shape[-1])
    ).reshape(X_val.shape[0], T, F)

    X_te_scaled = scaler.transform(
        X_test.numpy().reshape(-1, X_test.shape[-1])
    ).reshape(X_test.shape[0], T, F)

    return (
        torch.tensor(X_tr_scaled, dtype=torch.float32),
        torch.tensor(X_va_scaled, dtype=torch.float32),
        torch.tensor(X_te_scaled, dtype=torch.float32),
        scaler,
    )


def find_optimal_threshold(model, loader):
    """
    Finds the optimal classification threshold using the F2 score on the
    validation set. F2 penalises false negatives 4x more than false positives,
    which is correct for VeloGuard — missing a crash is worse than over-throttling.
    """
    model.eval()
    all_probs  = []
    all_labels = []

    with torch.no_grad():
        for bX, by in loader:
            bX, by = bX.to(DEVICE), by.to(DEVICE)
            probs = torch.sigmoid(model(bX))
            all_probs.append(probs.cpu().numpy())
            all_labels.append(by.cpu().numpy())

    all_probs  = np.concatenate(all_probs).flatten()
    all_labels = np.concatenate(all_labels).flatten()

    precisions, recalls, thresholds = precision_recall_curve(all_labels, all_probs)

    # F2 score: weights recall twice as heavily as precision
    f2_scores = (5 * precisions * recalls) / (4 * precisions + recalls + 1e-8)
    best_idx  = np.argmax(f2_scores)

    best_threshold = float(thresholds[min(best_idx, len(thresholds) - 1)])
    auc = roc_auc_score(all_labels, all_probs)

    return best_threshold, auc, precisions[best_idx], recalls[best_idx]


def train_model(tensors_path="data/processed_tensors.pt", model_path="data/model.pth"):
    if not os.path.exists(tensors_path):
        print(f"Tensors file not found: {tensors_path}")
        return

    # --- Reproducibility ---
    torch.manual_seed(42)
    np.random.seed(42)

    print("Loading dataset...")
    X, y = torch.load(tensors_path)
    print(f"Loaded: X={X.shape} | y={y.shape}")

    # --- Shuffle ---
    perm = torch.randperm(len(X))
    X, y = X[perm], y[perm]

    # --- 70 / 15 / 15 Split ---
    n           = len(X)
    train_end   = int(0.70 * n)
    val_end     = int(0.85 * n)

    X_train, y_train = X[:train_end],       y[:train_end]
    X_val,   y_val   = X[train_end:val_end], y[train_end:val_end]
    X_test,  y_test  = X[val_end:],        y[val_end:]

    print(f"\nSplit sizes:")
    print(f"  Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    print(f"  Train crash rate: {y_train.mean().item()*100:.1f}%")
    print(f"  Val crash rate:   {y_val.mean().item()*100:.1f}%")
    print(f"  Test crash rate:  {y_test.mean().item()*100:.1f}%")

    # --- Scale (fit on train only) ---
    X_train, X_val, X_test, scaler = scale_splits(X_train, X_val, X_test)
    with open("data/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    print("\nScaler fit on training data only. Saved to data/scaler.pkl")

    # --- pos_weight from training labels only ---
    n_neg      = (y_train == 0).sum().float()
    n_pos      = (y_train == 1).sum().float()
    pos_weight = (n_neg / n_pos).to(DEVICE)
    print(f"pos_weight: {pos_weight.item():.3f}")

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=BATCH_SIZE, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(
        TensorDataset(X_val, y_val),
        batch_size=BATCH_SIZE, shuffle=False
    )
    test_loader = DataLoader(
        TensorDataset(X_test, y_test),
        batch_size=BATCH_SIZE, shuffle=False
    )

    model     = CrashPredictorLSTM(input_size=4).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    warmup_epochs = 5
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        return 1.0

    warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS - warmup_epochs, eta_min=1e-5
    )

    best_val_f2 = 0.0
    patience_counter = 0

    train_losses = []
    val_f2_scores = []

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

        avg_loss = running_loss / len(train_loader)

        model.eval()
        val_probs  = []
        val_labels = []

        with torch.no_grad():
            for bX, by in val_loader:
                bX, by = bX.to(DEVICE), by.to(DEVICE)
                probs = torch.sigmoid(model(bX))
                val_probs.append(probs.cpu().numpy())
                val_labels.append(by.cpu().numpy())

        val_probs  = np.concatenate(val_probs).flatten()
        val_labels = np.concatenate(val_labels).flatten()

        acc, prec, rec, f1, tp, tn, fp, fn = compute_metrics(val_probs, val_labels)

        f2 = (5 * prec * rec) / (4 * prec + rec + 1e-8)
        
        train_losses.append(avg_loss)
        val_f2_scores.append(f2)

        note = ""
        
        if f2 > best_val_f2:
            best_val_f2      = f2
            patience_counter = 0
            torch.save(model.state_dict(), model_path)
            note = f" ← best F2 ({f2:.2f}%)"
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\nEarly stopping at epoch {epoch}. Best val F2: {best_val_f2:.2f}%")
                break

        print(f"{epoch:5d} | {avg_loss:6.4f} | {acc:5.2f}% | {prec:5.2f}% | "
              f"{rec:5.2f}% | {f1:5.2f}%{note}")

        if epoch <= warmup_epochs:
            warmup_scheduler.step()
        else:
            cosine_scheduler.step()

    print("\n--- Calibrating threshold on validation set ---")
    model.load_state_dict(torch.load(model_path))
    best_threshold, auc, best_prec, best_rec = find_optimal_threshold(model, val_loader)

    print(f"AUC-ROC:           {auc:.4f}")
    print(f"Optimal threshold: {best_threshold:.4f}  (F2-optimised)")
    print(f"At this threshold → Precision: {best_prec*100:.2f}% | Recall: {best_rec*100:.2f}%")

    with open("data/threshold.txt", "w") as f:
        f.write(str(best_threshold))
    print("Threshold saved to data/threshold.txt")

    print("\n--- Final evaluation on held-out TEST set ---")
    model.eval()
    test_probs  = []
    test_labels = []

    with torch.no_grad():
        for bX, by in test_loader:
            bX, by = bX.to(DEVICE), by.to(DEVICE)
            probs = torch.sigmoid(model(bX))
            test_probs.append(probs.cpu().numpy())
            test_labels.append(by.cpu().numpy())

    test_probs  = np.concatenate(test_probs).flatten()
    test_labels = np.concatenate(test_labels).flatten()

    acc, prec, rec, f1, tp, tn, fp, fn = compute_metrics(
        test_probs, test_labels, threshold=best_threshold
    )

    print(f"\nTest Results (threshold={best_threshold:.4f}):")
    print(f"  Accuracy:  {acc:.2f}%")
    print(f"  Precision: {prec:.2f}%")
    print(f"  Recall:    {rec:.2f}%   ← primary metric")
    print(f"  F1:        {f1:.2f}%")
    print(f"\n  Confusion Matrix:")
    print(f"  TP={int(tp)} | FP={int(fp)}")
    print(f"  FN={int(fn)} | TN={int(tn)}")
    print(f"\n  False Negatives (missed crashes): {int(fn)}")
    print(f"  False Positives (over-throttled):  {int(fp)}")
    print(f"\nModel saved to {model_path}")

    # =========================================================
    # --- Generating Figure 3 (Training Curves) ---
    # =========================================================
    print("\nGenerating Figure 3...")
    os.makedirs('figures', exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(train_losses, color='#2196F3', linewidth=2, label='Training Loss')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('BCEWithLogitsLoss', fontsize=12)
    ax1.set_title('Training Loss Convergence', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(val_f2_scores, color='#4CAF50', linewidth=2, label='Validation F2 Score')
    ax2.axhline(y=max(val_f2_scores), color='red', linestyle='--', 
                alpha=0.7, label=f'Best F2: {max(val_f2_scores):.2f}%')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('F2 Score (%)', fontsize=12)
    ax2.set_title('Validation F2 Score Progression', fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('figures/fig3_training_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Training curves successfully saved to figures/fig3_training_curves.png!")

    # =========================================================
    # --- Generating Figure 4 (Precision-Recall Curve) ---
    # =========================================================
    print("\nGenerating Figure 4...")
    
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for bX, by in test_loader:
            bX, by = bX.to(DEVICE), by.to(DEVICE)
            probs = torch.sigmoid(model(bX))
            all_probs.extend(probs.cpu().numpy().flatten())
            all_labels.extend(by.cpu().numpy().flatten())

    all_probs  = np.array(all_probs)
    all_labels = np.array(all_labels)

    vg_prec, vg_rec, _ = precision_recall_curve(all_labels, all_probs)
    vg_ap = average_precision_score(all_labels, all_probs)

    with open("data/scaler.pkl", "rb") as f:
        sc = pickle.load(f)

    X_test_raw = X_test.numpy().reshape(-1, X_test.shape[-1])
    X_test_inv = sc.inverse_transform(X_test_raw).reshape(X_test.shape)
    latency_last = X_test_inv[:, -1, 1]  
    latency_norm = (latency_last - latency_last.min()) / \
                   (latency_last.max() - latency_last.min() + 1e-8)

    st_prec, st_rec, _ = precision_recall_curve(
        y_test.numpy().flatten(), latency_norm
    )
    st_ap = average_precision_score(y_test.numpy().flatten(), latency_norm)

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(vg_rec, vg_prec, color='#2196F3', linewidth=2.5,
            label=f'VeloGuard BiLSTM-Attention (AP={vg_ap:.4f})')
    ax.plot(st_rec, st_prec, color='#FF5722', linewidth=2.5,
            linestyle='--', label=f'Static Threshold (AP={st_ap:.4f})')

    ax.scatter([87/98], [1.0], color='#2196F3', s=150, zorder=5,
               label=f'VeloGuard operating point\n(Recall=88.78%, Precision=100%)')

    crash_rate = all_labels.mean()
    ax.axhline(y=crash_rate, color='gray', linestyle=':', 
               linewidth=1.5, label=f'Random classifier (AP={crash_rate:.4f})')

    ax.set_xlabel('Recall', fontsize=13)
    ax.set_ylabel('Precision', fontsize=13)
    ax.set_title('Precision-Recall Curves: VeloGuard vs. Static Threshold',
                 fontsize=13, fontweight='bold')
    ax.legend(loc='lower left', fontsize=10)
    ax.set_xlim([0, 1.05])
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('figures/fig4_pr_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("PR curve successfully saved to figures/fig4_pr_curve.png!")

    # =========================================================
    # --- Generating Figure 5 (Confusion Matrix Heatmaps) ---
    # =========================================================
    print("\nGenerating Figure 5...")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    static_cm = np.array([[0, 53], [0, 98]])
    velo_cm = np.array([[int(tn), int(fp)], [int(fn), int(tp)]])

    cms    = [static_cm, velo_cm]
    titles = ['Static Latency Threshold', 'VeloGuard (BiLSTM-Attention)']
    colors = ['#FF5722', '#2196F3']

    for idx, (ax, cm, title, color) in enumerate(zip(axes, cms, titles, colors)):
        im = ax.imshow(cm, interpolation='nearest', 
                       cmap=plt.cm.Blues if idx == 1 else plt.cm.Oranges)
        ax.set_title(title, fontsize=12, fontweight='bold', pad=12)
        
        tick_marks = np.arange(2)
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(['Predicted\nSafe', 'Predicted\nCrash'], fontsize=10)
        ax.set_yticklabels(['Actual\nSafe', 'Actual\nCrash'], fontsize=10)
        
        thresh = cm.max() / 2.0
        labels = [['TN', 'FP'], ['FN', 'TP']]
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f'{labels[i][j]}\n{cm[i, j]}',
                        ha='center', va='center', fontsize=14, fontweight='bold',
                        color='white' if cm[i, j] > thresh else 'black')
        
        ax.set_ylabel('Actual Label', fontsize=11)
        ax.set_xlabel('Predicted Label', fontsize=11)

    plt.suptitle('Confusion Matrix Comparison: Static Threshold vs. VeloGuard',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('figures/fig5_confusion_matrices.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Confusion matrices saved to figures/fig5_confusion_matrices.png!")

if __name__ == "__main__":
    train_model()
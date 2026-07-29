import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from sklearn.preprocessing import MinMaxScaler
from lstm_model import CrashPredictorLSTM

DEVICE = "cpu"

def main():
    torch.manual_seed(42); np.random.seed(42)
    X, y = torch.load("data/processed_tensors.pt")
    perm = torch.randperm(len(X))
    X, y = X[perm], y[perm]
    n = len(X)
    tr_end, val_end = int(0.70*n), int(0.85*n)
    X_train, X_test = X[:tr_end], X[val_end:]
    y_test = y[val_end:].numpy().flatten()

    scaler = MinMaxScaler()
    scaler.fit(X_train.reshape(-1, X_train.shape[-1]).numpy())
    shp = X_test.shape
    X_test_s = torch.tensor(scaler.transform(X_test.reshape(-1, shp[-1]).numpy()).reshape(shp), dtype=torch.float32)

    model = CrashPredictorLSTM().to(DEVICE)
    model.load_state_dict(torch.load("data/model.pth", map_location=DEVICE))
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(X_test_s.to(DEVICE))).cpu().numpy().flatten()

    p, r, _ = precision_recall_curve(y_test, probs)

    plt.figure(figsize=(5, 4))
    plt.plot(r, p, label="VeloGuard (BiLSTM+Attn.)", linewidth=2)
    plt.scatter([0.8878], [1.0000], marker="*", s=140, color="black", zorder=5, label="VeloGuard operating point")
    plt.scatter([0.8776], [1.0000], marker="s", s=70, color="red", zorder=5, label="Static Threshold")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title("Precision-Recall Curve (Random Split, N=151)")
    plt.legend(loc="lower left", fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("fig4_pr.png", dpi=300)
    print("Saved fig4_pr.png")

if __name__ == "__main__":
    main()

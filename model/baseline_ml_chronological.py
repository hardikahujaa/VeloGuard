import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from train_chronological import chronological_split

def compute_metrics(preds, labels):
    tp = ((preds == 1) & (labels == 1)).sum()
    tn = ((preds == 0) & (labels == 0)).sum()
    fp = ((preds == 1) & (labels == 0)).sum()
    fn = ((preds == 0) & (labels == 1)).sum()
    precision = tp / (tp + fp + 1e-8) * 100
    recall    = tp / (tp + fn + 1e-8) * 100
    f1        = 2 * precision * recall / (precision + recall + 1e-8)
    return precision, recall, f1, tp, tn, fp, fn

def summarize(X):
    X = X.numpy()
    return np.concatenate([X.mean(1), X.std(1), X.min(1), X.max(1), X[:, -1, :]], axis=1)

def main():
    torch.manual_seed(42); np.random.seed(42)
    X, y = torch.load("data/processed_tensors.pt")
    X_train, y_train, X_val, y_val, X_test, y_test = chronological_split(X, y)

    Xs_train, Xs_test = summarize(X_train), summarize(X_test)
    y_train_np, y_test_np = y_train.numpy().flatten(), y_test.numpy().flatten()

    models = {
        "Logistic Regression": LogisticRegression(max_iter=5000, class_weight="balanced"),
        "Random Forest":       RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42),
        "XGBoost":             XGBClassifier(n_estimators=300, eval_metric="logloss",
                                              scale_pos_weight=(y_train_np==0).sum()/(y_train_np==1).sum(), random_state=42),
    }
    print(f"{'Model':<22}{'Recall':>9}{'Precision':>11}{'F1':>8}")
    for name, clf in models.items():
        clf.fit(Xs_train, y_train_np)
        preds = clf.predict(Xs_test)
        prec, rec, f1, tp, tn, fp, fn = compute_metrics(preds, y_test_np)
        print(f"{name:<22}{rec:>8.2f}%{prec:>10.2f}%{f1:>7.2f}%   (TP={int(tp)} FP={int(fp)} FN={int(fn)} TN={int(tn)})")

if __name__ == "__main__":
    main()

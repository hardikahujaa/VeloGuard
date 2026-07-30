import torch, torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_recall_curve

DEVICE = "cpu"
BATCH_SIZE, EPOCHS, LR, PATIENCE = 32, 100, 1e-3, 25
COLUMN_NAMES = ['timestamp', 'req_count', 'latency_ms', 'cpu_usage', 'mem_usage']

def load_aggregated(log_path="data/api_traffic.log"):
    df = pd.read_csv(log_path, names=COLUMN_NAMES)
    df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
    df = df.dropna(subset=['timestamp', 'cpu_usage', 'mem_usage'])
    df['timestamp_sec'] = df['timestamp'].astype(int)
    df = df.sort_values('timestamp_sec')
    rows = []
    for ts, g in df.groupby('timestamp_sec'):
        rows.append({
            'rps': float(len(g)),
            'avg_latency': pd.to_numeric(g['latency_ms'], errors='coerce').mean(),
            'cpu_usage':   pd.to_numeric(g['cpu_usage'],  errors='coerce').mean(),
            'mem_usage':   pd.to_numeric(g['mem_usage'],  errors='coerce').mean(),
        })
    return pd.DataFrame(rows).dropna().reset_index(drop=True)

def build_windows(df_agg, L, k):
    feats = df_agg[['rps', 'avg_latency', 'cpu_usage', 'mem_usage']].values.astype(np.float32)
    crash = ((df_agg['avg_latency'] > 1000) | (df_agg['cpu_usage'] > 90)).values.astype(int)
    X, y = [], []
    n = len(df_agg)
    for i in range(n - L - k + 1):
        X.append(feats[i:i+L])
        y.append(int(crash[i+L:i+L+k].max()))
    return torch.tensor(np.array(X)), torch.tensor(np.array(y), dtype=torch.float32).reshape(-1, 1)

def chronological_split(X, y, L, train_frac=0.70, val_frac=0.15):
    n = len(X)
    tr_end, val_end = int(train_frac*n), int((train_frac+val_frac)*n)
    gap = L
    return (X[:max(tr_end-gap,1)], y[:max(tr_end-gap,1)],
            X[tr_end+gap:val_end-gap], y[tr_end+gap:val_end-gap],
            X[val_end+gap:], y[val_end+gap:])

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.LSTM(4, 128, num_layers=2, batch_first=True, bidirectional=True, dropout=0.2)
        self.attn = nn.Sequential(nn.Linear(256,64), nn.Tanh(), nn.Linear(64,1))
        self.norm = nn.LayerNorm(256); self.drop = nn.Dropout(0.2); self.out = nn.Linear(256,1)
    def forward(self, x):
        h,_ = self.rnn(x)
        w = torch.softmax(self.attn(h), dim=1)
        ctx = (w*h).sum(1)
        return self.out(self.drop(self.norm(ctx)))

def compute_metrics(preds, labels):
    tp=((preds==1)&(labels==1)).sum(); tn=((preds==0)&(labels==0)).sum()
    fp=((preds==1)&(labels==0)).sum(); fn=((preds==0)&(labels==1)).sum()
    prec=tp/(tp+fp+1e-8)*100; rec=tp/(tp+fn+1e-8)*100
    f1=2*prec*rec/(prec+rec+1e-8)
    return prec,rec,f1

def find_threshold(model, X, y):
    with torch.no_grad():
        probs = torch.sigmoid(model(X)).numpy().flatten()
    p, r, t = precision_recall_curve(y, probs)
    f2 = 5*p*r/(4*p+r+1e-8)
    return t[np.argmax(f2[:-1])] if len(t) > 0 else 0.5

def train_and_eval(L, k, label):
    torch.manual_seed(42); np.random.seed(42)
    df_agg = load_aggregated()
    X, y = build_windows(df_agg, L, k)
    X_train, y_train, X_val, y_val, X_test, y_test = chronological_split(X, y, L)
    if len(X_train) < 20 or len(X_val) < 5 or len(X_test) < 5:
        print(f"{label:<14} SKIPPED (not enough windows: train={len(X_train)} val={len(X_val)} test={len(X_test)})")
        return
    scaler = MinMaxScaler()
    scaler.fit(X_train.reshape(-1,4).numpy())
    def scale(Xp):
        shp = Xp.shape
        return torch.tensor(scaler.transform(Xp.reshape(-1,4).numpy()).reshape(shp), dtype=torch.float32)
    X_train, X_val, X_test = scale(X_train), scale(X_val), scale(X_test)

    model = Model()
    pos_weight = torch.tensor([(y_train==0).sum()/(y_train==1).sum().clamp(min=1)])
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    best_f2, best_state, patience_ct = -1, None, 0
    for epoch in range(EPOCHS):
        model.train()
        perm = torch.randperm(len(X_train))
        for i in range(0, len(X_train), BATCH_SIZE):
            idx = perm[i:i+BATCH_SIZE]
            opt.zero_grad()
            loss = crit(model(X_train[idx]), y_train[idx])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            val_probs = torch.sigmoid(model(X_val)).numpy().flatten()
        preds = (val_probs > 0.5).astype(int)
        _,rec_v,_ = compute_metrics(preds, y_val.numpy().flatten())
        prec_v = ((preds==1)&(y_val.numpy().flatten()==1)).sum() / (max((preds==1).sum(),1))
        f2 = 5*prec_v*rec_v/(4*prec_v+rec_v+1e-8)
        if f2 > best_f2:
            best_f2, best_state, patience_ct = f2, {kk: vv.clone() for kk, vv in model.state_dict().items()}, 0
        else:
            patience_ct += 1
            if patience_ct >= PATIENCE: break
    model.load_state_dict(best_state)
    model.eval()
    thresh = find_threshold(model, X_val, y_val.numpy().flatten())
    with torch.no_grad():
        test_probs = torch.sigmoid(model(X_test)).numpy().flatten()
    preds = (test_probs > thresh).astype(int)
    prec, rec, f1 = compute_metrics(preds, y_test.numpy().flatten())
    print(f"{label:<14} recall={rec:6.2f}%  precision={prec:6.2f}%  F1={f1:6.2f}%  (train={len(X_train)} val={len(X_val)} test={len(X_test)})")

def main():
    print("k/L sensitivity sweep (chronological split)\n")
    train_and_eval(L=60, k=2, label="L=60, k=2")
    train_and_eval(L=60, k=4, label="L=60, k=4")
    train_and_eval(L=45, k=3, label="L=45, k=3")
    train_and_eval(L=90, k=3, label="L=90, k=3")
    print("\n(reference) L=60, k=3 (current model): recall=83.02%  precision=65.67%  F1=73.33%")

if __name__ == "__main__":
    main()

import torch, torch.nn as nn
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_recall_curve
from train_chronological import chronological_split

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE, EPOCHS, LR, PATIENCE = 32, 100, 1e-3, 25

class Attn(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.attn = nn.Sequential(nn.Linear(hidden,64), nn.Tanh(), nn.Linear(64,1))
        self.norm = nn.LayerNorm(hidden)
        self.drop = nn.Dropout(0.2)
        self.out  = nn.Linear(hidden,1)
    def head(self, h):
        w = torch.softmax(self.attn(h), dim=1)
        ctx = (w*h).sum(1)
        return self.out(self.drop(self.norm(ctx)))

class GRUModel(Attn):
    def __init__(self):
        super().__init__(256)
        self.rnn = nn.GRU(4,128,num_layers=2,batch_first=True,bidirectional=True,dropout=0.2)
    def forward(self,x):
        h,_ = self.rnn(x); return self.head(h)

class UniLSTMModel(Attn):
    def __init__(self):
        super().__init__(128)
        self.rnn = nn.LSTM(4,128,num_layers=2,batch_first=True,bidirectional=False,dropout=0.2)
    def forward(self,x):
        h,_ = self.rnn(x); return self.head(h)

def compute_metrics(preds, labels):
    tp=((preds==1)&(labels==1)).sum(); tn=((preds==0)&(labels==0)).sum()
    fp=((preds==1)&(labels==0)).sum(); fn=((preds==0)&(labels==1)).sum()
    prec=tp/(tp+fp+1e-8)*100; rec=tp/(tp+fn+1e-8)*100
    f1=2*prec*rec/(prec+rec+1e-8)
    return prec,rec,f1,tp,tn,fp,fn

def find_threshold(model, X, y):
    with torch.no_grad():
        probs = torch.sigmoid(model(X.to(DEVICE))).cpu().numpy().flatten()
    p, r, t = precision_recall_curve(y, probs)
    f2 = 5*p*r/(4*p+r+1e-8)
    return t[np.argmax(f2[:-1])] if len(t)>0 else 0.5

def run(name, ModelClass, X_train,y_train,X_val,y_val,X_test,y_test):
    torch.manual_seed(42)
    model = ModelClass().to(DEVICE)
    pos_weight = torch.tensor([(y_train==0).sum()/(y_train==1).sum()]).to(DEVICE)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    best_f2, best_state, patience_ct = -1, None, 0
    Xtr, ytr = X_train.to(DEVICE), y_train.to(DEVICE)
    for epoch in range(EPOCHS):
        model.train()
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), BATCH_SIZE):
            idx = perm[i:i+BATCH_SIZE]
            opt.zero_grad()
            out = model(Xtr[idx])
            loss = crit(out, ytr[idx])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            val_probs = torch.sigmoid(model(X_val.to(DEVICE))).cpu().numpy().flatten()
        preds = (val_probs > 0.5).astype(int)
        _,_,f1,tp,tn,fp,fn = compute_metrics(preds, y_val.numpy().flatten())
        rec = tp/(tp+fn+1e-8); prec = tp/(tp+fp+1e-8)
        f2 = 5*prec*rec/(4*prec+rec+1e-8)
        if f2 > best_f2:
            best_f2, best_state, patience_ct = f2, {k:v.clone() for k,v in model.state_dict().items()}, 0
        else:
            patience_ct += 1
            if patience_ct >= PATIENCE:
                break
    model.load_state_dict(best_state)
    model.eval()
    thresh = find_threshold(model, X_val, y_val.numpy().flatten())
    with torch.no_grad():
        test_probs = torch.sigmoid(model(X_test.to(DEVICE))).cpu().numpy().flatten()
    preds = (test_probs > thresh).astype(int)
    prec, rec, f1, tp, tn, fp, fn = compute_metrics(preds, y_test.numpy().flatten())
    print(f"{name:<22}{rec:>8.2f}%{prec:>10.2f}%{f1:>7.2f}%   (TP={int(tp)} FP={int(fp)} FN={int(fn)} TN={int(tn)})  thresh={thresh:.3f}")

def main():
    torch.manual_seed(42); np.random.seed(42)
    X, y = torch.load("data/processed_tensors.pt")
    X_train, y_train, X_val, y_val, X_test, y_test = chronological_split(X, y)

    scaler = MinMaxScaler()
    flat_train = X_train.reshape(-1, X_train.shape[-1]).numpy()
    scaler.fit(flat_train)
    def scale(Xp):
        shp = Xp.shape
        return torch.tensor(scaler.transform(Xp.reshape(-1,shp[-1]).numpy()).reshape(shp), dtype=torch.float32)
    X_train_s, X_val_s, X_test_s = scale(X_train), scale(X_val), scale(X_test)

    print(f"{'Model':<22}{'Recall':>9}{'Precision':>11}{'F1':>8}")
    run("GRU (bidirectional)", GRUModel, X_train_s,y_train,X_val_s,y_val,X_test_s,y_test)
    run("Uni-directional LSTM", UniLSTMModel, X_train_s,y_train,X_val_s,y_val,X_test_s,y_test)

if __name__ == "__main__":
    main()

#lstm_model.py
import torch
import torch.nn as nn

class CrashPredictorLSTM(nn.Module):
    def __init__(self, input_size=4, hidden_size=128, num_layers=2, dropout=0.2):
        super().__init__()
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=True,
        )

        # Attention over all timesteps — fixes the broken last-step extraction
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

        self.norm    = nn.LayerNorm(hidden_size * 2)
        self.dropout = nn.Dropout(dropout)

        # Raw logit output — NO sigmoid here
        # Sigmoid is applied externally: BCEWithLogitsLoss during training,
        # torch.sigmoid() during inference in control_plane.py
        self.fc = nn.Linear(hidden_size * 2, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)               # [B, T, H*2]

        attn_scores  = self.attention(lstm_out)           # [B, T, 1]
        attn_weights = torch.softmax(attn_scores, dim=1)  # [B, T, 1]
        context      = (attn_weights * lstm_out).sum(dim=1)  # [B, H*2]

        context = self.norm(context)
        context = self.dropout(context)
        return self.fc(context)                  # [B, 1] — raw logit
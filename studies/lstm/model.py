import torch
from torch import nn


class LSTMModel(nn.Module):
    """LSTM stateful: riceve le 112 feature riga per riga e restituisce [t0, t1]."""

    def __init__(self, n_features=112, hidden=128, layers=2, dropout=0.0):
        super().__init__()
        self.config = dict(n_features=n_features, hidden=hidden, layers=layers, dropout=dropout)
        self.lstm = nn.LSTM(n_features, hidden, layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Linear(hidden, 2)

    def forward(self, x, state=None):
        out, state = self.lstm(x, state)
        return self.head(out), state


class StepModel(nn.Module):
    """Wrapper per l'export ONNX: un singolo step con stato (h, c) esplicito."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, features, h, c):
        pred, (h, c) = self.model(features, (h, c))
        return pred, h, c


def weighted_pearson_torch(y, p):
    """Weighted Pearson della metrica (clip a [-2,2], pesi |y|), su tensori 1D. Differenziabile in p."""
    y = y.clamp(-2, 2)
    w = y.abs()
    total = w.sum().clamp_min(1e-8)
    yc = y - (w * y).sum() / total
    pc = p - (w * p).sum() / total
    cov = (w * yc * pc).sum() / total
    sy = ((w * yc * yc).sum() / total).sqrt()
    sp = ((w * pc * pc).sum() / total).sqrt()
    return cov / (sy * sp).clamp_min(1e-8)

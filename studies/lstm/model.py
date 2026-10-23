import torch
from torch import nn


class LSTMModel(nn.Module):
    """RNN stateful (LSTM o GRU): riceve le feature riga per riga e restituisce [t0, t1].

    Opzioni (tutte spente per default, così i vecchi checkpoint si caricano uguale):
      proj           : se > 0, Linear(n_features -> proj) + GELU davanti alla RNN
      input_dropout  : dropout sugli input (solo in training)
      input_noise    : rumore gaussiano sugli input (solo in training)
      cell           : 'lstm' oppure 'gru'
      lags           : metadato: l'input (n_features = 112*(1+len(lags))) contiene anche le differenze x[t]-x[t-k]
                       per ogni lag (le calcola features.with_lags fuori dal modello)
      forget_bias    : valore iniziale del bias della porta di dimenticanza dell'LSTM (solo inizializzazione)
      skip           : la testa vede anche le feature in ingresso (collegamento diretto input -> uscita)
      head_hidden    : se > 0, la testa è Linear -> GELU -> Linear con questo strato intermedio
      diff           : vecchio metadato (equivale a lags=[1]), mantenuto per i checkpoint già salvati
    """

    def __init__(self, n_features=112, hidden=128, layers=2, dropout=0.0, proj=0,
                 input_dropout=0.0, input_noise=0.0, cell='lstm', diff=False, lags=None, skip=False, head_hidden=0,
                 forget_bias=0.0):
        super().__init__()
        self.config = dict(n_features=n_features, hidden=hidden, layers=layers, dropout=dropout, proj=proj,
                           input_dropout=input_dropout, input_noise=input_noise, cell=cell, diff=diff,
                           lags=list(lags or []), skip=skip, head_hidden=head_hidden,
                           forget_bias=forget_bias)
        self.input_noise = input_noise
        self.in_drop = nn.Dropout(input_dropout)
        self.proj = nn.Sequential(nn.Linear(n_features, proj), nn.GELU()) if proj else None
        rnn = {'lstm': nn.LSTM, 'gru': nn.GRU}[cell]
        # il nome `lstm` resta anche per la GRU: è la chiave dei pesi nei checkpoint
        self.lstm = rnn(proj or n_features, hidden, layers, batch_first=True,
                        dropout=dropout if layers > 1 else 0.0)
        if forget_bias and cell == 'lstm':
            with torch.no_grad():
                for name, p in self.lstm.named_parameters():
                    if name.startswith('bias_ih'):
                        p[hidden:2 * hidden].fill_(forget_bias)    # ordine delle porte: i, f, g, o
        head_in = hidden + (n_features if skip else 0)
        self.skip = skip
        self.head = nn.Sequential(nn.Linear(head_in, head_hidden), nn.GELU(), nn.Linear(head_hidden, 2)) \
            if head_hidden else nn.Linear(head_in, 2)

    def forward(self, x, state=None):
        if self.training and self.input_noise > 0:
            x = x + self.input_noise * torch.randn_like(x)
        x = self.in_drop(x)
        raw = x
        if self.proj is not None:
            x = self.proj(x)
        out, state = self.lstm(x, state)
        if self.skip:
            out = torch.cat([out, raw], dim=-1)
        return self.head(out), state


def detach_state(state):
    """Stacca lo stato dal grafo (tupla (h, c) per l'LSTM, tensore h per la GRU)."""
    return tuple(h.detach() for h in state) if isinstance(state, tuple) else state.detach()


class StepModel(nn.Module):
    """Wrapper per l'export ONNX: un singolo step con stato (h, c) esplicito (LSTM)."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, features, h, c):
        pred, (h, c) = self.model(features, (h, c))
        return pred, h, c


class StepModelGRU(nn.Module):
    """Come StepModel ma per la GRU: lo stato è il solo h."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, features, h):
        pred, h = self.model(features, h)
        return pred, h


def weighted_pearson_torch(y, p, extra_w=None):
    """Weighted Pearson della metrica (clip a [-2,2], pesi |y|), su tensori 1D. Differenziabile in p.
    extra_w: pesi aggiuntivi per riga (es. probabilità che la riga venga valutata), moltiplicati a |y|."""
    y = y.clamp(-2, 2)
    w = y.abs()
    if extra_w is not None:
        w = w * extra_w
    total = w.sum().clamp_min(1e-8)
    yc = y - (w * y).sum() / total
    pc = p - (w * p).sum() / total
    cov = (w * yc * pc).sum() / total
    sy = ((w * yc * yc).sum() / total).sqrt()
    sp = ((w * pc * pc).sum() / total).sqrt()
    return cov / (sy * sp).clamp_min(1e-8)

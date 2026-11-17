"""LSTM con testa HL-Gauss: il modello di tutte le run (insegnanti e studenti).

Il modello legge le righe di una sequenza in ordine, con stato, e restituisce [t0, t1] per ogni riga.
- Testa principale: per ciascun target una distribuzione su `tick_bins` valori equispaziati in [-2, 2] (HL-Gauss,
  Farebrother et al. 2024). L'uscita è E[|y| y] / E[|y|] sulla distribuzione: il valore che ottimizza la Pearson pesata
  con |y| della metrica.
- Testa ausiliaria `yfut_head` (solo in addestramento): variazioni future vere del target (train.yfut_targets).
In inferenza conta solo la LSTM + testa principale; la scala di calibrazione è nel checkpoint (scale_t0, scale_t1).
"""
import torch
from torch import nn


class LSTMModel(nn.Module):
    def __init__(self, n_features=112, hidden=256, layers=3, tick_bins=81, aux_yfut=20, lookahead=0):
        super().__init__()
        # lookahead: solo annotazione. Un insegnante con anticipo k è stato addestrato con gli ingressi spostati di k righe
        # (alla riga t legge x_{t+k}); in distillazione train.py riallinea le sue predizioni.
        self.config = dict(n_features=n_features, hidden=hidden, layers=layers, tick_bins=tick_bins, aux_yfut=aux_yfut,
                           lookahead=lookahead)
        self.tick_bins = tick_bins
        # l'ordine di creazione dei moduli fissa l'inizializzazione casuale: lstm, head, yfut_head
        self.lstm = nn.LSTM(n_features, hidden, layers, batch_first=True)
        self.head = nn.Linear(hidden, 2 * tick_bins)
        self.yfut_head = nn.Linear(hidden, aux_yfut) if aux_yfut else None
        self.register_buffer('centers', torch.linspace(-2.0, 2.0, tick_bins), persistent=False)
        self.last_logits = None         # (B, T, 2, K), solo in addestramento: entropia incrociata HL-Gauss
        self.last_yfut = None           # (B, T, aux_yfut), solo in addestramento

    def forward(self, x, state=None):
        out, state = self.lstm(x, state)
        if self.yfut_head is not None and self.training:
            self.last_yfut = self.yfut_head(out)
        logits = self.head(out).view(*out.shape[:-1], 2, self.tick_bins)
        if self.training:
            self.last_logits = logits
        pr = torch.softmax(logits, -1)
        a = self.centers.abs()
        return (pr * a * self.centers).sum(-1) / (pr * a).sum(-1).clamp_min(1e-6), state


def build_model(config):
    return LSTMModel(**config)


def detach_state(state):
    """Stacca lo stato (h, c) dal grafo tra un blocco di BPTT e il successivo."""
    return tuple(s.detach() for s in state)


def weighted_pearson_torch(y, p):
    """Weighted Pearson della metrica (y tagliato a ±2, pesi |y|) su tensori 1D."""
    y = y.clamp(-2, 2)
    w = y.abs()
    total = w.sum().clamp_min(1e-8)
    yc = y - (w * y).sum() / total
    pc = p - (w * p).sum() / total
    cov = (w * yc * pc).sum() / total
    sy = ((w * yc * yc).sum() / total).sqrt()
    sp = ((w * pc * pc).sum() / total).sqrt()
    return cov / (sy * sp).clamp_min(1e-8)


def weighted_corr(a, b, w):
    """Correlazione pesata (pesi w dati) tra due vettori 1D; differenziabile in a."""
    tot = w.sum().clamp_min(1e-8)
    ac, bc = a - (w * a).sum() / tot, b - (w * b).sum() / tot
    den = (((w * ac * ac).sum() / tot).sqrt() * ((w * bc * bc).sum() / tot).sqrt()).clamp_min(1e-8)
    return (w * ac * bc).sum() / tot / den

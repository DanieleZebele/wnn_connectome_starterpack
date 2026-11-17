"""Probabilità che una riga venga valutata (is_scored), per i pesi della loss di train.py.

La maschera is_scored esiste solo nel validation e si prevede bene dagli ingressi (AUC 0.958 sulle sequenze mai viste).
Una LSTM bidirezionale a finestre da 1000 righe (usa anche il contesto futuro: va bene per pesare il training, non serve in
gara) si addestra sull'80% delle sequenze di validation e stima p per ogni riga di train e valid:
/dev/shm/wnn_cache/pscore_bi_{train,valid}.npy (float16).

    python mask_model.py              # rigenera pscore_bi dalla rete salvata in runs/mask_net/ (identico bit per bit, ~2 min)
    python mask_model.py --train      # riaddestra la rete (14 epoche, ~1 ora) e la salva in runs/mask_net/
"""
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn

import data

WARMUP = 99
OUT = Path(__file__).resolve().parent / 'runs' / 'mask_net'
TAG = 'pscore_bi'


class MaskNet(nn.Module):
    def __init__(self, hidden=128, layers=2):
        super().__init__()
        self.lstm = nn.LSTM(112, hidden, layers, batch_first=True, bidirectional=True)
        self.head = nn.Linear(2 * hidden, 1)

    def forward(self, x):
        o, _ = self.lstm(x)
        return self.head(o).squeeze(-1)


def batches(X, idx, bs):
    for b in range(0, len(idx), bs):
        ii = idx[b:b + bs]
        yield ii, torch.from_numpy(np.array(X[ii])).cuda().float()


def auc(score, y):
    order = score.argsort(); r = torch.empty_like(score); r[order] = torch.arange(len(score), device=score.device, dtype=score.dtype)
    pos = y > .5; n1, n0 = pos.sum(), (~pos).sum()
    return float((r[pos].sum() - n1 * (n1 - 1) / 2) / (n1 * n0))


@torch.no_grad()
def predict(net, X, idx, bs=64, chunk=1000):
    """p(scored) per riga, (len(idx), 20000) float16; ogni finestra di 1000 righe è indipendente."""
    net.eval()
    out = np.zeros((len(idx), X.shape[1]), np.float16)
    for k, (ii, x) in enumerate(batches(X, idx, bs)):
        ps = [torch.sigmoid(net(x[:, s:s + chunk])) for s in range(0, x.shape[1], chunk)]
        out[k * bs:k * bs + len(ii)] = torch.cat(ps, 1).cpu().numpy().astype(np.float16)
    return out


def train(epochs=14, hold=0.2, seed=0):
    va = data.load('valid')
    N = len(va['X']); rng = np.random.default_rng(seed)
    perm = rng.permutation(N); n_hold = int(hold * N)
    hold_idx, fit_idx = np.sort(perm[:n_hold]), np.sort(perm[n_hold:])
    torch.manual_seed(seed)
    net = MaskNet().cuda()
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3)
    steps = epochs * int(np.ceil(len(fit_idx) / 32)) * 20
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda t: 0.02 + 0.98 * 0.5 * (1 + np.cos(np.pi * min(t / steps, 1))))
    Xv, Sv = va['X'], va['scored']
    for ep in range(1, epochs + 1):
        net.train()
        for ii, x in batches(Xv, fit_idx[rng.permutation(len(fit_idx))], 32):
            y = torch.from_numpy(np.array(Sv[ii])).cuda().float()
            for s in range(0, x.shape[1], 1000):
                logit = net(x[:, s:s + 1000])
                lo = max(WARMUP - s, 0)
                loss = nn.functional.binary_cross_entropy_with_logits(logit[:, lo:], y[:, s + lo:s + 1000])
                opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step()
        p = predict(net, Xv, hold_idx)
        ps = torch.from_numpy(p[:, WARMUP:].astype(np.float32)).cuda().flatten()[::5]
        ys = torch.from_numpy(np.array(Sv[hold_idx])[:, WARMUP:]).cuda().float().flatten()[::5]
        print(f'epoca {ep:2d}: AUC sulle sequenze escluse {auc(ps, ys):.4f}  P(scored) vera {ys.mean():.3f} predetta {ps.mean():.3f}', flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), OUT / 'mask_net.pt')
    return net


def write_scores(net):
    for split in ('valid', 'train'):
        d = data.load(split)
        p = predict(net, d['X'], np.arange(len(d['X'])))
        np.save(data.CACHE / f'{TAG}_{split}.npy', p)
        print(f'{split}: p media {p[:, WARMUP:].mean():.3f} -> {data.CACHE / f"{TAG}_{split}.npy"}', flush=True)


if __name__ == '__main__':
    if '--train' in sys.argv:
        net = train()
    else:
        net = MaskNet().cuda()
        net.load_state_dict(torch.load(OUT / 'mask_net.pt'))
    write_scores(net)

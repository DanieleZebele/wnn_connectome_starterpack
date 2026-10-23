"""Classificatore causale della maschera is_scored (LSTM che legge le righe in ordine, come il modello principale).

Addestrato su una parte del validation (le sole sequenze con la maschera) e usato per stimare, su ogni riga del train,
la probabilità p(scored | passato) che quella riga venga valutata. Serve a pesare la loss di training (--score-weights).
Le sequenze di validation NON usate qui (hold-out pulito) servono a valutare i modelli senza contaminazione.

    python mask_model.py          # causale: scrive /dev/shm/wnn_cache/pscore_{train,valid}.npy (float16) e mask_split.npz
    python mask_model.py --bidir [--hidden 192 --epochs 20 --tag pscore_bi2]
                                  # bidirezionale a finestre da 1000 righe: usa anche il contesto futuro (va bene per i
                                  # pesi di training, non per l'inferenza): scrive pscore_bi_{train,valid}.npy
"""
from pathlib import Path

import numpy as np
import torch
from torch import nn

import data

CACHE = Path('/dev/shm/wnn_cache')
WARMUP = 99


class MaskNet(nn.Module):
    def __init__(self, hidden=128, layers=2, bidir=False):
        super().__init__()
        self.bidir = bidir
        self.lstm = nn.LSTM(112, hidden, layers, batch_first=True, bidirectional=bidir)
        self.head = nn.Linear(hidden * (2 if bidir else 1), 1)

    def forward(self, x, state=None):
        o, state = self.lstm(x, None if self.bidir else state)
        return self.head(o).squeeze(-1), state


def batches(X, idx, bs):
    for b in range(0, len(idx), bs):
        ii = idx[b:b + bs]
        yield ii, torch.from_numpy(np.array(X[ii])).cuda().float()


def auc(score, y):
    order = score.argsort(); r = torch.empty_like(score); r[order] = torch.arange(len(score), device=score.device, dtype=score.dtype)
    pos = y > .5; n1, n0 = pos.sum(), (~pos).sum()
    return float((r[pos].sum() - n1 * (n1 - 1) / 2) / (n1 * n0))


@torch.no_grad()
def predict(net, X, idx, bs=64, chunk=2000):
    net.eval()
    chunk = 1000 if net.bidir else chunk
    out = np.zeros((len(idx), X.shape[1]), np.float16)
    for k, (ii, x) in enumerate(batches(X, idx, bs)):
        state, ps = None, []
        for s in range(0, x.shape[1], chunk):
            logit, state = net(x[:, s:s + chunk], state)
            ps.append(torch.sigmoid(logit))
        out[k * bs:k * bs + len(ii)] = torch.cat(ps, 1).cpu().numpy().astype(np.float16)
    return out


def main(epochs=14, hold=0.2, seed=0, bidir=False, hidden=128, tag=None):
    tag = tag or ('pscore_bi' if bidir else 'pscore')
    va, tr = data.load('valid'), data.load('train')
    N = len(va['X']); rng = np.random.default_rng(seed)
    perm = rng.permutation(N); n_hold = int(hold * N)
    hold_idx, fit_idx = np.sort(perm[:n_hold]), np.sort(perm[n_hold:])
    if not (CACHE / 'mask_split.npz').exists():   # lo split resta lo stesso tra le due versioni (hold-out pulito)
        np.savez(CACHE / 'mask_split.npz', fit=fit_idx, hold=hold_idx)
    else:
        z = np.load(CACHE / 'mask_split.npz'); fit_idx, hold_idx = z['fit'], z['hold']
    torch.manual_seed(seed)
    net = MaskNet(hidden=hidden, bidir=bidir).cuda()
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3)
    steps = epochs * int(np.ceil(len(fit_idx) / 32)) * 20
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda t: 0.02 + 0.98 * 0.5 * (1 + np.cos(np.pi * min(t / steps, 1))))
    Xv, Sv = va['X'], va['scored']
    for ep in range(1, epochs + 1):
        net.train()
        for ii, x in batches(Xv, fit_idx[rng.permutation(len(fit_idx))], 32):
            y = torch.from_numpy(np.array(Sv[ii])).cuda().float()
            state = None
            for s in range(0, x.shape[1], 1000):
                logit, state = net(x[:, s:s + 1000], state)
                state = None if bidir else tuple(t.detach() for t in state)
                lo = max(WARMUP - s, 0)
                if lo >= logit.shape[1]:
                    continue
                loss = nn.functional.binary_cross_entropy_with_logits(logit[:, lo:], y[:, s + lo:s + 1000])
                opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step()
        p = predict(net, Xv, hold_idx)
        ps = torch.from_numpy(p[:, WARMUP:].astype(np.float32)).cuda().flatten()[::5]
        ys = torch.from_numpy(np.array(Sv[hold_idx])[:, WARMUP:]).cuda().float().flatten()[::5]
        bce = nn.functional.binary_cross_entropy(ps.clamp(1e-6, 1 - 1e-6), ys).item()
        print(f'epoca {ep:2d}: hold-out AUC {auc(ps, ys):.4f}  BCE {bce:.4f}  P(scored) vero {ys.mean():.3f} predetto {ps.mean():.3f}', flush=True)
    torch.save(net.state_dict(), CACHE / f'mask_net_{tag}.pt')
    np.save(CACHE / f'{tag}_valid.npy', predict(net, Xv, np.arange(N)))
    ptr = predict(net, tr['X'], np.arange(len(tr['X'])))
    np.save(CACHE / f'{tag}_train.npy', ptr)
    print('pscore train: media %.3f, quantili 50/90/99%%: %s' % (ptr[:, WARMUP:].mean(), np.percentile(ptr[::20, WARMUP:], [50, 90, 99]).round(3)))


if __name__ == '__main__':
    import sys
    arg = lambda k, d: (type(d) if d is not None else str)(sys.argv[sys.argv.index(k) + 1]) if k in sys.argv else d
    main(epochs=arg('--epochs', 14), hidden=arg('--hidden', 128), tag=arg('--tag', None), bidir='--bidir' in sys.argv)

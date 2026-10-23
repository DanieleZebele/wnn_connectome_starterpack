"""Analisi post-hoc sulle predizioni di modelli già addestrati (nessun training).

Le predizioni sul validation vengono calcolate una volta e messe in cache in /dev/shm/wnn_preds.

    python posthoc.py smooth lstm_h128_l2                       # smoothing causale (EMA) delle predizioni
    python posthoc.py ensemble lstm_h128_l2 lstm_h128_l2_offset # media di più modelli
"""
from pathlib import Path
import sys

import numpy as np
import torch

import data
from features import lags_of, with_lags
from model import LSTMModel, weighted_pearson_torch

HERE = Path(__file__).resolve().parent
CACHE = Path('/dev/shm/wnn_preds')
WARMUP = 99
_valid = {}


def valid():
    if not _valid:
        va = data.load('valid')
        _valid['Y'] = torch.from_numpy(np.array(va['Y'])).cuda()
        _valid['scored'] = torch.from_numpy(np.array(va['scored'])).cuda()
        need = torch.zeros_like(_valid['scored'])
        need[:, WARMUP:] = True
        _valid['need'] = need
        _valid['X'] = va['X']
    return _valid


@torch.no_grad()
def get_preds(run, ckpt='best.pt', batch=256, chunk=2000):
    """Predizioni grezze (1873, 20000, 2) float32 sul validation, in cache."""
    CACHE.mkdir(exist_ok=True)
    path = CACHE / f'{run}__{ckpt[:-3]}.npy'
    if path.exists():
        return torch.from_numpy(np.load(path)).cuda()
    c = torch.load(HERE / 'runs' / run / ckpt, map_location='cuda')
    model = LSTMModel(**c['config']).cuda().eval()
    model.load_state_dict(c['state_dict'])
    X, lags = valid()['X'], lags_of(c['config'])
    out = torch.empty(len(X), X.shape[1], 2, device='cuda')
    for b in range(0, len(X), batch):
        x = torch.from_numpy(np.array(X[b:b + batch])).cuda()      # float16, come in training
        state = None
        for s in range(0, x.shape[1], chunk):
            e = min(s + chunk, x.shape[1])
            p, state = model(with_lags(x, s, e, lags), state)
            out[b:b + batch, s:e] = p
    np.save(path, out.cpu().numpy())
    return out


@torch.no_grad()
def score(P, mask_kind='scored'):
    """Global WP (come la metrica) su righe is_scored oppure su tutte le righe dallo step 99."""
    v = valid()
    mask = v['need'] & v['scored'] if mask_kind == 'scored' else v['need']
    y, p = v['Y'][mask].double(), P[mask].double().clamp(-2, 2)
    return float(np.mean([weighted_pearson_torch(y[:, k], p[:, k]).item() for k in range(2)]))


@torch.no_grad()
def ema_smooth(P, alpha):
    """Filtro causale s[t] = alpha*s[t-1] + (1-alpha)*p[t] lungo gli step, per ogni sequenza."""
    out = torch.empty_like(P)
    s = P[:, 0].clone()
    out[:, 0] = s
    for t in range(1, P.shape[1]):
        s = alpha * s + (1 - alpha) * P[:, t]
        out[:, t] = s
    return out


if __name__ == '__main__':
    cmd, runs = sys.argv[1], sys.argv[2:]
    if cmd == 'smooth':
        P = get_preds(runs[0])
        print(f'{runs[0]}  senza smoothing: scored {score(P):.4f}  tutte {score(P, "all"):.4f}')
        for alpha in (0.5, 0.7, 0.8, 0.9, 0.95, 0.98):
            Ps = ema_smooth(P, alpha)
            print(f'  EMA alpha={alpha:<5} (≈{1 / (1 - alpha):.0f} step): scored {score(Ps):.4f}  tutte {score(Ps, "all"):.4f}')
    elif cmd == 'ensemble':
        Ps = [get_preds(r) for r in runs]
        for r, P in zip(runs, Ps):
            print(f'  {r:28s} scored {score(P):.4f}  tutte {score(P, "all"):.4f}')
        for k in range(2, len(Ps) + 1):
            E = torch.stack(Ps[:k]).mean(0)
            print(f'  media dei primi {k}: scored {score(E):.4f}  tutte {score(E, "all"):.4f}')


def regime_labels(kind='oracle'):
    """Etichette dei regimi del validation: 'oracle' (dai target) oppure 'knn' (dai soli input, come in gara)."""
    import regimes
    rt, _, lt = regimes.load('train')
    rv, _, lv = regimes.load('valid')
    if kind == 'oracle':
        return lv
    d = ((rv[:, None, :5] - rt[None, :, :5]) ** 2).sum(-1)
    votes = lt[np.argsort(d, 1)[:, :5]]
    return np.array([np.bincount(v, minlength=4).argmax() for v in votes]).astype(np.int8)


def per_regime_report(generalist, specialists, ckpt='best.pt'):
    """specialists: dict lettera -> nome della run. Stampa la WP per regime e quella complessiva della combinazione."""
    import regimes
    v = valid()
    Pg = get_preds(generalist, ckpt)
    lab_true = regime_labels('oracle')
    scored = v['need'] & v['scored']

    def wp(P, sel):
        m = scored.clone()
        m[~sel] = False
        y, p = v['Y'][m].double(), P[m].double().clamp(-2, 2)
        return float(np.mean([weighted_pearson_torch(y[:, k], p[:, k]).item() for k in range(2)]))

    print(f'{"regime":8s} {"seq":>5s} {"righe":>9s} {"generalista":>12s} {"specialista":>12s}')
    Pc = Pg.clone()
    Ps = {c: get_preds(r, ckpt) for c, r in specialists.items()}
    for k, c in enumerate(regimes.NAMES):
        sel = torch.from_numpy(lab_true == k).cuda()[:, None].expand_as(scored)
        n = int((scored & sel).sum())
        s = f'{wp(Ps[c], sel):12.4f}' if c in Ps else f'{"":12s}'
        print(f'{c:8s} {int((lab_true == k).sum()):5d} {n:9d} {wp(Pg, sel):12.4f} {s}')
    for kind in ('oracle', 'knn'):
        lab = torch.from_numpy(regime_labels(kind)).cuda()
        Pc = Pg.clone()
        for k, c in enumerate(regimes.NAMES):
            if c in Ps:
                sel = (lab == k)
                Pc[sel] = Ps[c][sel]
        print(f'combinazione (regime {kind}): scored {score(Pc):.4f}  tutte {score(Pc, "all"):.4f}   | generalista: scored {score(Pg):.4f}  tutte {score(Pg, "all"):.4f}')

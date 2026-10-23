"""Regimi (strumenti) delle sequenze.

Ogni sequenza appartiene a uno di 4 regimi con granularità del target diversa (unità: passo minimo non nullo di |t|):
A 0.66, B 0.47, C 0.24, D 0.13. Il regime si riconosce dai valori "di riposo" di dp/dv (moda sui primi 99 step di
i0 e i1), che si leggono dagli input; l'unità dei target serve solo per le etichette di training.

    python regimes.py        # (ri)crea la cache in /dev/shm/wnn_cache/regime_{train,valid}.npz
"""
from pathlib import Path

import numpy as np

import data

CACHE = Path('/dev/shm/wnn_cache')
UNITS = np.array([0.66, 0.47, 0.24, 0.13])          # A, B, C, D
NAMES = 'ABCD'
REST_COLS = [44, 45, 46, 47, 48, 96, 97, 98, 99, 100]   # dp0-3, dv0 di i0 e di i1


def rest_vectors(X, n=99):
    """Moda sui primi n step di REST_COLS, per ogni sequenza: (N, 10)."""
    out = np.zeros((len(X), len(REST_COLS)), np.float32)
    for i in range(len(X)):
        block = np.array(X[i, :n][:, REST_COLS])
        for j in range(block.shape[1]):
            u, k = np.unique(block[:, j], return_counts=True)
            out[i, j] = u[k.argmax()]
    return out


def units(Y):
    """Unità del target per sequenza: min |t0| non nullo (righe >= 99)."""
    out = np.zeros(len(Y))
    for i in range(len(Y)):
        a = np.abs(np.array(Y[i, 99:, 0]))
        a = a[a > 1e-4]
        out[i] = a.min() if len(a) else np.nan
    return out


def build(split):
    d = data.load(split)
    rest, unit = rest_vectors(d['X']), units(d['Y'])
    label = np.abs(unit[:, None] - UNITS[None]).argmin(1).astype(np.int8)
    np.savez(CACHE / f'regime_{split}.npz', rest=rest, unit=unit, label=label)
    return rest, unit, label


def load(split):
    path = CACHE / f'regime_{split}.npz'
    if not path.exists():
        build(split)
    z = np.load(path)
    return z['rest'], z['unit'], z['label']


if __name__ == '__main__':
    for s in ('valid', 'train'):
        r, u, l = build(s)
        print(s, {NAMES[k]: int((l == k).sum()) for k in range(4)}, '| unità fuori dai 4 valori:',
              int((np.abs(u[:, None] - UNITS[None]).min(1) > 0.02).sum()))

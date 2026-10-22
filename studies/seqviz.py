"""Funzioni comuni ai notebook di studies/: caricamento sequenze e utilità di plot."""
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
DATASETS = ROOT / 'datasets'
TRAIN = DATASETS / 'train.parquet'
VALID = DATASETS / 'valid.parquet'
WARMUP = 99
TRADE_COLORS = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']

_files = {}
_seq_index = {}


def open_dataset(path):
    path = Path(path)
    if path not in _files:
        _files[path] = pq.ParquetFile(path)
    return _files[path]


def seq_index(path):
    """seq_ix -> row group. Legge solo la colonna seq_ix (lento sul train, ~minuti)."""
    path = Path(path)
    if path not in _seq_index:
        ids = open_dataset(path).read(columns=['seq_ix'])['seq_ix'].to_numpy()[::20_000]
        _seq_index[path] = {int(s): g for g, s in enumerate(ids)}
    return _seq_index[path]


def load_sequence(path, group=None, seq_ix=None):
    """Carica una sequenza per indice di row group oppure per seq_ix."""
    if seq_ix is not None:
        group = seq_index(path)[seq_ix]
    df = open_dataset(path).read_row_group(group).to_pandas()
    add_price_proxies(df)
    return df


def add_price_proxies(df):
    for inst in ('i0', 'i1'):
        bid = df[[f'{inst}_p{j}' for j in range(11)]].mean(axis=1)
        ask = df[[f'{inst}_p{j}' for j in range(11, 22)]].mean(axis=1)
        df[f'{inst}_bid'] = bid
        df[f'{inst}_ask'] = ask
        df[f'{inst}_mid'] = (bid + ask) / 2


def shade_mask(ax, steps, mask, **kw):
    """Disegna bande verticali sui tratti consecutivi in cui mask è True."""
    edges = np.diff(np.r_[0, mask.astype(int), 0])
    for a, b in zip(np.where(edges == 1)[0], np.where(edges == -1)[0]):
        ax.axvspan(steps[a], steps[b - 1] + 1, lw=0, **kw)

"""Cache dei dataset in /dev/shm come array numpy (feature float16, target float32, maschera is_scored).

Uso:  python data.py [--force]  # crea la cache di train e valid se manca (~58 GB in RAM, qualche minuto)
La cache vive in RAM e sparisce al riavvio della macchina: in quel caso basta rilanciare.
"""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sys
import time

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from utils import FEATURE_COLUMNS, SEQUENCE_LENGTH, TARGET_COLUMNS  # noqa: E402

DATASETS = ROOT / 'datasets'
CACHE = Path('/dev/shm/wnn_cache')
SPLITS = ('train', 'valid')


def _paths(split):
    return {name: CACHE / f'{split}_{name}.npy' for name in ('X', 'Y', 'scored', 'seq')}


def _fill(args):
    split, groups = args
    pf = pq.ParquetFile(DATASETS / f'{split}.parquet')
    p = _paths(split)
    X = np.load(p['X'], mmap_mode='r+')
    Y = np.load(p['Y'], mmap_mode='r+')
    S = np.load(p['scored'], mmap_mode='r+')
    for g in groups:
        t = pf.read_row_group(g, use_threads=False)
        X[g] = np.column_stack([t[c].to_numpy() for c in FEATURE_COLUMNS]).astype(np.float16)
        Y[g] = np.column_stack([t[c].to_numpy() for c in TARGET_COLUMNS])
        if 'is_scored' in t.column_names:
            S[g] = t['is_scored'].to_numpy()
    X.flush(); Y.flush(); S.flush()
    return len(groups)


def build_cache(split, workers=32):
    """Una sequenza per row group del parquet: (N, 20000, 112) float16, (N, 20000, 2) float32, (N, 20000) bool."""
    pf = pq.ParquetFile(DATASETS / f'{split}.parquet')
    n = pf.num_row_groups
    CACHE.mkdir(parents=True, exist_ok=True)
    p = _paths(split)
    np.lib.format.open_memmap(p['X'], 'w+', np.float16, (n, SEQUENCE_LENGTH, len(FEATURE_COLUMNS)))
    np.lib.format.open_memmap(p['Y'], 'w+', np.float32, (n, SEQUENCE_LENGTH, 2))
    np.lib.format.open_memmap(p['scored'], 'w+', np.bool_, (n, SEQUENCE_LENGTH))
    seq = pf.read(columns=['seq_ix'])['seq_ix'].to_numpy()[::SEQUENCE_LENGTH]
    np.save(p['seq'], seq)
    chunks = [(split, list(range(i, n, workers))) for i in range(workers)]
    t0 = time.time()
    with ProcessPoolExecutor(workers) as ex:
        for _ in ex.map(_fill, chunks):
            pass
    print(f'{split}: {n} sequenze in cache ({time.time() - t0:.0f}s)')


def load(split):
    """dict con X (N,20000,112) f16, Y (N,20000,2) f32, scored (N,20000) bool, seq (N,), in memmap dalla cache."""
    p = _paths(split)
    if not p['seq'].exists():
        build_cache(split)
    return {name: np.load(path, mmap_mode='r') for name, path in p.items()}


if __name__ == '__main__':
    for split in SPLITS:
        if '--force' in sys.argv or not _paths(split)['seq'].exists():
            build_cache(split)
        else:
            print(f'{split}: cache già presente in {CACHE} (--force per ricrearla)')

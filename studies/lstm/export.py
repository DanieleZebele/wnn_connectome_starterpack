"""Esporta un checkpoint in ONNX (un step per chiamata), verifica end-to-end, misura la velocità, crea lo ZIP.

Uso:  python studies/lstm/export.py studies/lstm/runs/<nome>/best.pt
Crea runs/<nome>/submission/{solution.py, model.onnx, config.json} e runs/<nome>/submission.zip
(per un checkpoint last.pt: runs/<nome>/submission_last/ e submission_last.zip)
"""
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import time
import zipfile

import numpy as np
import torch

import data
from features import lags_of, with_lags
from model import LSTMModel, StepModel, StepModelGRU

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from utils import DataPoint  # noqa: E402

TEST_TIME_LIMIT_US = 3600 / 39_400_000 * 1e6   # ~91 µs per riga sul test


def main(ckpt_path):
    ckpt_path = Path(ckpt_path)
    ckpt = torch.load(ckpt_path, map_location='cpu')
    cfg = ckpt['config']
    model = LSTMModel(**cfg)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()

    suffix = '' if ckpt_path.stem == 'best' else f'_{ckpt_path.stem}'
    sub = ckpt_path.parent / f'submission{suffix}'
    sub.mkdir(exist_ok=True)
    gru = cfg.get('cell', 'lstm') == 'gru'
    h = torch.zeros(cfg['layers'], 1, cfg['hidden'])
    dummy = (torch.zeros(1, 1, cfg['n_features']), h) if gru else (torch.zeros(1, 1, cfg['n_features']), h, h.clone())
    torch.onnx.export((StepModelGRU if gru else StepModel)(model), dummy, str(sub / 'model.onnx'),
                      input_names=['features', 'h'] if gru else ['features', 'h', 'c'],
                      output_names=['pred', 'h_out'] if gru else ['pred', 'h_out', 'c_out'],
                      dynamo=False, opset_version=17)
    shutil.copy(HERE / 'solution.py', sub / 'solution.py')
    (sub / 'config.json').write_text(json.dumps({'lags': lags_of(cfg), 'cell': cfg.get('cell', 'lstm')}))

    # verifica end-to-end: il solution.py dello ZIP, riga per riga, contro PyTorch sulle prime 3000 righe
    va = data.load('valid')
    X = np.asarray(va['X'][0, :3000], dtype=np.float32)    # già float16 come in training
    with torch.no_grad():
        x = torch.from_numpy(X)[None]
        ref = model(with_lags(x, 0, x.shape[1], lags_of(cfg)))[0][0].numpy()
    spec = importlib.util.spec_from_file_location('submission_solution', sub / 'solution.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    pm = module.PredictionModel()
    out = np.empty((len(X), 2), np.float32)
    t = time.perf_counter()
    for i, row in enumerate(X):
        out[i] = pm.predict(DataPoint(1, i, True, row))
    us = (time.perf_counter() - t) / len(X) * 1e6
    print(f'solution.py vs PyTorch: errore max {np.abs(out - ref).max():.2e}')
    print(f'velocità: {us:.1f} µs/riga (limite test ≈ {TEST_TIME_LIMIT_US:.0f} µs/riga, '
          f'tempo stimato test ≈ {us * 39.4e6 / 60e6:.0f} min; con CPU 1.5x più lenta ≈ {us * 39.4e6 / 60e6 * 1.5:.0f} min)')

    zip_path = ckpt_path.parent / f'submission{suffix}.zip'
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as z:
        for f in ('solution.py', 'model.onnx', 'config.json'):
            z.write(sub / f, f)
    print(f'ZIP: {zip_path} ({zip_path.stat().st_size / 1e6:.2f} MB, limite 20 MB)')


if __name__ == '__main__':
    main(sys.argv[1])

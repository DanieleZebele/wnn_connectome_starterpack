"""Esporta un checkpoint in ONNX (un step per chiamata), verifica e misura la velocità, crea lo ZIP.

Uso:  python studies/lstm/export.py studies/lstm/runs/<nome>/best.pt
Crea runs/<nome>/submission/{solution.py, model.onnx} e runs/<nome>/submission.zip
"""
from pathlib import Path
import shutil
import sys
import time
import zipfile

import numpy as np
import onnxruntime as ort
import torch

import data
from model import LSTMModel, StepModel

HERE = Path(__file__).resolve().parent
TEST_TIME_LIMIT_US = 3600 / 39_400_000 * 1e6   # ~91 µs per riga sul test


def main(ckpt_path):
    ckpt_path = Path(ckpt_path)
    ckpt = torch.load(ckpt_path, map_location='cpu')
    model = LSTMModel(**ckpt['config'])
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    cfg = ckpt['config']

    sub = ckpt_path.parent / 'submission'
    sub.mkdir(exist_ok=True)
    onnx_path = sub / 'model.onnx'
    h = torch.zeros(cfg['layers'], 1, cfg['hidden'])
    torch.onnx.export(StepModel(model), (torch.zeros(1, 1, cfg['n_features']), h, h.clone()), str(onnx_path),
                      input_names=['features', 'h', 'c'], output_names=['pred', 'h_out', 'c_out'],
                      dynamo=False, opset_version=17)
    shutil.copy(HERE / 'solution.py', sub / 'solution.py')

    # verifica: ONNX step-by-step == PyTorch sull'intera prima sequenza del validation (prime 3000 righe)
    X = np.asarray(data.load('valid')['X'][0, :3000], dtype=np.float32)
    with torch.no_grad():
        ref = model(torch.from_numpy(X)[None])[0][0].numpy()
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = opts.inter_op_num_threads = 1
    sess = ort.InferenceSession(str(onnx_path), opts, providers=['CPUExecutionProvider'])
    hs = np.zeros((cfg['layers'], 1, cfg['hidden']), np.float32)
    cs = hs.copy()
    out = np.empty((len(X), 2), np.float32)
    t = time.perf_counter()
    for i, x in enumerate(X):
        p, hs, cs = sess.run(None, {'features': x[None, None], 'h': hs, 'c': cs})
        out[i] = p[0, 0]
    us = (time.perf_counter() - t) / len(X) * 1e6
    err = np.abs(out - ref).max()
    print(f'ONNX vs PyTorch: errore max {err:.2e}')
    print(f'velocità: {us:.1f} µs/riga (limite test ≈ {TEST_TIME_LIMIT_US:.0f} µs/riga, '
          f'tempo stimato test ≈ {us * 39.4e6 / 60e6:.0f} min)')

    zip_path = ckpt_path.parent / 'submission.zip'
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as z:
        for f in ('solution.py', 'model.onnx'):
            z.write(sub / f, f)
    print(f'ZIP: {zip_path} ({zip_path.stat().st_size / 1e6:.2f} MB, limite 20 MB)')


if __name__ == '__main__':
    main(sys.argv[1])

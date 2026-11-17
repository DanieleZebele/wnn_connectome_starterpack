"""Costruisce la consegna (ZIP con kernel C int8) da uno o più checkpoint: l'insieme è la media delle uscite calibrate.

Uso:  python export_c.py OUT_DIR RUN1 [RUN2 ...] [--no-check]
  RUN: nome di una run in runs/ (usa best.pt se c'è, altrimenti last.pt) oppure percorso di un .pt.
  L'ordine conta: è la priorità dei modelli per la protezione del tempo di solution.py (si toglie l'ultimo).
Passi:
  1. pesi LSTM int8 simmetrici per canale d'uscita; le colonne del primo strato assorbono la scala di ogni feature
     (99.9 percentile di |x| su 200 sequenze di train), testa lineare int8 per colonna;
  2. scala di calibrazione di ogni modello (quella del checkpoint) e scala finale dell'insieme (std 0.7 della media sulle
     righe valutate di 150 sequenze di validation), precalcolate per ogni prefisso di k modelli;
  3. compilazione di lstm_int8.so, verifica contro PyTorch fp32 su 3 sequenze, velocità per riga, ZIP.
"""
import importlib.util
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TEMPLATE = HERE / 'submission'           # solution.py e lstm_int8.c copiati nello ZIP
sys.path.insert(0, str(ROOT))
import data  # noqa: E402
from model import build_model  # noqa: E402
from utils import DataPoint  # noqa: E402

CAL_SEQ = slice(None, None, 12)          # 150 sequenze del validation per la scala finale dell'insieme


def find_ckpt(s):
    p = Path(s)
    if p.suffix == '.pt' and p.exists():
        return p
    d = HERE / 'runs' / s
    for f in ('best.pt', 'last.pt'):
        if (d / f).exists():
            return d / f
    raise FileNotFoundError(s)


def raw_preds(model, X):
    out = []
    with torch.no_grad():
        for b in range(0, len(X), 16):
            x = torch.from_numpy(np.asarray(X[b:b + 16], dtype=np.float32))
            state, parts = None, []
            for a in range(0, x.shape[1], 2000):
                p, state = model(x[:, a:a + 2000], state)
                parts.append(p)
            out.append(torch.cat(parts, 1).numpy())
    return np.concatenate(out)


def qcols(W):
    """int8 simmetrico per riga (canale d'uscita): (q, scala)."""
    s = np.abs(W).max(1) / 127.0
    s[s == 0] = 1.0
    return np.clip(np.rint(W / s[:, None]), -127, 127).astype(np.int8), s.astype(np.float32)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    check = '--no-check' not in sys.argv
    out = Path(args[0]); specs = args[1:]
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(16)
    va = data.load('valid'); tr = data.load('train')
    Xc = va['X'][CAL_SEQ][:150]
    scc = np.array(va['scored'][CAL_SEQ][:150]).copy(); scc[:, :99] = False
    xs = np.asarray(tr['X'][::50][:200, 99::7], dtype=np.float32).reshape(-1, tr['X'].shape[2])
    in_s = np.maximum(np.percentile(np.abs(xs), 99.9, axis=0), 1e-3).astype(np.float32)
    Z = {'n_models': np.int64(len(specs))}
    models = []
    for i, name in enumerate(specs):
        ck_path = find_ckpt(name)
        ck = torch.load(ck_path, map_location='cpu'); cfg = ck['config']
        model = build_model(cfg); model.load_state_dict(ck['state_dict']); model.eval()
        sc = np.array([ck['scale_t0'], ck['scale_t1']], np.float32)
        models.append((model, sc, raw_preds(model, Xc) * sc))
        sd = ck['state_dict']; L, H, F = cfg['layers'], cfg['hidden'], cfg['n_features']
        p = f'm{i}_'
        Z[p + 'H'] = np.full(L, H, np.int32); Z[p + 'F'] = np.int64(F); Z[p + 'K'] = np.int64(cfg['tick_bins'])
        Z[p + 'in_scale'] = (1.0 / in_s).astype(np.float32)
        for l in range(L):
            wx = sd[f'lstm.weight_ih_l{l}'].numpy().astype(np.float64)
            if l == 0:
                wx = wx * in_s[None, :]                                      # x' = x / s  ->  W' = W * s
            Z[p + f'wx{l}'], Z[p + f'swx{l}'] = qcols(wx)
            Z[p + f'wh{l}'], Z[p + f'swh{l}'] = qcols(sd[f'lstm.weight_hh_l{l}'].numpy().astype(np.float64))
            Z[p + f'b{l}'] = (sd[f'lstm.bias_ih_l{l}'] + sd[f'lstm.bias_hh_l{l}']).numpy().astype(np.float32)
        q, s_ = qcols(sd['head.weight'].numpy().astype(np.float64))
        Z[p + 'woq'] = q.T.copy(); Z[p + 'sco'] = s_; Z[p + 'bo'] = sd['head.bias'].numpy()
        Z[p + 'centers'] = np.linspace(-2.0, 2.0, cfg['tick_bins']).astype(np.float32)
        Z[p + 'scale'] = sc
        print(f'modello {i}: {ck_path}  {H}x{L}  scala {sc.round(3)}', flush=True)
    # per ogni prefisso k (protezione del tempo): pesi uguali e scala finale (std 0.7 sulla media; 1 per k = 1)
    n = len(specs)
    wk = np.zeros((n, n), np.float32); fk = np.ones((n, 2), np.float32)
    for k in range(1, n + 1):
        wk[k - 1, :k] = 1.0 / k
        if k > 1:
            Pa = sum(wk[k - 1, i] * models[i][2] for i in range(k))
            fk[k - 1] = [0.7 / Pa[..., j][scc].std() for j in range(2)]
    Z['ens_w_k'] = wk; Z['fin_scale_k'] = fk
    print(f'insieme di {n}: scale finali per prefisso {fk.round(3).tolist()}', flush=True)
    np.savez(out / 'weights.npz', **Z)
    shutil.copy(TEMPLATE / 'solution.py', out / 'solution.py')
    shutil.copy(TEMPLATE / 'lstm_int8.c', out / 'lstm_int8.c')
    r = subprocess.run(['gcc', '-O3', '-march=x86-64-v2', '-fPIC', '-shared', '-o', str(out / 'lstm_int8.so'),
                        str(out / 'lstm_int8.c'), '-lm'], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    if check:
        spec = importlib.util.spec_from_file_location('csol', out / 'solution.py')
        mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
        pm = mod.PredictionModel()
        ws, fin = wk[n - 1], fk[n - 1]
        seqs, rows, outs, refs, tt = [0, 700, 1400], 3000, [], [], 0.0
        for s in seqs:
            x = np.asarray(va['X'][s, :rows], dtype=np.float32)
            with torch.no_grad():
                ref = sum(w * m(torch.from_numpy(x)[None])[0][0].numpy() * sc for (m, sc, _), w in zip(models, ws)) * fin
            o = np.zeros((rows, 2), np.float32)
            t = time.perf_counter()
            for i in range(rows):
                p = pm.predict(DataPoint(s, i, i >= 99, x[i]))
                if p is not None:
                    o[i] = p
            tt += time.perf_counter() - t
            outs.append(o[99:]); refs.append(ref[99:])
        O, R = np.concatenate(outs), np.concatenate(refs)
        c = [np.corrcoef(O[:, k], R[:, k])[0, 1] for k in range(2)]
        print(f'C vs PyTorch fp32: corr {c[0]:.6f}/{c[1]:.6f}, max diff {np.abs(O - R).max():.4f}; '
              f'velocità {tt / (len(seqs) * rows) * 1e6:.1f} µs/riga (VNNI {pm._lib.lstm_has_vnni()})', flush=True)
    zp = out.with_suffix('.zip')
    with zipfile.ZipFile(zp, 'w', zipfile.ZIP_DEFLATED) as zf:
        for f in ('solution.py', 'weights.npz', 'lstm_int8.c', 'lstm_int8.so'):
            zf.write(out / f, f)
    print(f'ZIP: {zp} ({zp.stat().st_size / 1e6:.2f} MB)')


if __name__ == '__main__':
    main()

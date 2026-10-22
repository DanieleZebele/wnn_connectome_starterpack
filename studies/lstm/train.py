"""Addestramento LSTM su GPU con truncated BPTT su sequenze intere.

Ogni batch prende B sequenze complete e le percorre a blocchi di `--chunk` step,
passando lo stato da un blocco al successivo (staccato dal grafo): è lo stesso
regime dell'inferenza, che parte da stato zero allo step 0 e arriva al 19.999.

Esempio:
    python studies/lstm/train.py --name lstm_h128 --epochs 10
    python studies/lstm/train.py --name smoke --limit-train 256 --epochs 1   # prova veloce
"""
import argparse
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

import data
from model import LSTMModel, weighted_pearson_torch

WARMUP = 99
RUNS = Path(__file__).resolve().parent / 'runs'


def to_gpu(arr, dtype, block=256):
    """Copia un array (anche memmap) sulla GPU a blocchi, senza duplicarlo in RAM."""
    out = torch.empty(arr.shape, dtype=dtype, device='cuda')
    for i in range(0, len(arr), block):
        out[i:i + block] = torch.from_numpy(np.array(arr[i:i + block])).to('cuda', dtype)
    return out


def loss_fn(kind, pred, y):
    """pred, y: (N, 2) sulle sole righe con need_prediction."""
    if kind == 'pearson':
        return -0.5 * (weighted_pearson_torch(y[:, 0], pred[:, 0]) + weighted_pearson_torch(y[:, 1], pred[:, 1]))
    if kind == 'wmse':
        yc = y.clamp(-2, 2)
        w = yc.abs()
        return (w * (pred - yc) ** 2).sum() / w.sum().clamp_min(1e-8)
    raise ValueError(kind)


@torch.no_grad()
def predict(model, X, batch=512, chunk=2000):
    model.eval()
    preds = torch.empty(X.shape[0], X.shape[1], 2, device='cuda')
    for b in range(0, X.shape[0], batch):
        state = None
        for s in range(0, X.shape[1], chunk):
            p, state = model(X[b:b + batch, s:s + chunk].float(), state)
            preds[b:b + batch, s:s + chunk] = p
    model.train()
    return preds


@torch.no_grad()
def global_wp(Y, P, mask):
    """WP globale come nella metrica ufficiale: per target su tutte le righe selezionate, poi media."""
    y, p = Y[mask].double(), P[mask].double().clamp(-2, 2)
    per = [weighted_pearson_torch(y[:, k], p[:, k]).item() for k in range(2)]
    return float(np.mean(per)), per


def evaluate(model, V):
    P = predict(model, V['X'])
    need = torch.zeros(P.shape[:2], dtype=torch.bool, device='cuda')
    need[:, WARMUP:] = True
    wp_scored, per = global_wp(V['Y'], P, need & V['scored'])
    wp_all, _ = global_wp(V['Y'], P, need)
    return {'wp_scored': wp_scored, 't0': per[0], 't1': per[1], 'wp_all_rows': wp_all}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', required=True)
    ap.add_argument('--hidden', type=int, default=256)
    ap.add_argument('--layers', type=int, default=2)
    ap.add_argument('--dropout', type=float, default=0.0)
    ap.add_argument('--loss', choices=['pearson', 'wmse'], default='pearson')
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--batch', type=int, default=256, help='sequenze per batch')
    ap.add_argument('--chunk', type=int, default=1000, help='step per blocco di BPTT')
    ap.add_argument('--lr', type=float, default=2e-3)
    ap.add_argument('--weight-decay', type=float, default=0.0)
    ap.add_argument('--limit-train', type=int, default=0, help='usa solo le prime N sequenze (debug)')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    out_dir = RUNS / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    t = time.time()
    tr, va = data.load('train'), data.load('valid')
    n_train = args.limit_train or len(tr['X'])
    X = to_gpu(tr['X'][:n_train], torch.float16)
    Y = to_gpu(tr['Y'][:n_train], torch.float32)
    V = {'X': to_gpu(va['X'], torch.float16), 'Y': to_gpu(va['Y'], torch.float32),
         'scored': to_gpu(va['scored'], torch.bool)}
    print(f'dati su GPU: train {n_train} seq, valid {len(V["X"])} seq ({time.time() - t:.0f}s)', flush=True)

    model = LSTMModel(hidden=args.hidden, layers=args.layers, dropout=args.dropout).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    seq_len = X.shape[1]
    steps_per_epoch = math.ceil(n_train / args.batch) * math.ceil(seq_len / args.chunk)
    total_steps = args.epochs * steps_per_epoch
    warmup = max(1, min(200, total_steps // 20))

    def lr_factor(step):   # warmup lineare, poi coseno fino al 2% del lr
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor)
    print(f'parametri: {sum(p.numel() for p in model.parameters()):,}  step/epoca: {steps_per_epoch}', flush=True)
    (out_dir / 'config.json').write_text(json.dumps(vars(args) | {'model': model.config}, indent=1))

    best = -1.0
    for epoch in range(1, args.epochs + 1):
        t = time.time()
        losses = []
        perm = torch.randperm(n_train, device='cuda')
        for b in range(0, n_train, args.batch):
            idx = perm[b:b + args.batch]
            xb, yb = X[idx], Y[idx]
            state = None
            for s in range(0, seq_len, args.chunk):
                pred, state = model(xb[:, s:s + args.chunk].float(), state)
                state = tuple(h.detach() for h in state)
                start = max(WARMUP - s, 0)          # niente loss sugli step di warm-up
                if start >= pred.shape[1]:
                    continue
                loss = loss_fn(args.loss, pred[:, start:].reshape(-1, 2), yb[:, s + start:s + args.chunk].reshape(-1, 2))
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                losses.append(loss.item())
        metrics = evaluate(model, V)
        log = {'epoch': epoch, 'train_loss': float(np.mean(losses)), **metrics,
               'lr': sched.get_last_lr()[0], 'sec': round(time.time() - t)}
        print(json.dumps(log), flush=True)
        with open(out_dir / 'log.jsonl', 'a') as f:
            f.write(json.dumps(log) + '\n')
        if metrics['wp_scored'] > best:
            best = metrics['wp_scored']
            torch.save({'state_dict': model.state_dict(), 'config': model.config, 'epoch': epoch, **metrics},
                       out_dir / 'best.pt')
    torch.save({'state_dict': model.state_dict(), 'config': model.config, 'epoch': args.epochs}, out_dir / 'last.pt')
    print(f'migliore wp_scored: {best:.4f}  -> {out_dir / "best.pt"}')


if __name__ == '__main__':
    main()

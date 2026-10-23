"""Addestramento RNN su GPU con truncated BPTT su sequenze intere.

Ogni batch prende B sequenze complete e le percorre a blocchi di `--chunk` step,
passando lo stato da un blocco al successivo (staccato dal grafo): è lo stesso
regime dell'inferenza, che parte da stato zero allo step 0 e arriva al 19.999.

Esempio:
    python studies/lstm/train.py --name lstm_h128 --epochs 10
    python studies/lstm/train.py --name smoke --limit-train 256 --epochs 1   # prova veloce

Per eseguire molti esperimenti di seguito senza ricaricare i dati ogni volta: sweep.py.
"""
import argparse
import copy
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

import data
import importlib

import features
import regimes
importlib.reload(features)   # sweep.py ricarica train.py: così si prende anche l'ultima versione di features.py
from features import lags_of, with_lags
from model import LSTMModel, detach_state, weighted_pearson_torch

WARMUP = 99
RUNS = Path(__file__).resolve().parent / 'runs'


def to_gpu(arr, dtype, block=256):
    """Copia uno o più array (anche memmap) sulla GPU a blocchi, concatenandoli sul primo asse,
    senza duplicarli in RAM."""
    arrays = arr if isinstance(arr, (list, tuple)) else [arr]
    out = torch.empty((sum(len(a) for a in arrays), *arrays[0].shape[1:]), dtype=dtype, device='cuda')
    offset = 0
    for a in arrays:
        for i in range(0, len(a), block):
            chunk = torch.from_numpy(np.array(a[i:i + block])).to('cuda', dtype)
            out[offset + i:offset + i + len(chunk)] = chunk
        offset += len(a)
    return out


def wmse(pred, y, extra_w=None):
    yc = y.clamp(-2, 2)
    w = yc.abs() if extra_w is None else yc.abs() * extra_w[:, None]
    return (w * (pred - yc) ** 2).sum() / w.sum().clamp_min(1e-8)


def pearson_loss(pred, y, extra_w=None):
    return -0.5 * (weighted_pearson_torch(y[:, 0], pred[:, 0], extra_w) +
                   weighted_pearson_torch(y[:, 1], pred[:, 1], extra_w))


def loss_fn(kind, pred, y, mix_lambda=0.2, extra_w=None):
    """pred, y: (N, 2) sulle sole righe con need_prediction. extra_w: (N,) pesi aggiuntivi per riga."""
    if kind == 'pearson':
        return pearson_loss(pred, y, extra_w)
    if kind == 'wmse':
        return wmse(pred, y, extra_w)
    if kind == 'mix':   # metrica + ancoraggio della scala delle predizioni
        return pearson_loss(pred, y, extra_w) + mix_lambda * wmse(pred, y, extra_w)
    raise ValueError(kind)


@torch.no_grad()
def predict(model, X, batch=512, chunk=2000):
    model.eval()
    lags = lags_of(getattr(model, 'config', {}))
    preds = torch.empty(X.shape[0], X.shape[1], 2, device='cuda')
    for b in range(0, X.shape[0], batch):
        state = None
        for s in range(0, X.shape[1], chunk):
            x = with_lags(X[b:b + batch], s, min(s + chunk, X.shape[1]), lags)
            p, state = model(x, state)
            preds[b:b + batch, s:s + chunk] = p
    model.train()
    return preds


@torch.no_grad()
def global_wp(Y, P, mask):
    """WP globale come nella metrica ufficiale: per target su tutte le righe selezionate, poi media."""
    y, p = Y[mask].double(), P[mask].double().clamp(-2, 2)
    per = [weighted_pearson_torch(y[:, k], p[:, k]).item() for k in range(2)]
    return float(np.mean(per)), per


def evaluate(model, V, idx=None):
    """idx: indici delle sequenze del validation da usare (None = tutte)."""
    Vx, Vy, Vs = (V['X'], V['Y'], V['scored']) if idx is None else (V['X'][idx], V['Y'][idx], V['scored'][idx])
    P = predict(model, Vx)
    need = torch.zeros(P.shape[:2], dtype=torch.bool, device='cuda')
    need[:, WARMUP:] = True
    wp_scored, per = global_wp(Vy, P, need & Vs)
    wp_all, _ = global_wp(Vy, P, need)
    return {'wp_scored': wp_scored, 't0': per[0], 't1': per[1], 'wp_all_rows': wp_all}


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', required=True)
    ap.add_argument('--hidden', type=int, default=128)
    ap.add_argument('--layers', type=int, default=2)
    ap.add_argument('--dropout', type=float, default=0.0, help='dropout tra gli strati RNN')
    ap.add_argument('--cell', choices=['lstm', 'gru'], default='lstm')
    ap.add_argument('--proj', type=int, default=0, help='se > 0: Linear+GELU di questa dimensione prima della RNN')
    ap.add_argument('--forget-bias', type=float, default=0.0, help='bias iniziale della porta di dimenticanza dell\'LSTM')
    ap.add_argument('--skip', action='store_true', help='la testa vede anche le feature in ingresso')
    ap.add_argument('--head-hidden', type=int, default=0, help='se > 0: testa a due strati con questo strato intermedio')
    ap.add_argument('--input-dropout', type=float, default=0.0, help='dropout sulle feature in ingresso')
    ap.add_argument('--lags', default='',
                    help='lista di lag separati da virgola (es. 1,10,50,200): aggiunge in ingresso x[t]-x[t-k] per ogni lag')
    ap.add_argument('--diff', action='store_true', help='come --lags 1')
    ap.add_argument('--input-noise', type=float, default=0.0, help='deviazione standard del rumore sulle feature')
    ap.add_argument('--loss', choices=['pearson', 'wmse', 'mix'], default='pearson')
    ap.add_argument('--mix-lambda', type=float, default=0.2, help='peso del wmse nella loss mix')
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--batch', type=int, default=256, help='sequenze per batch')
    ap.add_argument('--chunk', type=int, default=1000, help='step per blocco di BPTT')
    ap.add_argument('--random-offset', action='store_true',
                    help='primo blocco di lunghezza casuale in [1, chunk] a ogni batch: i bordi dei blocchi cambiano')
    ap.add_argument('--lr', type=float, default=2e-3)
    ap.add_argument('--lr-final', type=float, default=0.02, help='lr finale come frazione del massimo')
    ap.add_argument('--weight-decay', type=float, default=0.0)
    ap.add_argument('--grad-clip', type=float, default=1.0)
    ap.add_argument('--ema', type=float, default=0.0,
                    help='se > 0: media mobile esponenziale dei pesi con questo decadimento (es. 0.999); '
                         'metriche e checkpoint usano i pesi mediati')
    ap.add_argument('--regime', default='', help='lettere dei regimi (A,B,C,D) su cui addestrare e valutare, es. A oppure BC; '
                                                 'vuoto = tutti. Vedi regimes.py')
    ap.add_argument('--score-weights', type=float, default=0.0,
                    help='se > 0: la loss pesa ogni riga anche per (floor + p)^alpha, dove p è la probabilità stimata che la '
                         'riga venga valutata (mask_model.py). 1 = proporzionale a p, 2 = più marcato')
    ap.add_argument('--score-weights-end', type=float, default=-1.0,
                    help='se >= 0: l\'esponente alpha passa linearmente da --score-weights a questo valore tra la prima e l\'ultima epoca')
    ap.add_argument('--score-file', default='pscore', help="probabilità da usare: 'pscore' (causale) o 'pscore_bi' (bidirezionale)")
    ap.add_argument('--score-floor', type=float, default=0.02, help='valore minimo aggiunto a p prima dell\'elevamento')
    ap.add_argument('--regime-weights', default='',
                    help="campiona le sequenze di training con probabilità diverse per regime: 'eval' = rapporto tra la quota "
                         "di righe valutate nel validation e la quota di training di ogni regime, 'sqrt' = radice di 'eval', "
                         "oppure esplicito 'A:1.1,B:0.25,C:1.3,D:2.3'")
    ap.add_argument('--limit-train', type=int, default=0, help='usa solo le prime N sequenze (debug)')
    ap.add_argument('--limit-valid', type=int, default=0, help='valuta solo sulle prime N sequenze del validation (debug)')
    ap.add_argument('--train-on', choices=['train', 'train+valid'], default='train',
                    help='train+valid: modello finale addestrato anche sul validation. Le metriche sul '
                         'validation diventano in-sample: niente best.pt, si usa last.pt con epoche fisse')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--force', action='store_true', help='permette di riusare un --name già esistente (sovrascrive)')
    return ap.parse_args(argv)


def load_data(train_on='train', limit_train=0, limit_valid=0):
    """Carica i dati sulla GPU. Restituisce (X, Y, V); con train+valid V è una vista sulla coda di X, Y."""
    t = time.time()
    tr, va = data.load('train'), data.load('valid')
    n_tr, nv = limit_train or len(tr['X']), limit_valid or len(va['X'])
    if train_on == 'train+valid':
        X = to_gpu([tr['X'][:n_tr], va['X'][:nv]], torch.float16)
        Y = to_gpu([tr['Y'][:n_tr], va['Y'][:nv]], torch.float32)
        V = {'X': X[n_tr:], 'Y': Y[n_tr:], 'scored': to_gpu(va['scored'][:nv], torch.bool)}
    else:
        X = to_gpu(tr['X'][:n_tr], torch.float16)
        Y = to_gpu(tr['Y'][:n_tr], torch.float32)
        V = {'X': to_gpu(va['X'][:nv], torch.float16), 'Y': to_gpu(va['Y'][:nv], torch.float32),
             'scored': to_gpu(va['scored'][:nv], torch.bool)}
    print(f'dati su GPU: training su {len(X)} seq ({train_on}), valid {len(V["X"])} seq '
          f'({time.time() - t:.0f}s)', flush=True)
    return X, Y, V


def run(args, X, Y, V):
    """Un esperimento completo. X, Y, V sono i tensori GPU di load_data (non vengono modificati)."""
    out_dir = RUNS / args.name
    if (out_dir / 'log.jsonl').exists() and not args.force:
        raise FileExistsError(f'{out_dir} esiste già: scegli un altro --name (o --force per sovrascrivere)')
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.force:
        (out_dir / 'log.jsonl').unlink(missing_ok=True)

    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    with_valid = args.train_on == 'train+valid'
    if with_valid:
        print('ATTENZIONE: il validation è nel training, le sue metriche sono in-sample. '
              'Si salva solo last.pt.', flush=True)
    # sottoinsieme di sequenze per regime (selezione per indice: nessuna copia dei dati)
    train_idx = valid_idx = None
    if args.regime:
        wanted = [regimes.NAMES.index(c) for c in args.regime.upper()]
        lab_va = regimes.load('valid')[2][:len(V['X'])]
        lab_tr = regimes.load('train')[2][:len(X) - (len(V['X']) if with_valid else 0)]
        labels = np.concatenate([lab_tr, lab_va]) if with_valid else lab_tr
        train_idx = torch.from_numpy(np.nonzero(np.isin(labels, wanted))[0]).cuda()
        valid_idx = torch.from_numpy(np.nonzero(np.isin(lab_va, wanted))[0]).cuda()
        print(f'[{args.name}] regime {args.regime}: {len(train_idx)} sequenze di training, '
              f'{len(valid_idx)} di validation', flush=True)
    n_train = len(X) if train_idx is None else len(train_idx)
    # hold-out "pulito" del validation: sequenze mai usate dal classificatore di maschera (non vale con train+valid)
    clean_idx, split_path = None, Path('/dev/shm/wnn_cache/mask_split.npz')
    if split_path.exists() and not with_valid and not args.regime and len(V['X']) == len(regimes.load('valid')[2]):
        clean_idx = torch.from_numpy(np.load(split_path)['hold']).cuda()

    # pesi di campionamento per regime (importance sampling verso la composizione del validation)
    sample_w = None
    if args.regime_weights:
        lab_tr = regimes.load('train')[2][:len(X) - (len(V['X']) if with_valid else 0)]
        lab_all = np.concatenate([lab_tr, regimes.load('valid')[2][:len(V['X'])]]) if with_valid else lab_tr
        if args.regime_weights in ('eval', 'sqrt'):
            lab_va = regimes.load('valid')[2][:len(V['X'])]
            scored_rows = V['scored'][:, WARMUP:].sum(1).cpu().numpy()
            eval_share = np.array([scored_rows[lab_va == k].sum() for k in range(4)], dtype=np.float64)
            eval_share /= eval_share.sum()
            train_share = np.array([(lab_tr == k).mean() for k in range(4)])
            ratio = eval_share / train_share
            ratio = ratio ** 0.5 if args.regime_weights == 'sqrt' else ratio
        else:
            ratio = np.array([float(dict(kv.split(':') for kv in args.regime_weights.split(','))[c]) for c in regimes.NAMES])
        w = ratio[lab_all]
        if train_idx is not None:
            w = w[train_idx.cpu().numpy()]
        sample_w = torch.from_numpy(w / w.mean()).float().cuda()
        print(f'[{args.name}] pesi di campionamento per regime: ' +
              ' '.join(f'{c}={r:.2f}' for c, r in zip(regimes.NAMES, ratio)), flush=True)

    # pesi per riga dalla probabilità di essere valutata (mask_model.py)
    row_w = None
    if args.score_weights > 0:
        cache = Path('/dev/shm/wnn_cache')
        n_tr_only = len(X) - (len(V['X']) if with_valid else 0)
        ps = np.load(cache / f'{args.score_file}_train.npy', mmap_mode='r')[:n_tr_only]
        if with_valid:
            ps = np.concatenate([ps, np.load(cache / f'{args.score_file}_valid.npy', mmap_mode='r')[:len(V['X'])]])
        p_gpu = torch.from_numpy(np.array(ps)).cuda().half()

        def make_row_w(alpha):
            g = (p_gpu.float() + args.score_floor) ** alpha
            return (g / g[:, WARMUP:].mean()).half()   # media 1 sulle righe da predire; float16

        row_w = make_row_w(args.score_weights)
        sw_note = f'{args.score_weights}' + (f' -> {args.score_weights_end}' if args.score_weights_end >= 0 else '')
        print(f'[{args.name}] pesi per riga: (p+{args.score_floor})^({sw_note})', flush=True)

    lags = sorted({int(k) for k in args.lags.split(',') if k} | ({1} if args.diff else set()))
    model = LSTMModel(n_features=X.shape[2] * (1 + len(lags)), hidden=args.hidden, layers=args.layers,
                      dropout=args.dropout, proj=args.proj, input_dropout=args.input_dropout,
                      input_noise=args.input_noise, cell=args.cell, lags=lags, skip=args.skip,
                      head_hidden=args.head_hidden, forget_bias=args.forget_bias).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    seq_len = X.shape[1]
    chunks_per_seq = math.ceil(seq_len / args.chunk) + (1 if args.random_offset else 0)   # massimo
    steps_per_epoch = math.ceil(n_train / args.batch) * chunks_per_seq
    total_steps = args.epochs * steps_per_epoch
    warmup = max(1, min(200, total_steps // 20))

    def lr_factor(step):   # warmup lineare, poi coseno fino a lr_final
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return args.lr_final + (1 - args.lr_final) * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'[{args.name}] parametri: {n_params:,}  step/epoca: {steps_per_epoch}', flush=True)
    (out_dir / 'config.json').write_text(json.dumps(vars(args) | {'model': model.config, 'n_params': n_params}, indent=1))

    ema_model = None
    if args.ema > 0:
        ema_model = copy.deepcopy(model)
        ema_model.lstm.flatten_parameters()
        for p in ema_model.parameters():
            p.requires_grad_(False)
        ema_params, live_params = list(ema_model.parameters()), list(model.parameters())

    rng = np.random.default_rng(args.seed)
    best, step, t_run = -1.0, 0, time.time()
    for epoch in range(1, args.epochs + 1):
        t = time.time()
        losses = []
        if row_w is not None and args.score_weights_end >= 0 and args.epochs > 1:
            alpha = args.score_weights + (args.score_weights_end - args.score_weights) * (epoch - 1) / (args.epochs - 1)
            row_w = make_row_w(alpha)
        perm = torch.randperm(n_train, device='cuda') if sample_w is None else \
            torch.multinomial(sample_w, n_train, replacement=True)
        if train_idx is not None:
            perm = train_idx[perm]
        for b in range(0, n_train, args.batch):
            idx = perm[b:b + args.batch]
            xb, yb = X[idx], Y[idx]
            state = None
            # la sequenza parte sempre dallo step 0; con --random-offset si sposta solo il primo bordo
            first = int(rng.integers(1, args.chunk + 1)) if args.random_offset else args.chunk
            bounds = [0, *range(first, seq_len, args.chunk), seq_len]
            for s, e in zip(bounds[:-1], bounds[1:]):
                pred, state = model(with_lags(xb, s, e, lags), state)
                state = detach_state(state)
                start = max(WARMUP - s, 0)          # niente loss sugli step di warm-up
                if start >= pred.shape[1]:
                    continue
                ew = None if row_w is None else row_w[idx][:, s + start:e].reshape(-1).float()
                loss = loss_fn(args.loss, pred[:, start:].reshape(-1, 2), yb[:, s + start:e].reshape(-1, 2),
                               args.mix_lambda, ew)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                opt.step()
                sched.step()
                step += 1
                if ema_model is not None:
                    # warmup (1+t)/(10+t) solo nei primi 1/(1-ema) passi, poi il decadimento richiesto
                    d = min(args.ema, (1 + step) / (10 + step)) if step < 1 / (1 - args.ema) else args.ema
                    with torch.no_grad():
                        torch._foreach_mul_(ema_params, d)
                        torch._foreach_add_(ema_params, live_params, alpha=1 - d)
                losses.append(loss.item())
        eval_model = ema_model if ema_model is not None else model
        metrics = evaluate(eval_model, V, valid_idx)
        if clean_idx is not None:
            metrics['wp_scored_clean'] = evaluate(eval_model, V, clean_idx)['wp_scored']
        log = {'epoch': epoch, 'train_loss': float(np.mean(losses)), **metrics,
               'lr': sched.get_last_lr()[0], 'sec': round(time.time() - t)}
        if ema_model is not None:
            raw = evaluate(model, V, valid_idx)
            log['wp_scored_raw'], log['wp_all_rows_raw'] = raw['wp_scored'], raw['wp_all_rows']
        if with_valid:
            log['valid_in_sample'] = True
        print(json.dumps(log), flush=True)
        with open(out_dir / 'log.jsonl', 'a') as f:
            f.write(json.dumps(log) + '\n')
        if not with_valid and metrics['wp_scored'] > best:
            best = metrics['wp_scored']
            torch.save({'state_dict': eval_model.state_dict(), 'config': model.config, 'epoch': epoch, **metrics},
                       out_dir / 'best.pt')
    final = ema_model if ema_model is not None else model
    torch.save({'state_dict': final.state_dict(), 'config': model.config, 'epoch': args.epochs}, out_dir / 'last.pt')
    if with_valid:
        print(f'modello finale (train+valid, {args.epochs} epoche) -> {out_dir / "last.pt"}', flush=True)
    else:
        print(f'[{args.name}] migliore wp_scored: {best:.4f}  -> {out_dir / "best.pt"} '
              f'({(time.time() - t_run) / 60:.1f} min)', flush=True)
    return best


def main():
    args = parse_args()
    try:
        X, Y, V = load_data(args.train_on, args.limit_train, args.limit_valid)
        run(args, X, Y, V)
    except FileExistsError as e:
        raise SystemExit(str(e))


if __name__ == '__main__':
    main()

"""Addestramento di insegnanti e studenti (LSTM + testa HL-Gauss) su GPU, con truncated BPTT su sequenze intere.

Ogni batch prende B sequenze complete e le percorre a blocchi di `--chunk` righe, passando lo stato da un blocco al
successivo (staccato dal grafo): è lo stesso regime dell'inferenza, che parte da stato zero alla riga 0.

Loss di ogni blocco (righe dopo il warm-up), tutte pesate per riga con (p + 0.02)^alpha, p = probabilità che la riga sia
valutata (mask_model.py, --score-weights alpha):
  1. entropia incrociata HL-Gauss: bersaglio = gaussiana N(y, sigma) integrata su 81 bin in [-2, 2], sigma = 1.5 bin;
  2. compito ausiliario yfut (--aux-yfut-w): correlazione della testa yfut con le variazioni future vere del target;
  3. solo studenti (--distill): correlazione pesata con la media standardizzata delle predizioni degli insegnanti.
Insegnante con anticipo (--lookahead k): gli ingressi sono spostati di k righe (alla riga t legge x_{t+k}).

Esempi (la ricetta finale completa è in run_all.sh):
    python train.py --name T_la20 --lookahead 20                       # insegnante 256x3 con anticipo, train+valid
    python train.py --name S_la20 --distill T_la20                     # studente distillato
    python train.py --name prova --train-on train --limit-train 256 --epochs 1   # prova veloce, valutazione fuori campione
"""
import argparse
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

import data
from model import LSTMModel, build_model, detach_state, weighted_corr, weighted_pearson_torch

WARMUP = 99                 # righe iniziali di ogni sequenza senza predizione
RUNS = Path(__file__).resolve().parent / 'runs'
SCORE_FLOOR = 0.02          # pesi per riga (p + SCORE_FLOOR)^alpha
CAL_STD = 0.7               # deviazione standard a cui si riportano le predizioni prima del taglio a ±2 della metrica
YFUT_EVENTS, YFUT_ROWS = (1, 2, 4, 8, 16, 32, 64), (20, 100, 500)
YFUT_Q = 2 * (len(YFUT_EVENTS) + len(YFUT_ROWS))


def to_gpu(arrays, dtype, block=256):
    """Copia uno o più array (anche memmap) sulla GPU a blocchi, concatenandoli sul primo asse, senza duplicarli in RAM."""
    arrays = arrays if isinstance(arrays, (list, tuple)) else [arrays]
    out = torch.empty((sum(len(a) for a in arrays), *arrays[0].shape[1:]), dtype=dtype, device='cuda')
    offset = 0
    for a in arrays:
        for i in range(0, len(a), block):
            chunk = torch.from_numpy(np.array(a[i:i + block])).to('cuda', dtype)
            out[offset + i:offset + i + len(chunk)] = chunk
        offset += len(a)
    return out


def load_data(train_on='train+valid', limit_train=0, limit_valid=0):
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
    print(f'dati su GPU: training su {len(X)} seq ({train_on}), valid {len(V["X"])} seq ({time.time() - t:.0f}s)', flush=True)
    return X, Y, V


def yfut_targets(xb, yb, dp=None):
    """Variazioni VERE future del target, per ogni riga t: y dopo il prossimo n-esimo evento di i0 meno y_t (n = 1, 2, 4,
    ..., 64; troncato all'ultimo evento della sequenza) e y_{t+R} - y_t (R = 20, 100, 500); y tagliato a ±3.
    Un evento è una riga in cui cambiano i codici dp0-3 di i0 (colonne 44-47; dp = codici non spostati con --lookahead).
    Restituisce (B, T, YFUT_Q) e la maschera di validità."""
    B, T = xb.shape[:2]
    dp = xb[:, :, 44:48] if dp is None else dp
    ev = torch.zeros(B, T, dtype=torch.bool, device=xb.device)
    ev[:, 1:] = (dp[:, 1:] != dp[:, :-1]).any(-1)
    y = yb.float().clamp(-3, 3)
    ar = torch.arange(T, device=xb.device)
    evpos = torch.where(ev, ar, torch.full_like(ar, T)).sort(dim=1).values
    nev = ev.sum(1)[:, None]
    cs = ev.cumsum(1)
    has_next = cs < nev
    two = lambda v: v[..., None].expand(-1, -1, 2)
    out, ok = [], []
    for n in YFUT_EVENTS:
        k = torch.minimum(cs + n - 1, nev - 1).clamp(min=0)
        pos = evpos.gather(1, k).clamp(max=T - 1)
        out.append(y.gather(1, two(pos)) - y); ok.append(two(has_next))
    for R in YFUT_ROWS:
        end = (ar + R).clamp(max=T - 1)
        out.append(y[:, end] - y); ok.append(((ar + R) < T)[None, :, None].expand(B, -1, 2))
    return torch.cat(out, -1), torch.cat(ok, -1)


def shift_time(X, k, block=1000):
    """In place: X[:, t] = X[:, t + k] (le ultime k righe a zero). Insegnante con anticipo: alla riga t legge x_{t+k}."""
    T = X.shape[1]
    for a in range(0, T - k, block):
        b = min(a + block, T - k)
        X[:, a:b] = X[:, a + k:b + k].clone()
    X[:, T - k:] = 0


def hl_gauss_probs(y, K, ratio, R=2.0):
    """HL-Gauss: probabilità dei K bin (centri equispaziati in [-R, R], larghezza w = 2R/(K-1)) di una gaussiana centrata
    in y con sigma = ratio * w, troncata al supporto [-R - w/2, R + w/2] e rinormalizzata. y: (N,) -> (N, K)."""
    w = 2 * R / (K - 1)
    edges = torch.linspace(-R - w / 2, R + w / 2, K + 1, device=y.device)
    sig = ratio * w
    cdf = torch.special.erf((edges[None] - y[:, None].float()) / (math.sqrt(2) * sig))
    return (cdf[:, 1:] - cdf[:, :-1]) / (cdf[:, -1:] - cdf[:, :1]).clamp_min(1e-12)


@torch.no_grad()
def predict(model, X, batch=512, chunk=2000):
    """Predizioni (N, T, 2) senza calibrazione, sequenze intere con stato."""
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
    """WP sul validation con le predizioni riportate a deviazione standard CAL_STD (per target, sulle righe valutate)
    prima del taglio a ±2: la loss non fissa la scala dell'uscita. I moltiplicatori finiscono nel checkpoint e nella consegna."""
    P = predict(model, V['X'])
    need = torch.zeros(P.shape[:2], dtype=torch.bool, device='cuda')
    need[:, WARMUP:] = True
    m = need & V['scored']
    scale = torch.stack([CAL_STD / P[..., k][m].std() for k in range(2)])
    P = P * scale
    wp_scored, per = global_wp(V['Y'], P, m)
    wp_all, _ = global_wp(V['Y'], P, need)
    return {'wp_scored': wp_scored, 't0': per[0], 't1': per[1], 'wp_all_rows': wp_all,
            'scale_t0': float(scale[0]), 'scale_t1': float(scale[1])}


def load_ckpt(name):
    """Checkpoint di una run: best.pt (addestrata sul solo train) oppure last.pt (train+valid)."""
    d = RUNS / name
    return torch.load(d / ('best.pt' if (d / 'best.pt').exists() else 'last.pt'), map_location='cuda')


def distill_target(names, X):
    """Media delle predizioni degli insegnanti, ciascuna standardizzata (media 0, std 1 sulle righe dopo il warm-up).
    Un insegnante con anticipo k risponde alla riga t+k per la riga t: le sue predizioni vengono riallineate."""
    acc = None
    for name in names:
        ck = load_ckpt(name)
        teacher = build_model(ck['config']).cuda()
        teacher.load_state_dict(ck['state_dict'])
        P = predict(teacher, X)
        k = ck['config'].get('lookahead', 0)
        if k:
            P[:, :-k] = P[:, k:].clone()
            P[:, -k:] = P[:, -k - 1:-k]
            print(f'insegnante {name}: anticipo di {k} righe, predizioni riallineate', flush=True)
        mu, sd = P[:, WARMUP:].mean((0, 1)), P[:, WARMUP:].std((0, 1))
        P = (P - mu) / sd
        acc = P if acc is None else acc + P
        del P, teacher
    out = (acc / float(len(names))).half()
    del acc
    torch.cuda.empty_cache()
    return out


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', required=True, help='nome della run: runs/<nome>/')
    ap.add_argument('--hidden', type=int, default=256)
    ap.add_argument('--layers', type=int, default=3)
    ap.add_argument('--train-on', choices=['train', 'train+valid'], default='train+valid',
                    help='train+valid: modello da consegna, le metriche sul validation sono in-sample e si salva solo last.pt; '
                         'train: valutazione fuori campione, si salva anche best.pt')
    ap.add_argument('--lookahead', type=int, default=0, help='insegnante con anticipo: alla riga t legge x_{t+k}')
    ap.add_argument('--distill', default='', help='insegnanti (nomi di run separati da virgola): studente distillato')
    ap.add_argument('--distill-weight', type=float, default=3.0)
    ap.add_argument('--aux-yfut-w', type=float, default=4.0, help='peso del compito ausiliario yfut')
    ap.add_argument('--score-weights', type=float, default=0.5, help='esponente alpha dei pesi per riga (p + 0.02)^alpha')
    ap.add_argument('--tick-bins', type=int, default=81)
    ap.add_argument('--hl-sigma', type=float, default=1.5, help='sigma di HL-Gauss in larghezze di bin')
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--batch', type=int, default=256, help='sequenze per batch')
    ap.add_argument('--chunk', type=int, default=1000, help='righe per blocco di BPTT')
    ap.add_argument('--lr', type=float, default=2e-3)
    ap.add_argument('--lr-final', type=float, default=0.02, help='lr finale come frazione del massimo (coseno)')
    ap.add_argument('--grad-clip', type=float, default=1.0)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--limit-train', type=int, default=0, help='solo le prime N sequenze di train (prove)')
    ap.add_argument('--limit-valid', type=int, default=0, help='solo le prime N sequenze di validation (prove)')
    ap.add_argument('--force', action='store_true', help='sovrascrive una run esistente')
    args = ap.parse_args(argv)
    assert args.chunk > WARMUP and not (args.lookahead and args.distill), 'un insegnante con anticipo non si distilla'
    return args


def run(args, X, Y, V):
    out_dir = RUNS / args.name
    if (out_dir / 'log.jsonl').exists() and not args.force:
        raise SystemExit(f'{out_dir} esiste già: scegli un altro --name (o --force per sovrascrivere)')
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'log.jsonl').unlink(missing_ok=True)
    with_valid = args.train_on == 'train+valid'
    n_valid_in_train = len(V['X']) if with_valid else 0

    DP = None
    if args.lookahead:      # i bersagli yfut si calcolano dai codici dp NON spostati
        DP = X[:, :, 44:48].clone()
        for T_ in ([X] if with_valid else [X, V['X']]):      # con train+valid V['X'] è una vista su X
            shift_time(T_, args.lookahead)
        print(f'[{args.name}] anticipo di {args.lookahead} righe: ingressi spostati', flush=True)

    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # pesi per riga dalla probabilità di essere valutata (mask_model.py), media 1 sulle righe da predire
    n_tr_only = len(X) - n_valid_in_train
    ps = np.load(data.CACHE / 'pscore_bi_train.npy', mmap_mode='r')[:n_tr_only]
    if n_valid_in_train:
        ps = np.concatenate([ps, np.load(data.CACHE / 'pscore_bi_valid.npy', mmap_mode='r')[:n_valid_in_train]])
    g = (torch.from_numpy(np.array(ps)).cuda().half().float() + SCORE_FLOOR) ** args.score_weights
    row_w = (g / g[:, WARMUP:].mean()).half()
    del g

    model = LSTMModel(n_features=X.shape[2], hidden=args.hidden, layers=args.layers, tick_bins=args.tick_bins,
                      aux_yfut=YFUT_Q if args.aux_yfut_w > 0 else 0, lookahead=args.lookahead).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    seq_len, n_train = X.shape[1], len(X)
    steps_per_epoch = math.ceil(n_train / args.batch) * math.ceil(seq_len / args.chunk)
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

    distill_T = None
    if args.distill:
        names = args.distill.split(',')
        distill_T = distill_target(names, X)
        print(f'[{args.name}] distillazione da {len(names)} insegnanti', flush=True)

    if with_valid:
        print('ATTENZIONE: il validation è nel training, le sue metriche sono in-sample. Si salva solo last.pt.', flush=True)
    K, best, t_run = args.tick_bins, -1.0, time.time()
    bounds = [0, *range(args.chunk, seq_len, args.chunk), seq_len]
    for epoch in range(1, args.epochs + 1):
        t, losses = time.time(), []
        perm = torch.randperm(n_train, device='cuda')
        for b in range(0, n_train, args.batch):
            idx = perm[b:b + args.batch]
            xb, yb = X[idx], Y[idx]
            if args.aux_yfut_w > 0:
                yf_t, yf_ok = yfut_targets(xb, yb, None if DP is None else DP[idx])
            state = None
            for s, e in zip(bounds[:-1], bounds[1:]):
                pred, state = model(xb[:, s:e].float(), state)
                state = detach_state(state)
                start = max(WARMUP - s, 0)                       # niente loss sulle righe di warm-up
                ew = row_w[idx][:, s + start:e].reshape(-1).float()
                yy = yb[:, s + start:e].reshape(-1, 2).clamp(-2, 2)
                # 1. HL-Gauss
                lg = model.last_logits[:, start:].reshape(-1, 2, K)
                tgt = hl_gauss_probs(yy.reshape(-1), K, args.hl_sigma)
                ce = F.cross_entropy(lg.reshape(-1, K).float(), tgt, reduction='none').view(-1, 2).mean(1)
                loss = (ce * ew).sum() / ew.sum()
                # 2. variazioni future vere (uscite senza righe valide nel blocco, orizzonti lunghi a fine sequenza: saltate)
                if args.aux_yfut_w > 0:
                    pf = model.last_yfut[:, start:]
                    tf, okf = yf_t[:, s + start:e], yf_ok[:, s + start:e]
                    has = okf.flatten(0, -2).any(0).tolist()
                    wf = okf.float().reshape(-1, YFUT_Q)
                    cors = [weighted_corr(pf[..., q].reshape(-1), tf[..., q].reshape(-1), wf[:, q]) for q in range(YFUT_Q) if has[q]]
                    loss = loss - args.aux_yfut_w * torch.stack(cors).mean()
                # 3. distillazione: correlazione pesata come la metrica (|y| per i pesi delle righe)
                if distill_T is not None:
                    tz = distill_T[idx][:, s + start:e].reshape(-1, 2).float()
                    ww = yy.abs() * ew[:, None]
                    pr = pred[:, start:].reshape(-1, 2)
                    loss = loss - args.distill_weight * 0.5 * (weighted_corr(pr[:, 0], tz[:, 0], ww[:, 0]) +
                                                               weighted_corr(pr[:, 1], tz[:, 1], ww[:, 1]))
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                opt.step()
                sched.step()
                losses.append(loss.item())
        metrics = evaluate(model, V)
        log = {'epoch': epoch, 'train_loss': float(np.mean(losses)), **metrics, 'lr': sched.get_last_lr()[0],
               'sec': round(time.time() - t)}
        if with_valid:
            log['valid_in_sample'] = True
        print(json.dumps(log), flush=True)
        with open(out_dir / 'log.jsonl', 'a') as f:
            f.write(json.dumps(log) + '\n')
        ck = {'state_dict': model.state_dict(), 'config': model.config, 'epoch': epoch,
              'scale_t0': metrics['scale_t0'], 'scale_t1': metrics['scale_t1']}
        if not with_valid and metrics['wp_scored'] > best:
            best = metrics['wp_scored']
            torch.save(ck | {'wp_scored': best}, out_dir / 'best.pt')
    torch.save(ck, out_dir / 'last.pt')
    print(f'[{args.name}] fine ({(time.time() - t_run) / 60:.1f} min) -> {out_dir / "last.pt"}' +
          ('' if with_valid else f'; migliore wp_scored {best:.4f} -> best.pt'), flush=True)


def main():
    args = parse_args()
    X, Y, V = load_data(args.train_on, args.limit_train, args.limit_valid)
    run(args, X, Y, V)


if __name__ == '__main__':
    main()

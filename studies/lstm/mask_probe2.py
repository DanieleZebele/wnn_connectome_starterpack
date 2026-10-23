"""La maschera dipende anche dai target? AUC sul hold-out di un classificatore che vede (a) solo gli input, (b) input + target
della riga (causale), (c) input + target con contesto bidirezionale in finestre da 1000 righe. Stessi split di mask_model.py."""
import sys
import numpy as np, torch
from torch import nn
import data
from mask_model import auc, WARMUP

va = data.load('valid'); sp = np.load('/dev/shm/wnn_cache/mask_split.npz'); fit, hold = sp['fit'], sp['hold']
mode = sys.argv[1]            # x | xy | xy_bi
use_y, bi = mode != 'x', mode.endswith('bi')
in_dim = 112 + (2 if use_y else 0)
torch.manual_seed(0); rng = np.random.default_rng(0)
lstm = nn.LSTM(in_dim, 96, 2, batch_first=True, bidirectional=bi).cuda(); head = nn.Linear(96 * (2 if bi else 1), 1).cuda()
params = list(lstm.parameters()) + list(head.parameters()); opt = torch.optim.AdamW(params, lr=2e-3)
EP, W = 8, 1000
steps = EP * int(np.ceil(len(fit) / 32)) * 20
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda t: 0.02 + 0.98 * 0.5 * (1 + np.cos(np.pi * min(t / steps, 1))))

def get(ii):
    x = torch.from_numpy(np.array(va['X'][ii])).cuda().float()
    if use_y:
        x = torch.cat([x, torch.from_numpy(np.array(va['Y'][ii])).cuda().clamp(-2, 2)], -1)
    return x, torch.from_numpy(np.array(va['scored'][ii])).cuda().float()

def forward(x, state):
    o, state = lstm(x, state)
    return head(o).squeeze(-1), state

for ep in range(1, EP + 1):
    for b in range(0, len(fit), 32):
        x, y = get(fit[rng.permutation(len(fit))][b:b + 32]); state = None
        for s in range(0, x.shape[1], W):
            logit, state = forward(x[:, s:s + W], None if bi else state)
            if not bi: state = tuple(t.detach() for t in state)
            lo = max(WARMUP - s, 0)
            if lo >= logit.shape[1]: continue
            loss = nn.functional.binary_cross_entropy_with_logits(logit[:, lo:], y[:, s + lo:s + W])
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sched.step()
    with torch.no_grad():
        ss, yy = [], []
        for b in range(0, len(hold), 32):
            x, y = get(hold[b:b + 32]); state = None; ls = []
            for s in range(0, x.shape[1], W):
                logit, state = forward(x[:, s:s + W], None if bi else state); ls.append(logit)
            ss.append(torch.cat(ls, 1)[:, WARMUP::5].flatten()); yy.append(y[:, WARMUP::5].flatten())
        print(f'[{mode}] epoca {ep}: hold-out AUC {auc(torch.cat(ss), torch.cat(yy)):.4f}', flush=True)

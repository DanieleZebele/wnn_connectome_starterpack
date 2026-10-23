"""Quanto è prevedibile la maschera is_scored dagli input? LSTM piccola addestrata sul validation (80%) e valutata sul 20%."""
import numpy as np, torch
from torch import nn
import data

va = data.load('valid')
N = len(va['X']); rng = np.random.default_rng(0)
perm = rng.permutation(N); tr_idx, te_idx = np.sort(perm[:int(.8 * N)]), np.sort(perm[int(.8 * N):])
torch.manual_seed(0)
lstm = nn.LSTM(112, 64, 1, batch_first=True).cuda(); head = nn.Linear(64, 1).cuda()
opt = torch.optim.AdamW(list(lstm.parameters()) + list(head.parameters()), lr=2e-3)

def run(idx, train):
    outs, ys = [], []
    for b in range(0, len(idx), 32):
        ii = idx[b:b + 32]
        x = torch.from_numpy(np.array(va['X'][ii])).cuda().float()[:, 99:]
        y = torch.from_numpy(np.array(va['scored'][ii])).cuda().float()[:, 99:]
        h = None; ls = []
        for s in range(0, x.shape[1], 1000):
            o, h = lstm(x[:, s:s + 1000], h); h = tuple(t.detach() for t in h) if train else h
            logit = head(o).squeeze(-1)
            if train:
                loss = nn.functional.binary_cross_entropy_with_logits(logit, y[:, s:s + 1000])
                opt.zero_grad(); loss.backward(); opt.step()
            else:
                ls.append(logit.detach())
        if not train:
            outs.append(torch.cat(ls, 1)[:, ::7].flatten()); ys.append(y[:, ::7].flatten())
    return (torch.cat(outs), torch.cat(ys)) if not train else None

def auc(score, y):
    order = score.argsort(); r = torch.empty_like(score); r[order] = torch.arange(len(score), device=score.device, dtype=score.dtype)
    pos = y > .5; n1, n0 = pos.sum(), (~pos).sum()
    return float((r[pos].sum() - n1 * (n1 - 1) / 2) / (n1 * n0))

for ep in range(1, 7):
    run(tr_idx[rng.permutation(len(tr_idx))], True)
    s, y = run(te_idx, False)
    print(f'epoca {ep}: AUC sul 20% tenuto da parte = {auc(s, y):.4f}  (0.5 = casuale)', flush=True)
p = torch.sigmoid(s); print('P(scored) media %.3f, quota di righe nel top-10%% di probabilità che sono davvero scored: %.3f'%(y.mean(), y[p >= p.quantile(.9)].mean()))

"""Tabella riassuntiva di tutte le run in runs/ (ordinata per miglior wp_scored).

    python summarize.py            # tabella a terminale
    python summarize.py --md       # in markdown
    python summarize.py s1_ s2_    # solo le run il cui nome inizia così
"""
import json
from pathlib import Path
import sys

RUNS = Path(__file__).resolve().parent / 'runs'
KEYS = ['hidden', 'layers', 'cell', 'proj', 'dropout', 'input_dropout', 'input_noise', 'loss', 'epochs', 'lr',
        'batch', 'chunk', 'weight_decay', 'ema', 'seed']
DEFAULTS = {'cell': 'lstm', 'proj': 0, 'dropout': 0.0, 'input_dropout': 0.0, 'input_noise': 0.0, 'loss': 'pearson',
            'weight_decay': 0.0, 'ema': 0.0, 'chunk': 1000, 'seed': 0}


def collect(prefixes=()):
    rows = []
    for d in sorted(RUNS.glob('*')):
        if prefixes and not d.name.startswith(tuple(prefixes)):
            continue
        try:
            cfg = json.loads((d / 'config.json').read_text())
            log = [json.loads(l) for l in (d / 'log.jsonl').read_text().splitlines()]
        except Exception:
            continue
        if not log:
            continue
        # i log possono contenere più run con lo stesso nome (vecchie versioni): si prende l'ultima
        starts = [i for i, r in enumerate(log) if r['epoch'] == 1]
        log = log[starts[-1]:]
        best = max(log, key=lambda r: r['wp_scored'])
        diff = [f'{k}={cfg.get(k, DEFAULTS.get(k))}' for k in KEYS
                if cfg.get(k, DEFAULTS.get(k)) != DEFAULTS.get(k) or k in ('hidden', 'layers', 'epochs', 'lr')]
        rows.append(dict(name=d.name, ep=len(log), of=cfg['epochs'], best=best['wp_scored'], at=best['epoch'],
                         last=log[-1]['wp_scored'], all=best['wp_all_rows'], t0=best['t0'], t1=best['t1'],
                         sec=log[-1]['sec'], params=cfg.get('n_params'), cfg=' '.join(diff),
                         tv=cfg.get('train_on') == 'train+valid'))
    return sorted(rows, key=lambda r: -r['best'])


if __name__ == '__main__':
    md = '--md' in sys.argv
    rows = collect([a for a in sys.argv[1:] if not a.startswith('--')])
    if md:
        print('| run | wp_scored (best@ep) | ultima ep | wp_all_rows | t0 | t1 | s/ep | parametri | config |')
        print('|---|---:|---:|---:|---:|---:|---:|---:|---|')
    for r in rows:
        done = f'{r["ep"]}/{r["of"]}' if r['ep'] < r['of'] else ''
        if md:
            print(f'| {r["name"]} | **{r["best"]:.4f}** @{r["at"]} | {r["last"]:.4f} | {r["all"]:.4f} | {r["t0"]:.4f} | '
                  f'{r["t1"]:.4f} | {r["sec"]} | {r["params"] or ""} | {r["cfg"]} {done} |')
        else:
            print(f'{r["name"]:30s} best {r["best"]:.4f}@{r["at"]:<3d} last {r["last"]:.4f} all {r["all"]:.4f} '
                  f't0 {r["t0"]:.4f} t1 {r["t1"]:.4f} {r["sec"]:4d}s/ep | {r["cfg"]} {done}'
                  f'{" [train+valid]" if r["tv"] else ""}')

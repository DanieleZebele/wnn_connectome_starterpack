"""Aggiunge esperimenti a queue.txt partendo dalla configurazione di riferimento.

    python queue_add.py nome --lr 0.001 --batch 128
Le opzioni date dopo il nome sovrascrivono quelle di riferimento.
"""
from pathlib import Path
import shlex
import sys

BASE = ('--hidden 128 --layers 2 --dropout 0 --epochs 10 --lr 0.002 --batch 256 --chunk 1000 '
        '--loss pearson --weight-decay 0 --seed 0')
QUEUE = Path(__file__).resolve().parent / 'queue.txt'

if __name__ == '__main__':
    name, extra = sys.argv[1], sys.argv[2:]
    existing = {l.split()[0] for l in QUEUE.read_text().splitlines() if l.strip() and not l.startswith('#')} \
        if QUEUE.exists() else set()
    if name in existing:
        raise SystemExit(f'{name} è già in coda')
    with open(QUEUE, 'a') as f:
        f.write(f'{name} {BASE} {shlex.join(extra)}\n')
    print('aggiunto:', name, *extra)

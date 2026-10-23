"""Esegue in sequenza gli esperimenti elencati in queue.txt, tenendo i dati sulla GPU tra una run e l'altra.

    nohup python sweep.py > sweep.log 2>&1 &

queue.txt: una riga per esperimento, "<nome> <opzioni di train.py>". Il file viene riletto dopo ogni run,
quindi si possono aggiungere righe mentre il processo gira (vedi queue_add.py). Una run è finita se esiste
runs/<nome>/last.pt; una run interrotta viene rifatta da capo.

File di controllo (da creare nella cartella lstm/):
    STOP    -> termina dopo la run in corso
    RELOAD  -> ricarica train.py/model.py prima della prossima run (per usare modifiche al codice)
Si ferma da solo dopo IDLE_LIMIT secondi con la coda vuota.
"""
import importlib
from pathlib import Path
import shlex
import time
import traceback

import torch

import model as model_mod
import train

HERE = Path(__file__).resolve().parent
QUEUE, STOP, RELOAD = HERE / 'queue.txt', HERE / 'STOP', HERE / 'RELOAD'
IDLE_LIMIT = 45 * 60


def next_job(skip):
    if not QUEUE.exists():
        return None
    for line in QUEUE.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        name, *argv = shlex.split(line)
        if name in skip or (train.RUNS / name / 'last.pt').exists():
            continue
        return name, argv
    return None


def main():
    global train
    data, key, failed, idle = None, None, set(), 0
    while not STOP.exists():
        if RELOAD.exists():
            try:
                importlib.reload(model_mod)
                train = importlib.reload(train)
                print('codice ricaricato', flush=True)
            except Exception:
                traceback.print_exc()
            RELOAD.unlink()
        job = next_job(failed)
        if job is None:
            time.sleep(20)
            idle += 20
            if idle > IDLE_LIMIT:
                print('coda vuota da troppo tempo: esco', flush=True)
                break
            continue
        idle = 0
        name, argv = job
        print(f'\n=== {time.strftime("%H:%M:%S")} {name} {" ".join(argv)}', flush=True)
        try:
            args = train.parse_args(['--name', name, *argv])
            started = (train.RUNS / name / 'log.jsonl').exists()
            args.force = args.force or started        # run interrotta in precedenza: si riparte
            if (args.train_on, args.limit_train, args.limit_valid) != key:
                data = None
                torch.cuda.empty_cache()
                data = train.load_data(args.train_on, args.limit_train, args.limit_valid)
                key = (args.train_on, args.limit_train, args.limit_valid)
            train.run(args, *data)
        except Exception:
            traceback.print_exc()
            failed.add(name)
        torch.cuda.empty_cache()
    print('sweep terminato', flush=True)


if __name__ == '__main__':
    main()

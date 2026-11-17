#!/bin/bash
# Pipeline finale completa, dai dati allo ZIP da consegnare (~2.5 ore su una GPU da 96 GB).
# Uso:  ./run_all.sh [PREFISSO]      (predefinito "rerun": le run vanno in runs/<PREFISSO>_*, senza toccare quelle esistenti)
# Le run consegnate (s1372, s1180-s1184, s1373, s1191) sono già in runs/: per la sola consegna basta l'ultimo comando con i loro nomi.
set -euo pipefail
cd "$(dirname "$0")"
P=${1:-rerun}
PY=${PY:-python}

$PY data.py                                  # cache dei dati in /dev/shm (se manca)
$PY mask_model.py                            # pesi per riga pscore_bi dalla rete salvata (--train per riaddestrarla)

# insegnanti, tutti LSTM HL-Gauss + yfut x4, train+valid, 10 epoche
$PY train.py --name ${P}_T_la20 --lookahead 20                         # 256x3 con anticipo di 20 righe   (= s1372)
$PY train.py --name ${P}_T3_256x3_s0 --seed 0                          # 256x3 causali                    (= s1180-s1182)
$PY train.py --name ${P}_T3_256x3_s1 --seed 1
$PY train.py --name ${P}_T3_256x3_s2 --seed 2
$PY train.py --name ${P}_T3_256x2_s0 --layers 2 --seed 0               # 256x2 causali                    (= s1183, s1184)
$PY train.py --name ${P}_T3_256x2_s1 --layers 2 --seed 1

# studenti distillati (peso 3)
$PY train.py --name ${P}_S_la20 --distill ${P}_T_la20                  # 256x3                            (= s1373)
$PY train.py --name ${P}_S_T3 --layers 2 --seed 1 \
    --distill ${P}_T3_256x3_s0,${P}_T3_256x3_s1,${P}_T3_256x3_s2,${P}_T3_256x2_s0,${P}_T3_256x2_s1   # 256x2 (= s1191)

# consegna: media dei due studenti nel kernel C int8 -> runs/${P}_ens.zip
$PY export_c.py runs/${P}_ens ${P}_S_la20 ${P}_S_T3

# Soluzione finale: studenti LSTM distillati, kernel C int8

Miglior punteggio sul test: **0.66704** (36 minuti) con la media di due LSTM, `s1373` (256×3) e `s1191` (256×2), eseguite riga per riga
da un kernel C int8. ZIP da consegnare: [runs/ens_c2_tv_s1373_s1191.zip](runs/ens_c2_tv_s1373_s1191.zip).
Tutti gli esperimenti che hanno portato qui, compresi quelli scartati, sono riassunti in [STUDIES.md](STUDIES.md).

## La ricetta

1. **Pesi per riga.** Una LSTM bidirezionale stima, dal solo ingresso, la probabilità p che una riga sia valutata (`is_scored` esiste
   solo nel validation, AUC 0.958). La loss di ogni modello pesa la riga per (p + 0.02)^0.5.
2. **Modello.** LSTM con testa HL-Gauss: per ciascun target una distribuzione su 81 bin in [−2, 2], addestrata con l'entropia incrociata
   verso una gaussiana centrata sul target (σ = 1.5 bin); l'uscita è E[|y|·y] / E[|y|]. Una testa ausiliaria, usata solo in training,
   prevede le variazioni vere future del target (dopo 1…64 eventi di i0 e dopo 20/100/500 righe), con peso 4.
3. **Insegnanti** (train+valid, 10 epoche):
   - `s1372`: 256×3 con **anticipo di 20 righe** (alla riga t legge x_{t+20});
   - `s1180`–`s1182`: 256×3 causali (semi 0–2); `s1183`, `s1184`: 256×2 causali (semi 0–1).
4. **Studenti** (train+valid, 10 epoche, distillazione per correlazione con la media standardizzata degli insegnanti, peso 3):
   - `s1373`: 256×3 dall'insegnante con anticipo;
   - `s1191`: 256×2 dai 5 insegnanti causali.
5. **Consegna.** Pesi int8 per canale, kernel C con istruzioni AVX-512 VNNI (con ripiego AVX2 o C semplice), media dei due studenti
   riportati a deviazione standard 0.7. Una protezione del tempo toglie il secondo modello se la proiezione supera 55 minuti.

## File

| File | Cosa fa |
|---|---|
| `data.py` | cache dei parquet in `/dev/shm/wnn_cache` (feature float16, ~58 GB di RAM) |
| `mask_model.py` | rete della maschera: scrive `pscore_bi_{train,valid}.npy` nella cache |
| `model.py` | LSTM + testa HL-Gauss + testa yfut |
| `train.py` | addestramento di insegnanti (`--lookahead`) e studenti (`--distill`); i valori predefiniti sono quelli della ricetta |
| `export_c.py` | quantizza uno o più checkpoint, compila il kernel, verifica contro PyTorch e crea lo ZIP |
| `submission/` | `solution.py` e `lstm_int8.c` che finiscono nello ZIP |
| `run_all.sh` | l'intera pipeline nell'ordine |
| `runs/` | `mask_net/` (rete della maschera), le 8 run della ricetta (`config.json`, `log.jsonl`, `last.pt`) e lo ZIP |

## Comandi

Dalla cartella `studies/lstm/`, con l'ambiente del progetto (`../../.venv/bin/python`; serve anche `gcc`):

```bash
python data.py                       # una volta per avvio della macchina (qualche minuto)
python mask_model.py                 # pscore_bi dalla rete salvata (~2 min); --train la riaddestra (~1 ora)
python export_c.py runs/ens_c2_tv_s1373_s1191 s1373_tv_stu_la20_l256x3_seed0 s1191_tv_stu_hl15_l256x2_seed1
./run_all.sh nuovo                   # tutto da capo con nomi nuovi: runs/nuovo_*, ~2.5 ore di GPU
```

Un'epoca su train+valid dura circa 2 minuti per un 256×3 e 1.4 per un 256×2 (RTX PRO 6000, dati interamente sulla GPU).
`train.py --train-on train` addestra sul solo train e valuta fuori campione sul validation (salva anche `best.pt`); con `train+valid`
le metriche nel log sono in campione. `--limit-train N --limit-valid M --epochs 1` serve per le prove veloci.

## Verifiche fatte sulla versione finale (17 novembre)

- Le 8 run e la rete della maschera, convertite nel formato di questo codice, danno predizioni identiche ai checkpoint originali.
- `train.py` riproduce bit per bit loss e WP del codice sperimentale (insegnante con anticipo, studente da due insegnanti, solo train).
- `mask_model.py` rigenera `pscore_bi` identico all'originale.
- `export_c.py` ricostruisce gli stessi pesi dello ZIP da 0.66704, e le uscite coincidono su 23 604 righe con quelle dello ZIP originale
  (che sul test ha girato con questi due modelli: la protezione del tempo aveva tolto il terzo).
- In locale lo ZIP fa 90 µs/riga: questo Xeon ha 2 MB di L2 e i pesi della coppia sono 2.4 MB. Sulla CPU di gara (Zen4) il tempo è
  circa 7.6 + 18.6 µs per MB per riga, ~36 minuti.

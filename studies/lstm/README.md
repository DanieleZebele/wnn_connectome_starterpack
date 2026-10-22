# LSTM per t0 / t1

Ambiente: `.venv` nella cartella principale del progetto (`source .venv/bin/activate`).

| File | Cosa fa |
|---|---|
| `data.py` | Copia train/valid in `/dev/shm/wnn_cache` (RAM) come `.npy`: feature float16, target, maschera `is_scored`. Si ricrea da solo se manca (es. dopo un riavvio). |
| `model.py` | `LSTMModel` (LSTM + testa lineare a 2 uscite), wrapper per l'export, Weighted Pearson in PyTorch. |
| `train.py` | Addestramento su GPU. Ogni batch = sequenze intere percorse a blocchi (`--chunk`) passando lo stato, come in inferenza. Valuta su tutto il validation a ogni epoca. |
| `export.py` | Checkpoint → `model.onnx` a step singolo, verifica contro PyTorch, misura µs/riga su 1 thread, crea `submission.zip`. |
| `solution.py` | Il `solution.py` della submission (copiato nello ZIP). |
| `runs/<nome>/` | `config.json`, `log.jsonl` (una riga per epoca), `best.pt`, `last.pt`, `submission/`. |

## Comandi

```bash
cd studies/lstm
python train.py --name lstm_h128_l2 --epochs 10            # addestra
python export.py runs/lstm_h128_l2/best.pt                 # ONNX + ZIP
python runs/lstm_h128_l2/submission/solution.py --validation ../../datasets/valid.parquet   # scorer ufficiale (lento)
```

## Iperparametri di `train.py`

| Opzione | Default | Cosa controlla |
|---|---|---|
| `--name` | obbligatorio | Nome dell'esperimento: i risultati vanno in `runs/<name>/`. Non riusarlo (il log si accoda e `best.pt` viene sovrascritto). |
| `--hidden` | `128` | Dimensione dello stato LSTM. Più grande = più capacità, ma più lento in inferenza (controllare i µs/riga con `export.py`). |
| `--layers` | `2` | Numero di strati LSTM impilati. |
| `--dropout` | `0.0` | Dropout tra gli strati LSTM (ha effetto solo con `--layers` ≥ 2). |
| `--loss` | `pearson` | `pearson`: −Weighted Pearson (la metrica, pesi \|y clippato\|). `wmse`: errore quadratico pesato con \|y clippato\|. Entrambe solo sugli step ≥ 99. |
| `--epochs` | `10` | Passate complete sul train (≈ 20 s ciascuna con i default). |
| `--batch` | `256` | Sequenze intere per batch. |
| `--chunk` | `1000` | Step per blocco di truncated BPTT: un aggiornamento dei pesi ogni blocco, stato passato al blocco successivo. |
| `--lr` | `0.002` | Learning rate massimo (AdamW). Warmup lineare sui primi step, poi coseno fino al 2% del valore. |
| `--weight-decay` | `0.0` | Weight decay di AdamW. |
| `--limit-train` | `0` | Usa solo le prime N sequenze del train (0 = tutte). Per prove veloci. |
| `--seed` | `0` | Seed per inizializzazione e ordine dei batch. |

Fissi nel codice: input = le 112 feature così come sono; gradient clipping a norma 1.0; TF32 attivo; valutazione a ogni epoca su tutto il validation; `best.pt` = epoca con `wp_scored` più alto.

## Metriche nel log

- `wp_scored`: Global WP sulle righe `is_scored` del validation (= metrica ufficiale; baseline GRU 0.6171).
- `t0`, `t1`: WP per singolo target.
- `wp_all_rows`: WP su tutte le righe dallo step 99: controllo contro l'overfitting sulla maschera.

## Vincoli della submission

1 CPU, 60 minuti per ~39,4 M righe → al massimo ~91 µs/riga; ZIP ≤ 20 MB. `export.py` stampa entrambi.

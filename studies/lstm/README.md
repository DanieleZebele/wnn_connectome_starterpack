# RNN (LSTM / GRU) per t0 / t1

Ambiente: `.venv` nella cartella principale del progetto (`source .venv/bin/activate`).

| File | Cosa fa |
|---|---|
| `data.py` | Copia train/valid in `/dev/shm/wnn_cache` (RAM) come `.npy`: feature float16, target, maschera `is_scored`. Si ricrea da solo se manca (es. dopo un riavvio). |
| `model.py` | `LSTMModel` (LSTM o GRU + testa lineare a 2 uscite, con opzioni di regolarizzazione), wrapper per l'export, Weighted Pearson in PyTorch. |
| `features.py` | Feature derivate identiche in training e inferenza: differenze `x[t]-x[t-k]` a più ritardi (`--lags`). Non aiutano (vedi `RESULTS.md`). |
| `regimes.py` | I 4 regimi (strumenti) delle sequenze, riconoscibili dai valori di riposo di `dp`/`dv` nei primi 99 step. |
| `mask_model.py` | Classificatore della maschera `is_scored` (AUC ~0.95): stima la probabilità che ogni riga venga valutata; serve a pesare il training (`--score-weights`). `--bidir` usa anche il contesto futuro (solo per i pesi, non per l'inferenza). |
| `mask_probe.py`, `mask_probe2.py` | Verifiche su quanto è prevedibile la maschera (dagli input, dai target, con contesto bidirezionale). |
| `train.py` | Addestramento su GPU. Ogni batch = sequenze intere percorse a blocchi (`--chunk`) passando lo stato, come in inferenza. Valuta su tutto il validation a ogni epoca. |
| `sweep.py`, `queue_add.py`, `queue.txt` | Coda di esperimenti eseguita da un processo che tiene i dati sulla GPU (vedi sotto). |
| `summarize.py` | Tabella di tutte le run (`python summarize.py [--md] [prefisso]`). |
| `posthoc.py` | Analisi sulle predizioni di modelli già addestrati: smoothing causale, media di più modelli, report per regime. |
| `RESULTS.md` | Risultati e conclusioni degli esperimenti sugli iperparametri. |
| `export.py` | Checkpoint → `model.onnx` a step singolo, verifica end-to-end con il `solution.py` dello ZIP, misura µs/riga su 1 thread, crea lo ZIP. |
| `solution.py` | Il `solution.py` della submission (copiato nello ZIP; legge `config.json` per sapere se servono le differenze e se la cella è LSTM o GRU). |
| `runs/<nome>/` | `config.json`, `log.jsonl` (una riga per epoca), `best.pt`, `last.pt`, `submission/`, `submission.zip`. |

## Comandi

```bash
cd studies/lstm
python train.py --name lstm_h128_l2 --epochs 10            # addestra
python export.py runs/lstm_h128_l2/best.pt                 # ONNX + ZIP
python runs/lstm_h128_l2/submission/solution.py --validation ../../datasets/valid.parquet   # scorer ufficiale (lento, ~25 min)
python summarize.py                                        # confronto di tutte le run
```

`train.py` rifiuta un `--name` già usato (per non sovrascrivere checkpoint e log): usare `--force` solo di proposito.

## Coda di esperimenti

Lanciare un training per volta ricarica ogni volta 58 GB di dati sulla GPU e ne occupa quasi tutta la memoria: due
training insieme non ci stanno. `sweep.py` resta in esecuzione, tiene i dati sulla GPU ed esegue le righe di `queue.txt`.

```bash
nohup python sweep.py > sweep.log 2>&1 &
python queue_add.py nome --lr 0.001 --batch 128    # aggiunge una riga (parte dalla configurazione di riferimento)
touch STOP                                         # termina dopo la run in corso
touch RELOAD                                       # ricarica train.py/model.py prima della prossima run
```

La coda viene riletta dopo ogni run, quindi si può riempire mentre gira. Una run è considerata finita se esiste
`runs/<nome>/last.pt`; una run interrotta riparte da capo. Il processo si ferma da solo dopo 45 minuti a coda vuota.
Configurazione di riferimento di `queue_add.py`: `--hidden 128 --layers 2 --dropout 0 --epochs 10 --lr 0.002 --batch 256 --chunk 1000 --loss pearson --weight-decay 0 --seed 0`.

## Opzioni di `train.py`

| Opzione | Default | Cosa controlla |
|---|---|---|
| `--name` | obbligatorio | Nome dell'esperimento: i risultati vanno in `runs/<name>/`. |
| `--hidden` | `128` | Dimensione dello stato RNN. Più grande = più capacità, ma più lento in inferenza (controllare i µs/riga con `export.py`). |
| `--layers` | `2` | Numero di strati RNN impilati. |
| `--cell` | `lstm` | `lstm` oppure `gru`. |
| `--proj` | `0` | Se > 0: Linear + GELU di questa dimensione davanti alla RNN (mescola le feature prima della memoria). |
| `--lags` | vuoto | Lista di ritardi (es. `1,10,50,200`): aggiunge in ingresso `x[t]-x[t-k]` per ogni ritardo (buffer circolare in `solution.py`). Non aiuta e rallenta molto. `--diff` equivale a `--lags 1`. |
| `--dropout` | `0.0` | Dropout tra gli strati RNN (ha effetto solo con `--layers` ≥ 2). |
| `--input-dropout` | `0.0` | Dropout sulle feature in ingresso. |
| `--input-noise` | `0.0` | Rumore gaussiano (deviazione standard) sulle feature in ingresso. |
| `--ema` | `0.0` | Se > 0 (es. 0.999): media mobile esponenziale dei pesi; metriche e checkpoint usano i pesi mediati (nel log anche `*_raw` per i pesi non mediati). Il decadimento ha un warmup `(1+t)/(10+t)` nei primi `1/(1-ema)` passi. A fine training, con lr basso, coincide con i pesi normali. |
| `--loss` | `pearson` | `pearson`: −Weighted Pearson (la metrica, pesi \|y clippato\|). `wmse`: errore quadratico pesato con \|y clippato\|. `mix`: pearson + `--mix-lambda` × wmse. Solo sugli step ≥ 99. |
| `--mix-lambda` | `0.2` | Peso del wmse nella loss `mix`. |
| `--epochs` | `20` | Passate complete sul train (≈ 20 s ciascuna con hidden 128, 2 strati). |
| `--batch` | `256` | Sequenze intere per batch. |
| `--chunk` | `1000` | Step per blocco di truncated BPTT: un aggiornamento dei pesi ogni blocco, stato passato al blocco successivo. |
| `--random-offset` | spento | A ogni batch il primo blocco ha lunghezza casuale in [1, `chunk`]: i bordi dei blocchi cadono in punti diversi. La sequenza parte comunque dallo step 0. |
| `--lr` | `0.002` | Learning rate massimo (AdamW). Warmup lineare sui primi step, poi coseno. |
| `--lr-final` | `0.02` | Learning rate finale come frazione del massimo. |
| `--weight-decay` | `0.0` | Weight decay di AdamW. |
| `--grad-clip` | `1.0` | Norma massima del gradiente. |
| `--score-weights` | `0` | Se > 0: la loss pesa ogni riga anche per `(p+floor)^alpha`, con p = probabilità stimata che la riga venga valutata (`mask_model.py`, da lanciare prima). **0.5 è il valore migliore**; sopra 1 peggiora le prestazioni su tutte le righe. |
| `--score-file` | `pscore` | Probabilità da usare: `pscore` (classificatore causale) oppure `pscore_bi` (bidirezionale, migliore). |
| `--score-floor` | `0.02` | Valore sommato a p prima dell'elevamento a potenza. |
| `--regime` | vuoto | Lettere dei regimi (A,B,C,D) su cui addestrare e valutare, es. `A`. I modelli specializzati non migliorano. |
| `--regime-weights` | vuoto | Campiona le sequenze con probabilità diverse per regime (`eval`, `sqrt` o `A:1.1,B:0.25,C:1.3,D:2.3`). Effetto piccolo (+0.001). |
| `--forget-bias`, `--skip`, `--head-hidden` | spenti | Bias iniziale della porta di dimenticanza; collegamento diretto input→uscita; testa a due strati. Nessuno aiuta. |
| `--limit-train`, `--limit-valid` | `0` | Usano solo le prime N sequenze di train / validation (0 = tutte). Solo per prove veloci e per usare poca memoria GPU. |
| `--train-on` | `train` | `train+valid`: addestra anche sulle 1.873 sequenze del validation (+18% di dati). Le metriche sul validation diventano in-sample (nel log `valid_in_sample: true`): non si salva `best.pt`, solo `last.pt`. Solo per il modello finale. |
| `--seed` | `0` | Seed per inizializzazione e ordine dei batch. Il rumore tra seed diversi è circa ±0.0005 di `wp_scored`. |
| `--force` | spento | Permette di riusare un `--name` esistente (sovrascrive). |

Fissi nel codice: input = le 112 feature così come sono (più le differenze con `--diff`); TF32 attivo; valutazione a ogni epoca su tutto il validation; `best.pt` = epoca con `wp_scored` più alto.

## Modello finale su train + valid

1. Scegli la configurazione con esperimenti normali (`--train-on train`), guardando `wp_scored` e `wp_all_rows`, **incluso il numero di epoche**.
2. Riaddestra da zero con la stessa configurazione e le epoche scelte, aggiungendo `--train-on train+valid`.
3. Esporta l'ultimo checkpoint (non c'è `best.pt`): `python export.py runs/<nome>/last.pt`.

Le metriche stampate in questa modalità sono sul validation già visto: servono solo a controllare che il training non sia andato storto, non a valutare il modello.

## Metriche nel log

- `wp_scored`: Global WP sulle righe `is_scored` del validation (= metrica ufficiale; baseline GRU 0.6171).
- `t0`, `t1`: WP per singolo target.
- `wp_all_rows`: WP su tutte le righe dallo step 99: controllo contro l'overfitting sulla maschera.
- `wp_scored_clean`: `wp_scored` sulle 373 sequenze di validation che il classificatore della maschera non ha mai visto (solo se `mask_model.py` è stato eseguito): serve a escludere contaminazioni quando si usa `--score-weights`.

## Vincoli della submission

1 CPU, 60 minuti per ~39,4 M righe → al massimo ~91 µs/riga; ZIP ≤ 20 MB. `export.py` stampa entrambi, più la stima con una CPU 1.5× più lenta
(la CPU di scoring è un AMD EPYC 9224 a 2.5 GHz; il baseline GRU gira a ~30 µs/riga sulla nostra macchina).

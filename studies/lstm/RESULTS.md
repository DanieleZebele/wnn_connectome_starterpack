# Esperimenti sugli iperparametri — notte del 22–23 ottobre

Tutti i numeri sono `wp_scored` (metrica ufficiale) sul validation, salvo dove indicato. Riferimento: `lstm_h128_l2`
(hidden 128, 2 strati, 10 epoche, lr 2e-3, loss Pearson) = **0.6804**. La tabella completa si rigenera con
`python summarize.py --md`; ogni run ha `config.json`, `log.jsonl`, `best.pt` e `last.pt` in `runs/<nome>/`.

## Cosa sappiamo con certezza

- **Rumore tra una run e l'altra: circa ±0.0005** (stessa configurazione, seed 0/1/2: 0.6804 / 0.6799 / 0.6795).
  Differenze più piccole di 0.001 non si possono interpretare. Dopo decine di confronti sul solito validation c'è
  anche il rischio di scegliere per caso: contano solo effetti netti, coerenti su `wp_scored` e `wp_all_rows`.
- **Il limite non è la capacità ma la generalizzazione**: hidden 128/256/512 e 1/2/4 strati stanno tutti tra 0.677 e 0.681.
  Con più epoche il training migliora e il validation peggiora (overfitting dall'epoca 5–8).
- **Costo in gara** (sola `session.run` ONNX, 1 thread, nostra CPU): h128/2 strati 32 µs, h128/4 strati 86 µs,
  h256/2 strati **164 µs**, h512/1 strato 215 µs. Oltre circa 1 MB di pesi i tempi esplodono (cache). Limite: ~91 µs/riga
  sul test, e la CPU di scoring (EPYC 9224, 2.5 GHz, 1 MB di L2 per core) potrebbe essere più lenta della nostra.

## Risultato principale: pesare il training per la probabilità che la riga venga valutata

La maschera `is_scored` **si può prevedere dagli input**: un piccolo classificatore (`mask_model.py`) addestrato sul 80% delle
sequenze di validation raggiunge **AUC 0.949** (causale) e **0.958** (bidirezionale, usa anche il contesto futuro: va bene per i
pesi di training, non per l'inferenza) sul 20% mai visto. I target non aggiungono informazione (AUC 0.9408 → 0.9411): la
selezione dipende dallo stato osservabile del mercato, non dal valore del target. A livello fine le righe valutate arrivano a
gruppi di 2–3 e sembrano casuali; ciò che si prevede è il **tasso** di righe valutate, che cambia molto tra sequenze e fasi (da
96 a quasi 12.000 righe per sequenza).

La loss di training pesa ogni riga per `(p + 0.02)^0.5`, con p la probabilità stimata di essere valutata (`--score-weights 0.5`):

| Configurazione (hidden 128, 2 strati, 10 epoche) | wp_scored | wp_all_rows |
|---|---:|---:|
| Riferimento, 3 seed (0.6804 / 0.6799 / 0.6795) | 0.6799 | 0.5083 |
| Pesi dalla probabilità causale, α = 0.5, 3 seed | 0.6833 | 0.5093 |
| **Pesi dalla probabilità bidirezionale, α = 0.5, 3 seed (0.6847 / 0.6844 / 0.6843)** | **0.6845** | 0.5084 |

- Il guadagno è **+0.0046** e molto stabile (le tre run differiscono di 0.0004, contro ±0.0005 del rumore).
- **Verificato sul hold-out pulito** (373 sequenze di validation che il classificatore non ha mai visto): +0.0063 contro +0.0042 sulle altre
  1.500. Non è un effetto della contaminazione.
- **Verificato con lo scorer ufficiale riga per riga** sul modello esportato `s16_bi05`: **0.6846993** (riferimento 0.6804132).
- **Non peggiora le righe non valutate**: `wp_all_rows` resta 0.508–0.509. È importante perché la FAQ avverte che la maschera del test
  è nascosta e può non somigliare a quella del validation. Con α = 2 invece `wp_all_rows` scende a 0.486: non esagerare.
- Scelta di α (probabilità bidirezionale): 0.35 → 0.6827, **0.5 → 0.6845**, 0.6 → 0.6849, 0.75 → 0.6838. Causale: 0.25 → 0.6815,
  0.5 → 0.6833, 1 → 0.6825, 2 → 0.6805. Floor 0.005 → 0.6849, 0.02 → 0.6845, 0.05 → 0.6831.
- Un classificatore più grande (192 unità, 20 epoche) ha AUC 0.9554, peggio di quello da 128 unità (0.9577): non serve.
- Nessun costo in gara: la probabilità serve solo in training; il modello e `solution.py` non cambiano.

**Intorno a questa configurazione tutto è fermo a ~0.684–0.685**: loss `wmse` 0.6849, loss mista 0.6833, 14 epoche 0.6844 (3 seed:
0.6852 / 0.6845 / 0.6834), 20 epoche 0.6847, lr 3e-3 0.6843, pesi per regime in più 0.6841, α = 0.6 con 2 seed 0.6849 / 0.6851.

Aumentare il peso durante il training (`--score-weights-end`): 0 → 1 0.6853 · 0.3 → 1 0.6832 · 0.5 → 1 0.6845 · 1 → 0.5 vedi
`summarize.py s19_`. Nessuna rampa batte il peso costante in modo netto (differenze nel rumore).

### Curva tempo di inferenza / qualità con i pesi migliori (α = 0.5, probabilità bidirezionale, 10 epoche)

| Modello | wp_scored | µs/riga (predict completo) | Stima sul test (nostra CPU / CPU 1.5× più lenta) |
|---|---:|---:|---:|
| h64, 1 strato | 0.6807 | 18.4 | 12 min / 18 min |
| h64, 2 strati | 0.6819 | 25.8 | 17 min / 25 min |
| h96, 2 strati | 0.6842 | 28.7 | 19 min / 28 min |
| **h128, 2 strati** | **0.6845** (3 seed) | 35.1 | 23 min / 35 min |
| h160, 2 strati | 0.6841 | non misurata (più lento) | |

Il modello da 64 unità e un solo strato con i pesi (0.6807) vale quanto il riferimento da 128 unità e 2 strati senza pesi (0.6799), a metà del tempo.

## Tutto il resto: nessun miglioramento oltre il rumore (riferimento 0.6799 ± 0.0005)

| Cosa | Risultato (wp_scored) |
|---|---|
| Learning rate | 3e-4 0.6727 · 1e-3 0.6786 · **2e-3** 0.6799 · 3e-3 (con wmse) 0.6819 · 4e-3 0.6761 |
| Batch | 64 0.6690 · 128 0.6803 · **256** · 512 0.6791 |
| Weight decay | 0.1 0.6793 · 1.0 0.6708 |
| Epoche | 6 0.6774 · 8 0.6792 · **10** · 12 0.6801 · 15 0.6808 · 20 0.6763 · 100 overfitting (picco all'epoca 5) |
| Dropout tra gli strati | 0.1 0.6798 · 0.2 0.6785 · 0.5 e 0.9 molto peggio |
| Chunk BPTT | 100 0.4832 (instabile) · 500 0.6797 · **1000** · 2000 0.6792 · 5000 0.6760; offset casuale 0.6801 |
| Dimensione | h128/1 strato 0.6779 · **h128/2** · h128/3 0.6797 · h128/4 0.6805 · h256/1 0.6770 · h256/2 0.6811 · h512/2 0.6799 |
| Cella | GRU 0.6583 (molto peggio della LSTM) |
| Media dei pesi (EMA) | 0.6804 in tutte le varianti |
| Rumore sugli input | 0.1 0.6796 · 0.3 0.6743 · dropout sugli input 0.1 0.6746 · 0.3 0.6690 |
| Differenze delle feature | rispetto alla riga prima 0.6795 · ritardi 1,10,50,200 0.6747 · 100,500,2000 0.6740 (e rallentano) |
| Testa e inizializzazione | collegamento input→uscita 0.6284 · testa a 2 strati 0.6775 · forget bias 1 0.6796 / 3 0.6771 |
| Proiezione prima della LSTM | 128 0.6805 · **256 0.6816 / 0.6815** (ma 59 µs/riga: troppo lento) |
| Loss | `wmse` 0.6815 / 0.6797 / 0.6807 · mista (0.2) 0.6812 · mista (0.5) 0.6803 |
| lr finale / clip | lr finale 0 0.6811 · 10% 0.6795 · clip 0.5 0.6791 |
| Pesi per regime (campionamento) | `sqrt` 0.6813 / 0.6812 / 0.6805 (media 0.6810) · `eval` 0.6808 |
| Modelli specializzati per regime | A 0.4276 (unico: 0.4255) · B 0.7224 (0.7248) · C 0.7651 (0.7657) · D 0.7955 (0.7922) |
| Più dati | 2.650 seq 0.6703 · 5.300 0.6763 · 7.950 0.6779 · 10.607 0.6799: un raddoppio vale ~+0.004; train+valid (+18%) ≈ +0.001 |

## Idee provate sulle predizioni (nessun training) — tutte scartate

| Idea | Risultato |
|---|---|
| Smoothing causale delle predizioni (EMA) | peggiora: 0.6804 → 0.6777 già con finestra di 2 step |
| Tenere ferma la predizione quando `dp` non cambia | peggiora: 0.6804 → 0.6365 |
| Calibrazione non lineare dell'uscita (stimata su metà validation, valutata sull'altra) | nessun effetto (da −0.0002 a +0.0005) |
| Calibrazione lineare per regime | nessun effetto (da −0.0001 a +0.0006) |
| Media di 7 modelli già addestrati | +0.0034 (0.6838): i modelli sbagliano quasi negli stessi punti. Troppo costosa in gara (ogni modello in più costa ≥ 30 µs) |

## Struttura dei dati scoperta

- **Il 100% dei salti dei target avviene su righe in cui cambia `dp` di i0** (il 32–37% delle righe). Sulle altre righe il target è identico
  a quello della riga prima.
- **Ogni sequenza appartiene a uno di 4 regimi** (4 strumenti), riconoscibili dai valori di riposo di `dp0-3` e `dv0` di i0 nei primi 99
  step, con il 99.6% di accuratezza (kNN addestrato sul train). L'unità dei target (passo minimo non nullo di |t|) è 0.66 (A), 0.47 (B),
  0.24 (C), 0.13 (D). Vedi `regimes.py`.

| Regime | Sequenze valid | Righe valutate | WP del modello unico | Tasso di trade |
|---|---:|---:|---:|---:|
| A | 697 | 1.72 M (49%) | **0.4255** | 7% |
| B | 431 | 0.28 M (8%) | 0.7248 | 15% |
| C | 347 | 0.54 M (15%) | 0.7657 | 32% |
| D | 398 | 0.99 M (28%) | 0.7922 | 13% |

- **Il regime A pesa metà del punteggio ed è molto più difficile degli altri** (poche trade, quindi meno informazione); i modelli
  specializzati non lo migliorano: è un limite dei dati.
- **Train e validation hanno composizioni diverse**: nel train A/B/C/D sono 43/33/12/12% delle sequenze, nel validation 37/23/19/21%.

## Costo di inferenza (sola `session.run` ONNX, 1 thread, nostra CPU)

h128/1 strato 18.5 µs · h64/2 strati 22.9 · **h128/2 strati 31.6** (35 µs con tutto `predict`) · h128/4 strati 86 · h256/2 strati **164** ·
h512/1 strato 215 · h128/2 + proiezione 256: 59. Oltre ~1 MB di pesi i tempi esplodono (cache). Limite ~91 µs/riga sul test; la CPU di scoring
(EPYC 9224, 2.5 GHz, 1 MB di L2 per core) potrebbe essere più lenta della nostra.

## Cosa consegnare

- `runs/s16_bi05/submission.zip`: addestrato sul solo train, **0.6847 verificato con lo scorer ufficiale**. 35 µs/riga, 0.96 MB.
- `runs/s20_tv_bi05_seed0/submission_last.zip` e `runs/s20_tv_bi05_seed1/submission_last.zip`: stessa configurazione addestrata su
  train + validation (+18% di dati, atteso circa +0.001), 10 epoche fisse. **Non verificabili in locale** (il validation è nel training): il
  loro punteggio sul validation è in-sample (0.692).
- Il metodo di training è `train.py --score-weights 0.5 --score-file pscore_bi` (prima `python mask_model.py` e
  `python mask_model.py --bidir`). Vedi il README.

## Cautele

- **La FAQ dice che la maschera del test può essere diversa da quella del validation.** Con α = 0.5 le prestazioni su tutte le righe non calano, ma
  il guadagno di +0.0046 vale soprattutto se il test somiglia al validation.
- Il classificatore della maschera è addestrato sul validation: i pesi dipendono quindi da una parte delle sue etichette. Il confronto sul
  hold-out pulito mostra che non gonfia il risultato.
- Dopo decine di confronti sul solito validation c'è il rischio di scegliere per caso: la scelta finale si appoggia solo su effetti con più seed
  (rumore ±0.0005).

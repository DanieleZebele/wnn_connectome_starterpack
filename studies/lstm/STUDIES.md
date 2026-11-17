# Studi: cosa è stato provato e cosa ne è rimasto (22 ottobre – 16 novembre 2026)

Riassunto di tutti gli esperimenti fatti per la competizione Wunder Fund "Alfa Connectome". Il codice che li ha prodotti è stato
rimosso: in questa cartella resta solo la pipeline finale (vedi [README.md](README.md)). Le opzioni citate tra apici inversi
(`--aux-flow-w`, `--tok-k`, ...) erano opzioni del vecchio `train.py`.

**Come leggere i numeri.**
- *Validation*: WP (Weighted Pearson globale, la metrica) sul validation fornito, con il modello addestrato sul solo train.
- *Test*: punteggio della classifica, per le consegne caricate.
- Il rumore tra semi è circa ±0.0005 sul validation: differenze sotto ~0.001 non sono interpretabili. Su decine di confronti c'è anche
  il rischio di scegliere per caso: contano solo gli effetti visti su più semi.
- Il validation sovrastima il test di circa 0.025–0.03 (il test ha più sequenze del regime A, il più difficile). Sul test arrivano di solito
  i due terzi del guadagno visto sul validation; guadagni sotto ~0.002 sul validation non si vedono sul test.

## 1. Il percorso sul test

| Consegna | Test | Cosa è cambiato |
|---|---|---|
| baseline ufficiale GRU | 0.617 (validation) | predice con una scala 3 volte troppo grande; ricalibrata vale 0.673 sul validation |
| `lstm_h128_l2` | 0.64993 | LSTM 128×2, loss di Pearson, solo train |
| `s16` / `s18_bi05_h96` | 0.65464 / 0.65483 | pesi della loss per la probabilità che la riga sia valutata (**+0.0049**) |
| `s20_tv_bi05` | 0.65720 | addestramento su train+valid |
| `s121_dist_w10` / `s180` (solo train) | 0.65863 / 0.65877 | distillazione da 11 modelli |
| `s140_tv_dist` / `s161_tv_student` | 0.65909 / 0.65991 | distillazione + train+valid; insegnanti riaddestrati anche sul validation (+0.0008) |
| `s250_tv_h256l1` | 0.65971 | studente LSTM 256×1 (nessun guadagno) |
| `s330_tv_gru_h128l2` | 0.66040 | studente GRU 128×2 con uscite riportate a std 0.7 |
| `s711_tv_student_gru_flowTV` | 0.66103 | insegnanti con compito ausiliario sul flusso dei codici |
| `s890` (GRU) / `s891` (LSTM 256×1) | 0.66262 / 0.66271 | compito ausiliario sulle variazioni vere future, yfut (**+0.0016**) |
| `s893` / `s930` | 0.66251 / 0.66247 | insegnanti più grandi, 16 epoche: il +0.0005 del validation non si vede sul test |
| `s1120` int8 (solo train) | 0.66078 | primo modello HL-Gauss 256×2 int8 |
| `s1191` (LSTM 256×2 int8, train+valid) | 0.66536 | HL-Gauss + studente più grande reso possibile da int8 (**+0.0027**) |
| `s1240` | 0.66479 | come s1191 ma da 8 insegnanti con trasformer: più insegnanti non aiutano |
| `s1373` (LSTM 256×3, kernel C) | 0.66618 | studente dall'insegnante con anticipo di 20 righe + kernel C (+0.0008) |
| **`ens_c3_tv_s1373_s1191_s1366`** | **0.66704** | insieme nel kernel C, 36 min. **Ha girato con 2 modelli** (s1373 + s1191): la protezione del tempo ha tolto subito il terzo |
| `ens_c2_tv_s1373_s1366` | 0.66649 | due studenti con anticipo: peggio della coppia con lo studente T3, al contrario del validation |
| `ens_c3_tv_s1373_s1191_s1394` | 0.66703 | terzo membro da insegnante GRU con anticipo, tolto di nuovo dalla protezione |
| `ens_c3r4_tv_s1373_s1191_s1394` | 0.66684 | pesi ricorrenti int4; girato a 2 modelli (protezione che decideva troppo presto) |
| `ens_c4r4i_tv_s1373_s1191_s1394_s1414` | 0.66702 | 4 modelli int4 in 49 min: gli insiemi sono saturi anche sul test |

**Totale: +0.0171** dalla prima LSTM (0.64993) al migliore (0.66704). Il primo posto in classifica era 0.67433.
La versione finale è la coppia **s1373 + s1191**, esattamente il calcolo che ha prodotto 0.66704.

## 2. Le migliorie, una per una (tutte nella pipeline finale)

### 2.1 Scala delle predizioni (calibrazione)
- La metrica taglia le predizioni a ±2 e una loss di correlazione non fissa la scala: GRU e trasformer uscivano con std 2–3, con il 40% delle
  righe tagliate. Le uscite vanno riportate a std 0.7 (scala nel checkpoint, applicata in consegna).
- È la differenza tra 0.617 e 0.673 della baseline ufficiale. Per un mese aveva fatto sembrare GRU e trasformer molto peggiori della LSTM
  (−0.02): ricalibrati valgono quanto la LSTM.
- La curva WP contro scala è piatta tra std 0.2 e 0.7 e cala solo per il taglio (std 1.0 −0.0005, 2.0 −0.017, 3.0 −0.033). Il taglio non aiuta mai.

### 2.2 Pesi per la probabilità che la riga sia valutata (`mask_model.py`)
- `is_scored` si prevede dagli ingressi: AUC 0.949 con una LSTM causale, 0.958 bidirezionale (va bene per pesare il training, non serve in
  gara). I target non aggiungono informazione (AUC 0.9408 → 0.9411): la selezione dipende dallo stato osservabile del mercato.
- La loss pesa ogni riga per (p + 0.02)^0.5: **+0.0046 sul validation (3 semi: 0.6847 / 0.6844 / 0.6843 contro 0.6804 / 0.6799 / 0.6795),
  +0.0049 sul test**. È il guadagno più grande di un singolo cambiamento. Verificato anche sulle 373 sequenze di validation mai viste dal
  classificatore (+0.0063) e con lo scorer ufficiale (0.6847).
- Esponente: 0.35 → 0.6827, **0.5 → 0.6845**, 0.6 → 0.6849, 0.75 → 0.6838; con p² e p³ peggiora e cala anche il WP su tutte le righe
  (0.478 / 0.467 contro 0.508). Floor 0.005 / 0.02 / 0.05: 0.6849 / 0.6845 / 0.6831. Rampe dell'esponente durante il training: nulla.
- Un classificatore più grande (192 unità, 20 epoche) ha AUC più bassa (0.9554 contro 0.9577).

### 2.3 Addestramento su train+valid
- Il 18% di dati in più e più sequenze del regime A. Sul test +0.0024 nel primo confronto (s18 → s20). Riaddestrare anche gli insegnanti
  su train+valid vale altri +0.0008 (s140 → s161). Da allora tutte le consegne usano train+valid, 10 epoche fisse e `last.pt`.
- Curva dei dati (solo train): 2650 sequenze 0.6703 · 5300 0.6763 · 7950 0.6779 · 10607 0.6799: un raddoppio vale ~+0.004.
- Pesare di più il validation nel training peggiora (1× 0.6990 sul hold-out, 3× 0.6948, 10× 0.6918).

### 2.4 Distillazione (insegnanti → studente)
- Lo studente impara a imitare, per correlazione pesata come la metrica, la media standardizzata delle predizioni di più insegnanti
  (calcolata una volta su tutte le righe di training). In gara costa quanto un modello solo.
- Prima versione (LSTM 128×2, 11 insegnanti): **+0.0024** (0.6845 → 0.6869, 3 semi), +0.0027 sul test. Cattura quasi tutto l'insieme degli
  insegnanti (0.6876).
- Peso: 0.5 → 0.6860, 1 → 0.6863, **3 → 0.6869**, 10 → 0.6873, 30 → 0.6871; con HL-Gauss 1.5 → peggio, 5–6 → uguale.
- Gli studenti crescono con la qualità degli insegnanti: insegnanti sul flusso (0.6891) → studenti 0.6889–0.6891; insegnanti yfut (0.6910) →
  0.6896–0.6904; insegnanti grandi (0.6926) → 0.6907. Restano circa 0.001 sotto l'insieme degli insegnanti.
- Più insegnanti dello stesso tipo non servono: 3 insegnanti HL 0.6926, 10 (LSTM + trasformer) 0.6927; sul test s1240 (8 insegnanti) 0.66479
  contro s1191 (5) 0.66536.

### 2.5 Compito ausiliario sulle variazioni vere future (yfut)
- Una testa usata solo in training prevede quanto cambierà il target dopo i prossimi 1, 2, 4, …, 64 eventi di i0 e dopo 20/100/500 righe
  (20 uscite, loss = −correlazione media, peso 4). Evento = riga in cui cambiano i codici dp0-3 di i0.
- **LSTM 128×2: 0.6845 → 0.6884** (peso 0.3 0.6855, 1 0.6870, 2 0.6876, **4 0.6884**, 8 0.6884, 16 0.6877). GRU: 0.6830 → 0.6855.
  Insiemi: 8 modelli 0.6891 → 0.6910. Studenti: GRU 0.6879 → 0.6902. Test **+0.0016** (s711 → s890).
- A differenza del compito sul flusso (2.10) si somma alla distillazione e funziona anche nello studente.
- Varianti senza guadagno: orizzonti lunghi (fino a 256 eventi / 2000 righe) 0.6878; orizzonti contati sui salti veri 0.6875; livelli
  futuri invece delle variazioni 0.6811/0.6789; anche le variazioni passate 0.6876; pesi della maschera nel compito 0.6881; con |y| 0.6878;
  orizzonti più fitti, solo eventi, solo righe: tutti 0.6876–0.6886 (rumore). Salto diviso in parte da codice e residuo: 0.6847 (perde tutto).
- Un modello "puro" con l'uscita come combinazione lineare delle 20 variazioni previste fa 0.6353: il compito aiuta, non sostituisce il target.

### 2.6 Loss HL-Gauss (Farebrother et al. 2024, "Stop Regressing")
- La rete prevede per ogni target una distribuzione su 81 bin in [−2, 2], addestrata con l'entropia incrociata verso una gaussiana centrata
  sul target (σ = 1.5 bin). L'uscita è E[|y|·y] / E[|y|], il valore che massimizza la Pearson pesata con |y| (con E[y]: −0.001).
- Stessa rete: LSTM 128×2 0.6883 → 0.6890, 256×3 0.6895 → 0.6921 (seme fortunato; altri semi 0.6902 / 0.6897), GRU 128×2 0.6859 → 0.6875.
  Soprattutto permette reti più grandi: con la Pearson oltre 128×2 non si guadagnava. Sul test, insieme allo studente più grande: **+0.0027**.
- Senza guadagno: σ variabile durante il training (3 → 1 0.6900), supporto oltre ±2 (±2.25 0.6906, ±2.5 0.6909, ±3 0.6899; sullo studente
  0.6920 contro 0.6926), 161/321 bin, teste per regime con bin sui tick (≤ 0.6886), sola entropia senza yfut con E[y] (0.6660, ottobre).

### 2.7 Insegnante con anticipo (lookahead)
- L'insegnante legge gli ingressi spostati di k righe (alla riga t vede x_{t+k}); le sue predizioni si riallineano e lo studente, causale,
  impara da lui. L'insegnante ha WP 0.739 (k=10), 0.769 (k=20), 0.791 (k=30), 0.826 (k=50).
- Studente 256×3: k=10 0.6936 / 0.6939, **k=20 0.6941** (seme 1: 0.6933), k=30 0.6933. 256×2: k=2 0.6914, k=10 0.6927, k=50 0.6923.
  Un insegnante bidirezionale (WP 0.978) copia il target e lo studente peggiora (0.6889).
- Soprattutto è un membro d'insieme diverso dagli altri: coppie 0.6945–0.6952. Insegnanti GRU (0.766) e trasformer (0.770) con anticipo
  danno studenti alla pari (0.6940 / 0.6938). Insegnante 384×3 con anticipo: studente peggiore (0.6927).
- Sul test: **+0.0008** (s1191 → s1373).

### 2.8 Quantizzazione int8 e kernel C
- **int8 dinamico (onnxruntime)**: LSTM 256×2 da 177 a 47–59 µs per riga, WP invariata (±0.0002). Ha reso consegnabili gli studenti 256×2.
- **Kernel C con istruzioni VNNI**: 3.5 volte più veloce dell'ONNX int8 e senza perdita (0.69267 = fp32). onnxruntime perdeva il tempo nel
  contorno dei nodi, non nel calcolo. Con il kernel C il limite diventa la banda di memoria: sullo Zen4 di gara il tempo è circa
  **7.6 + 18.6 µs per MB di pesi int8 per riga** (previsioni 34.6 e 41 min contro 36 e 41 reali).
- Altre quantizzazioni (studente 256×2): int8 statico QDQ come il dinamico; int4 a blocchi di onnxruntime −0.0009; LSTM riscritta in MatMul 64 µs.
- **int4 a gruppi di 64 sui pesi ricorrenti** (nel kernel C): senza perdita (0.69406 contro 0.69405), −23% di byte (256×3 da 1.47 a 1.13 MB),
  anche sugli ingressi degli strati ≥ 1 −0.0002. Ha permesso 4 modelli in 49 min, ma sul test non ha dato guadagno (0.66702): tolto.

### 2.9 Insiemi in gara
- Media di 2–4 studenti diversi tra loro nello stesso processo C. Validation (media sui semi): singolo 0.6941, coppia 0.6950, trio 0.6953,
  quattro 0.6955, cinque 0.6956: saturano. Sul test **+0.0009** (0.66618 → 0.66704).
- Contano la diversità dell'insegnante (anticipo, GRU, causale) e la media su più semi della ricetta; scegliere la combinazione migliore sul
  validation è ottimistico di +0.0005 e i pesi ottimizzati non servono (±0.0001). Studente + insegnante vale più di due studenti dallo
  stesso insieme.
- Sul test la coppia "anticipo + T3" (s1373 + s1191) ha battuto "anticipo 20 + anticipo 10" (0.66704 contro 0.66649), al contrario del validation.
- **Protezione del tempo** (in `solution.py`): proietta il tempo totale e toglie l'ultimo modello oltre 55 min. Decidere troppo presto
  (sulle prime 60 000 righe, gonfiate dall'avvio del valutatore) ha tolto il terzo modello due volte senza motivo; ora salta le prime
  3 sequenze e misura almeno 300 000 righe.

### 2.10 Compito ausiliario sul flusso dei codici (superato da yfut)
- I codici dp0-3 di i0 sono categorie e il target segue il flusso dei codici futuri. Una testa prevedeva il salto atteso (dalla tabella
  "codice → salto medio" del train) dei prossimi eventi 1–4 e delle somme su 8/16/32/64 eventi, con 3 canali dai codici in ingresso.
- +0.0015 sul validation (0.6860–0.6862 contro 0.6845), anche negli insiemi; +0.0006 sul test. Senza i canali in ingresso solo +0.0005;
  nello studente non si somma alla distillazione. Superato da yfut, che non richiede tabelle.

## 3. Cosa abbiamo capito dei dati

**Regimi.** Ogni sequenza appartiene a uno di 4 strumenti, riconoscibili dai valori di riposo di dp0-3 e dv0 nei primi 99 passi (99.6%).

| Regime | Sequenze valid | Righe valutate | WP (LSTM finale circa) | Unità del target |
|---|---:|---:|---:|---:|
| A | 697 | 49% | ~0.44 | 0.65690 |
| B | 431 | 8% | ~0.73 | 0.46854 |
| C | 347 | 15% | ~0.77 | 0.24132 |
| D | 398 | 28% | ~0.80 | 0.12544 |

- Train e validation hanno composizioni diverse (A/B/C/D 43/33/12/12% contro 37/23/19/21% delle sequenze); il test ha più A del validation.
- A pesa metà del punteggio ed è il più difficile; specialisti per regime, più peso ad A, scala per regime: nessun effetto.
- Per arrivare a 0.70 globale servirebbe A da 0.436 a ~0.50, oppure D da 0.80 a 0.84; B da solo non basta neanche se perfetto.

**I target.**
- Sono multipli interi dell'unità del regime e cambiano **solo** sulle righe in cui cambiano i codici dp0-3 di i0 (32–46% delle righe);
  il 65–81% di questi eventi non li fa saltare. t0 e t1 sono quasi opposti (correlazione −0.87 A, −0.85 B, −0.62 C, −0.50 D).
- **a6 si azzera esattamente quando il target cambia** (riga per riga, in entrambe le direzioni): è il tempo dall'ultimo cambio del target.
  Le colonne a sono trasformate per rango verso una normale (−5.2 = lo zero originale). Tre famiglie: a0, a1, a4, a7 (variazioni legate tra
  loro); a2, a3 (lente e lisce); a5, a6 (cronometri).
- **Forma**: t0 ≈ bid attuale − massimo del bid nei prossimi movimenti del book; t1 ≈ minimo dell'ask futuro − ask attuale (esatto nell'86–87%
  delle righe di A con 4–6 movimenti). t0 + t1 = −spread quando il riferimento futuro è lo stesso: sono il risultato di una vendita al bid e
  di un acquisto all'ask chiusi a un prezzo futuro. La finestra non è un numero fisso di righe, eventi o salti: la regola esatta non è
  identificabile senza prezzi assoluti (in B, C, D la ricostruzione di bid e ask dai codici deriva).
- **Il target dipende dal futuro**: un GRU bidirezionale lo ricostruisce con WP **0.985** (0.988 guardando solo il futuro, 0.979 con i soli
  codici dp futuri). Un GRU causale con D righe di ritardo: D=0 0.662, 10 0.694, 30 0.750, 100 0.842, 300 0.909. Il target è una funzione
  della sequenza degli eventi dp di i0 (la durata delle pause non aggiunge nulla); il passato conta poco.
- La somma s = t0 + t1 non si impara dagli ingressi (corr² ≤ 0.16 anche con una testa dedicata); se fosse nota il WP salirebbe a 0.730 (+0.043).
- Al salto, il codice da solo prevede il salto esatto nel 97.6% (A), 85–87% (B, C), 81–84% (D); con il livello prima del target 89–99%.
  Ma l'errore della metrica non sta nei salti: sta nel livello, cioè nel flusso futuro (A: livelli grandi = 30% delle righe, 66% del peso,
  92% dell'errore; segno sbagliato = 17% del peso, 55% dell'errore).

**Gli ingressi.**
- I codici dp0-3 sono token categorici (vettori identici in tutte le sequenze del regime), non prezzi: A ha 4 codici che fanno l'81% degli
  eventi e spostano ciascuno un solo target di ±1 unità. Le colonne di prezzo sono relative e trasformate per colonna.
- Il 99% dell'influenza sul modello sta nelle ultime 5–12 righe (gradiente rispetto agli ingressi); con memoria limitata a 32–64 righe si
  perde solo ~0.001. Volumi di i0 62–69% dell'influenza, prezzi di i0 28–32%.
- Ablazioni riaddestrando (LSTM 128×2 + yfut): senza tutte le a −0.0027, senza a0/a1/a4/a7 −0.0010, senza a5/a6 −0.0009, senza i1 −0.0079.
- La cache in float16 non perde nulla (float32: +0.0004, nel rumore).

**Conseguenza.** Il tetto sembra la predicibilità del flusso futuro dei codici: architetture e regolarizzazioni arrivano tutte allo stesso
livello, e i guadagni sono venuti da come si costruisce la supervisione (pesi, compiti ausiliari, insegnanti con anticipo, HL-Gauss), non dal modello.

## 4. Cosa non ha funzionato (per non rifarlo)

Riferimenti: LSTM 128×2 di base 0.6799; con i pesi sulla maschera 0.6845; + yfut 0.6883; 256×2 HL + yfut 0.6900–0.6906; studente 256×3 0.6941.

| Area | Provato | Esito |
|---|---|---|
| Iperparametri (base 0.6799) | lr 3e-4…4e-3, batch 64…512, weight decay 0.1/1, 6…100 epoche, dropout 0.1–0.9, chunk BPTT 100…5000 e offset casuale, lr finale, clip | 2e-3, batch 256, 10 epoche, chunk 1000 sono già ottimi; tutto il resto nel rumore o peggio |
| Batch per le prove (256×2 HL) | 128 / 512 (lr 2.8e-3) / 1024 | 512 = stessa WP in metà tempo (utile per le prove); 1024 −0.001 |
| Dimensione | LSTM 128…1024 unità, 1…4 strati; 768×3 e 1024×2 regolarizzate; studenti 320×3, 384×2 | oltre 256×3 nessun guadagno (≤ +0.001 da 1 M a 13 M parametri); profondità meglio della larghezza |
| Architetture | GRU, GRU sotto la LSTM, TCN (anche sotto la LSTM), trasformer a finestra (32–256 righe, fino a 384×6), più scale temporali in parallelo, rete a due velocità, MoE sulla proiezione d'ingresso, MIMO (anche 512×2 con ripetizione degli ingressi), Hydra, LRU e RG-LRU (stile Mamba/Griffin), minGRU, GRU-D, ODE-RNN, Time-Aware LSTM, attenzione sugli ultimi stati | stesso livello o peggio (±0.001); orologio a eventi invece che a righe −0.05; MoE più esperti = peggio; MIMO −0.002 su 256×2 |
| Ingressi | differenze e ritardi delle feature, proiezione o convoluzione davanti alla LSTM, collegamento ingresso → uscita, testa a 2 strati, forget bias, canali dai codici (salto atteso, v2, v3), canali in tick ricostruiti, canali dalle a, ingressi di tempo (+0.0005 sul singolo, nulla sullo studente), RevIN causale, dizionario dei codici come token (memorizza: 0.6890–0.6893) | nulla o peggio |
| Regolarizzazione | dropout, rumore e dropout sugli ingressi, EMA dei pesi, collo di bottiglia variazionale (VIB), entropia ordinale | nulla o peggio |
| Ottimizzatori | Muon (metà epoche per la stessa WP, poi sovradatta anche con weight decay), Lion, model soup (anche con riavvio e rami divergenti: al più +0.0002) | nessun guadagno di qualità |
| Dati | aumenti (scala, offset, magnitude warping, permutazione dei livelli), specchio bid ↔ ask ricostruito dai dati (anche come TTA), dati sintetici riscalando tra regimi (feature non scalabili: scartato senza addestrare), potatura con TracIn (−10% blocchi −0.003), pesi per regime, pesi sulle righe senza cambio o con \|y\| grande, due fasi (prima solo yfut) | nulla o peggio |
| Loss | MSE pesata, loss mista, Pearson con taglio come in gara, correlazione anche su t0 + t1, testa BCE "livello grande", testa su s = t0 + t1 e d = t0 − t1, uscita con tanh (= calibrazione), target in tick del regime, modello a incrementi (0.6414) | nulla o peggio |
| Compiti ausiliari | target passati/futuri a distanza fissa, feature delle righe future, estremi futuri del book, conteggi futuri dei codici, compito esteso a 30 obiettivi | nulla oltre yfut |
| Distillazione | KL sulla distribuzione intera (0.6915 da sola, 0.6921 con la correlazione), gaussiana con media e sd degli insegnanti, pesi di accordo tra insegnanti, più insegnanti misti, born-again e seconda generazione, insieme di studenti come insegnante, distillazione delle feature (TimeKD), correlazione negativa, peso 5–6, studenti dai trasformer | nulla: la correlazione con la media resta il canale giusto per una metrica di correlazione |
| Modelli strutturati | filtro bayesiano sui livelli con la tabella P(salto \| codice, livello) (WP negativa), regola "ask-bid reversal" (0.725 esatto in A), LightGBM sopra la LSTM e residual learning (peggio con più alberi), secondo stadio MLP | peggio della LSTM |
| Post-elaborazione | lisciatura o media nei tratti senza evento, salto forzato dalla tabella, calibrazione non lineare o per regime, scala scelta per riga da una rete, specialisti e scelta del modello per regime, insiemi oltre 3–4 modelli | nulla o peggio |
| Consegna | più modelli in un solo grafo ONNX (2×128: 116 min), due sessioni ONNX | troppo lenti: risolto con il kernel C |

## 5. Lezioni di lavoro

- **Dati in memoria condivisa** (`/dev/shm/wnn_cache`, float16) e **interamente sulla GPU**: un'epoca dura 1–2 minuti invece di decine.
- **Code di esperimenti da file**, modificabili mentre girano, con un guardiano che segnala errori e GPU ferma: decine di run per notte.
  Attenzione: un controllo "nessun `train.py --name` nei processi" trovava la riga di comando della shell stessa e ha fermato una coda 20 minuti.
- **Tempi in gara**: misurare sempre prima di consegnare. Il rapporto piattaforma/locale non è costante (ONNX 1.0–1.5×; kernel C: modello per
  MB di pesi, sezione 2.8). Limite 60 minuti per 39.4 M righe (~91 µs/riga).
- **Verifica di ogni ZIP estratto**, sia con la libreria precompilata sia ricompilata con gcc, contro PyTorch fp32.
- `last.pt` delle run calibrate deve contenere la scala (bug corretto a fine ottobre: prima veniva ricalcolata all'export).
- I test sul test sono pochi (5 al giorno) e rumorosi: ogni caricamento dovrebbe isolare un solo effetto.

## 6. Il codice rimosso

Tutto il codice sperimentale è stato rimosso il 17 novembre. Le prime versioni (fino a "AUC LSTM") sono nella storia git
(commit `2252b95`, `d89b5e7`). Cosa facevano i file principali:

| File | Contenuto |
|---|---|
| `train.py` (vecchio, 1500 righe) | tutte le opzioni citate sopra: architetture, loss, compiti ausiliari, aumenti, MIMO/Hydra/MoE, ottimizzatori |
| `model.py` (vecchio) | LSTM/GRU con tutte le varianti, modello a incrementi, LSTM a più scale, TCN, export ONNX a un passo |
| `window_model.py`, `multirate.py`, `moe.py`, `ssm.py`, `event_rnn.py` | trasformer a finestra, rete a due velocità, esperti, LRU/RG-LRU, architetture a tempo irregolare |
| `future_cnn.py`, `future_events.py` | modelli non causali (quanto il target è ricostruibile dal futuro) |
| `jumpfeat*.py`, `tickfeat.py`, `afeat.py`, `features.py` | canali dai codici, in tick, dalle a; differenze e ritardi |
| `mirror.py`, `mirror_tables/` | specchio bid ↔ ask ricostruito dai dati |
| `regimes.py`, `regime_eval.py` | riconoscimento dei regimi e WP per regime |
| `export.py`, `solution.py` (ONNX), `pair_export.py`, `streaming.py` | consegne ONNX (anche int8 e insiemi in un grafo) prima del kernel C |
| `cnative/` (int4, dizionario dei codici, testa a due strati) | kernel C con le varianti non usate nella consegna finale |
| `posthoc.py`, `ensemble_*.py`, `soup.py`, `memory_profile.py`, `mask_probe*.py` | analisi a posteriori, insiemi, model soup, profilo di influenza, sonde sulla maschera |
| `sweep.py`, `queue*.txt`, `summarize.py`, `runs_util.py` | code di esperimenti e riepiloghi |
| `RESULTS.md`, `REPORT.md`, `CODICI.md`, `TARGET_STRUTTURA.md` | diario degli esperimenti, tabelle dei codici per regime, struttura dei target (riassunti qui) |
| `studies/01_visualize_sequences.ipynb`, `02_prediction_errors.ipynb`, `seqviz.py`, `models.py` | notebook di esplorazione dei dati e degli errori |

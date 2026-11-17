"""Consegna con kernel C (LSTM int8 + testa HL, AVX-512 VNNI se presente): insieme di uno o più modelli.
File nello ZIP: solution.py, weights.npz, lstm_int8.c, lstm_int8.so (precompilata). Se la .so non si carica, la si
ricompila con gcc (build-essential è nell'immagine di gara).
Protezione del tempo: i modelli sono in ordine di priorità; a ogni inizio di sequenza si proietta il tempo totale
(ritmo misurato con k modelli x righe rimanenti) e, se supera BUDGET_S, si passa a k-1 modelli (mai sotto 1). Pesi e scala
finale dell'insieme sono precalcolati per ogni prefisso."""
from pathlib import Path
import ctypes
import os
import subprocess
import tempfile
import time

import numpy as np

HERE = Path(__file__).resolve().parent
TOTAL_ROWS = 39_400_000      # righe del test (60 min ~ 91 µs/riga)
WARM_SEQ = 3                 # sequenze iniziali non misurate (avvio del valutatore)
MIN_ROWS = 300_000           # righe misurate con l'insieme attuale prima di decidere
BUDGET_S = float(os.environ.get('WNN_BUDGET_S', 55 * 60))   # tempo totale proiettato oltre il quale si riduce l'insieme


def _load_lib():
    so = HERE / "lstm_int8.so"
    try:
        return ctypes.CDLL(str(so))
    except OSError:
        pass
    out = Path(tempfile.mkdtemp()) / "lstm_int8.so"
    for march in ("-march=native", "-march=x86-64-v2"):
        r = subprocess.run(["gcc", "-O3", march, "-fPIC", "-shared", "-o", str(out), str(HERE / "lstm_int8.c"), "-lm"],
                           capture_output=True)
        if r.returncode == 0:
            return ctypes.CDLL(str(out))
    raise RuntimeError("impossibile caricare o compilare lstm_int8")


class PredictionModel:
    def __init__(self):
        lib = _load_lib()
        P = ctypes.c_void_p
        lib.lstm_create.restype = P
        lib.lstm_create.argtypes = [ctypes.c_int, ctypes.c_int, P, ctypes.c_int]
        lib.lstm_set_input_scale.argtypes = [P, P]
        lib.lstm_set_layer.argtypes = [P, ctypes.c_int, P, P, P, P, P]
        lib.lstm_set_head_q.argtypes = [P, P, P, P, P, ctypes.c_float, ctypes.c_float]
        lib.ens_step.argtypes = [P, ctypes.c_int, P, P, P, ctypes.c_float, ctypes.c_float]
        lib.ens_step.restype = None
        lib.ens_reset.argtypes = [P, ctypes.c_int]
        z = np.load(HERE / "weights.npz")
        self._keep = []                     # array numpy referenziati finché servono

        def ptr(a, dt):
            a = np.ascontiguousarray(a, dtype=dt)
            self._keep.append(a)
            return a.ctypes.data

        n = int(z["n_models"])
        ms = []
        for i in range(n):
            p = f"m{i}_"
            H = np.asarray(z[p + "H"], dtype=np.int32)
            L, F, K = len(H), int(z[p + "F"]), int(z[p + "K"])
            m = lib.lstm_create(L, F, ptr(H, np.int32), K)
            lib.lstm_set_input_scale(m, ptr(z[p + "in_scale"], np.float32))
            for l in range(L):
                lib.lstm_set_layer(m, l, ptr(z[p + f"wx{l}"], np.int8), ptr(z[p + f"swx{l}"], np.float32),
                                   ptr(z[p + f"wh{l}"], np.int8), ptr(z[p + f"swh{l}"], np.float32), ptr(z[p + f"b{l}"], np.float32))
            sc = z[p + "scale"]
            lib.lstm_set_head_q(m, ptr(z[p + "woq"], np.int8), ptr(z[p + "sco"], np.float32), ptr(z[p + "bo"], np.float32),
                                ptr(z[p + "centers"], np.float32), float(sc[0]), float(sc[1]))
            ms.append(m)
        self._lib = lib
        self._n = ctypes.c_int(n)
        self._ms = (ctypes.c_void_p * n)(*ms)
        self._wk = [np.ascontiguousarray(w, dtype=np.float32) for w in z["ens_w_k"]]     # pesi per ogni prefisso k = 1..n
        self._fsk = np.asarray(z["fin_scale_k"], dtype=np.float32)                          # (n, 2)
        self._x = np.zeros(int(z["m0_F"]), dtype=np.float32)
        self._out = np.zeros(2, dtype=np.float32)
        self._xp, self._op = (ctypes.c_void_p(a.ctypes.data) for a in (self._x, self._out))
        self._step = lib.ens_step
        self.current_seq_ix = None
        self._t0 = None; self._rows = 0                    # totale dall'inizio
        self._tk = None; self._rows_k = 0                  # dal passaggio all'insieme attuale
        self._nseq = 0                                     # sequenze iniziate
        self._set_k(n)

    def _set_k(self, k):
        self._k = k
        self._nk = ctypes.c_int(k)
        self._wp = ctypes.c_void_p(self._wk[k - 1].ctypes.data)
        self._fs0, self._fs1 = ctypes.c_float(float(self._fsk[k - 1][0])), ctypes.c_float(float(self._fsk[k - 1][1]))
        self._tk, self._rows_k = None, 0

    def _check_time(self):
        """A ogni inizio di sequenza. Le prime WARM_SEQ sequenze non si misurano (avvio del valutatore: lettura dei dati,
        riscaldamento); poi il ritmo dell'insieme attuale si misura su almeno MIN_ROWS righe prima di decidere. Proiezione =
        tempo trascorso + ritmo x righe rimanenti; oltre BUDGET_S si toglie l'ultimo modello."""
        now = time.perf_counter()
        if self._t0 is None:
            self._t0 = now
        self._nseq += 1
        if self._nseq <= WARM_SEQ:
            return
        if self._tk is None:
            self._tk, self._rows_k = now, 0
            return
        if self._k > 1 and self._rows_k >= MIN_ROWS:
            rate = (now - self._tk) / self._rows_k
            proj = (now - self._t0) + rate * max(TOTAL_ROWS - self._rows, 0)
            if proj > BUDGET_S:
                self._set_k(self._k - 1)

    def predict(self, data_point):
        if self.current_seq_ix != data_point.seq_ix or data_point.step_in_seq == 0:
            self.current_seq_ix = data_point.seq_ix
            self._check_time()
            self._lib.ens_reset(self._ms, self._n)
        self._x[:] = data_point.state
        self._step(self._ms, self._nk, self._wp, self._xp, self._op, self._fs0, self._fs1)
        self._rows += 1; self._rows_k += 1
        if not data_point.need_prediction:
            return None
        return self._out.copy()

"""Submission RNN stateful (ONNX, CPU, 1 thread). Va nello ZIP insieme a model.onnx e config.json."""
from pathlib import Path
import json
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import onnxruntime as ort


class PredictionModel:
    def __init__(self):
        here = Path(__file__).resolve().parent
        config = json.loads((here / "config.json").read_text()) if (here / "config.json").exists() else {}
        # input = feature + differenze x[t]-x[t-k] per ogni lag (vecchio flag diff = lag 1)
        self.lags = np.array(config.get("lags") or ([1] if config.get("diff") else []), dtype=np.int64)
        self.window = int(self.lags.max()) + 1 if len(self.lags) else 0
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        options.add_session_config_entry("session.inter_op.allow_spinning", "0")
        self.session = ort.InferenceSession(
            str(here / "model.onnx"),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        inputs = self.session.get_inputs()
        self.has_c = any(i.name == "c" for i in inputs)   # LSTM: (h, c); GRU: solo h
        self.state_shape = next(i.shape for i in inputs if i.name == "h")
        self.current_seq_ix = None
        self._reset()

    def _reset(self):
        self.h = np.zeros(self.state_shape, dtype=np.float32)
        self.c = np.zeros(self.state_shape, dtype=np.float32) if self.has_c else None
        self.hist, self.t = None, 0   # buffer circolare delle ultime righe (solo con i lag)

    def predict(self, data_point):
        # nuova sequenza: stato azzerato (anche se lo stesso seq_ix ripartisse dallo step 0)
        if self.current_seq_ix != data_point.seq_ix or data_point.step_in_seq == 0:
            self.current_seq_ix = data_point.seq_ix
            self._reset()
        x = np.asarray(data_point.state, dtype=np.float32)
        if self.window:
            x = x.astype(np.float16).astype(np.float32)   # come in training, dove gli input sono float16
            if self.hist is None:                         # prima riga: il buffer vale x[0] (differenze nulle)
                self.hist = np.tile(x, (self.window, 1))
            self.hist[self.t % self.window] = x
            x = np.concatenate([x, (x - self.hist[(self.t - self.lags) % self.window]).ravel()])
            self.t += 1
        feed = {"features": x.reshape(1, 1, -1), "h": self.h}
        if self.has_c:
            feed["c"] = self.c
        out = self.session.run(None, feed)
        pred, self.h = out[0], out[1]
        if self.has_c:
            self.c = out[2]
        if not data_point.need_prediction:
            return None
        return np.asarray(pred[0, 0], dtype=np.float32)


if __name__ == "__main__":
    import argparse
    import sys
    # utils.py sta nella radice dello starterpack: la cerca risalendo le cartelle
    sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "utils.py").exists())))
    from utils import ScorerStepByStep

    parser = argparse.ArgumentParser()
    parser.add_argument("--validation", required=True)
    args = parser.parse_args()
    print(ScorerStepByStep(args.validation).score(PredictionModel()))

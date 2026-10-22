"""Submission LSTM stateful (ONNX, CPU, 1 thread). Va nello ZIP insieme a model.onnx."""
from pathlib import Path
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import onnxruntime as ort


class PredictionModel:
    def __init__(self):
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        options.add_session_config_entry("session.inter_op.allow_spinning", "0")
        self.session = ort.InferenceSession(
            str(Path(__file__).resolve().parent / "model.onnx"),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        shape = next(i.shape for i in self.session.get_inputs() if i.name == "h")
        self.h = np.zeros(shape, dtype=np.float32)
        self.c = np.zeros(shape, dtype=np.float32)
        self.current_seq_ix = None

    def predict(self, data_point):
        # nuova sequenza: stato azzerato (anche se lo stesso seq_ix ripartisse dallo step 0)
        if self.current_seq_ix != data_point.seq_ix or data_point.step_in_seq == 0:
            self.current_seq_ix = data_point.seq_ix
            self.h.fill(0.0)
            self.c.fill(0.0)
        features = np.asarray(data_point.state, dtype=np.float32).reshape(1, 1, -1)
        pred, self.h, self.c = self.session.run(None, {"features": features, "h": self.h, "c": self.c})
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

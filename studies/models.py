"""Caricamento dei modelli da analizzare nei notebook, tutti con la stessa interfaccia:
model(x, state) -> (pred, state), con x di forma (batch, step, 112) e pred (batch, step, 2).

    load_model('baseline')                                   # GRU ufficiale, pesi presi da baseline/baseline.onnx
    load_model('lstm/runs/lstm_h128_l2/best.pt')             # un nostro checkpoint (percorso relativo a studies/)
"""
from pathlib import Path
import sys

import numpy as np
import onnx
from onnx import numpy_helper
import torch
from torch import nn

STUDIES = Path(__file__).resolve().parent
BASELINE_ONNX = STUDIES.parent / 'baseline' / 'baseline.onnx'


class BaselineGRU(nn.Module):
    """Ricostruzione in PyTorch del baseline ONNX: 2 GRU da 128 + testa lineare."""

    def __init__(self, n_features=112, hidden=128):
        super().__init__()
        self.gru0 = nn.GRU(n_features, hidden, batch_first=True)
        self.gru1 = nn.GRU(hidden, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 2)

    def forward(self, x, state=None):
        h0, h1 = state if state is not None else (None, None)
        out, h0 = self.gru0(x, h0)
        out, h1 = self.gru1(out, h1)
        return self.head(out), (h0, h1)


def _onnx_gru_to_torch(gru, W, R, B):
    """ONNX ordina le porte come (z, r, h), PyTorch come (r, z, n): si riordinano i blocchi."""
    def reorder(a):
        z, r, h = np.split(a, 3, axis=0)
        return np.concatenate([r, z, h], axis=0)
    hidden = R.shape[-1]
    with torch.no_grad():
        gru.weight_ih_l0.copy_(torch.from_numpy(reorder(W[0])))
        gru.weight_hh_l0.copy_(torch.from_numpy(reorder(R[0])))
        gru.bias_ih_l0.copy_(torch.from_numpy(reorder(B[0, :3 * hidden])))
        gru.bias_hh_l0.copy_(torch.from_numpy(reorder(B[0, 3 * hidden:])))


def load_baseline(path=BASELINE_ONNX):
    graph = onnx.load(str(path)).graph
    weights = {t.name: numpy_helper.to_array(t) for t in graph.initializer}
    grus = [n for n in graph.node if n.op_type == 'GRU']
    for n in grus:   # il baseline usa la stessa variante di GRU di PyTorch
        assert any(a.name == 'linear_before_reset' and a.i == 1 for a in n.attribute)
    model = BaselineGRU()
    for gru, node in zip((model.gru0, model.gru1), grus):
        _onnx_gru_to_torch(gru, *(weights[name] for name in node.input[1:4]))
    matmul = next(n for n in graph.node if n.op_type == 'MatMul')
    with torch.no_grad():
        model.head.weight.copy_(torch.from_numpy(weights[matmul.input[1]].T.copy()))
        model.head.bias.copy_(torch.from_numpy(weights['reg_head.bias'].copy()))
    return model.eval()


def load_model(name, device='cpu'):
    """'baseline' oppure il percorso di un checkpoint .pt (assoluto o relativo a studies/)."""
    if name == 'baseline':
        return load_baseline().to(device)
    sys.path.insert(0, str(STUDIES / 'lstm'))
    from model import LSTMModel
    path = Path(name) if Path(name).is_absolute() else STUDIES / name
    ckpt = torch.load(path, map_location=device)
    model = LSTMModel(**ckpt['config'])
    model.load_state_dict(ckpt['state_dict'])
    return model.to(device).eval()

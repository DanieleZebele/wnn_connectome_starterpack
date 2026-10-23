"""Feature derivate dagli input grezzi, identiche in training (GPU, sequenze intere) e in inferenza (riga per riga).

Differenze a più ritardi: per ogni lag k, x[t] - x[t-k]. Dove t-k < 0 si usa la riga 0 (la differenza vale 0 all'inizio
della sequenza): in inferenza basta un buffer circolare inizializzato con la prima riga.
Gli input del training sono float16 (cache): in inferenza va arrotondato a float16 prima di sottrarre.
"""
import torch


def lags_of(config):
    """Lista dei lag di un checkpoint (supporta il vecchio flag `diff` = lag [1])."""
    return list(config.get('lags') or ([1] if config.get('diff') else []))


def with_lags(xb, s, e, lags):
    """xb: (B, T, F) sequenze intere (float16 o float32). Restituisce (B, e-s, F*(1+len(lags))) per gli step s..e-1:
    le feature seguite da x[t] - x[t-k] per ogni lag."""
    x = xb[:, s:e].float()
    if not lags:
        return x
    t = torch.arange(s, e, device=xb.device)
    return torch.cat([x] + [x - xb[:, (t - k).clamp(min=0)].float() for k in lags], dim=-1)

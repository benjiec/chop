from typing import Iterable, List
import numpy as np
from utils.constants import GenePredictionClass as P


def _logit(x: np.ndarray) -> np.ndarray:
    # Use float64 for stability and a slightly wider clamp to avoid infs
    xf = np.asarray(x, dtype=np.float64)
    xf = np.clip(xf, 1e-6, 1 - 1e-6)
    return np.log(xf) - np.log(1 - xf)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


EVENT_CLASSES: List[int] = [int(P.START), int(P.STOP), int(P.DSS), int(P.ASS)]


def batch_global_logit_standardize(batch_probs: List[np.ndarray], beta: float = 1.0) -> List[np.ndarray]:
    """Apply per-class global logit standardization across a batch of sequences.

    For each event class c, gather logits over the batch, compute mean/std, then
    standardize each sequence's logits and map back with sigmoid(beta * z).

    Only event-class columns are adjusted; other classes remain unchanged.
    """
    if not batch_probs:
        return batch_probs

    # Stack logits per class across batch
    class_logits = {c: [] for c in EVENT_CLASSES}
    for probs in batch_probs:
        for c in EVENT_CLASSES:
            class_logits[c].append(_logit(probs[:, c]))

    mu = {}
    sigma = {}
    for c, parts in class_logits.items():
        allc = np.concatenate(parts, axis=0)
        # Guard against NaNs (shouldn't occur after clipping, but be robust)
        allc = allc[np.isfinite(allc)]
        if allc.size == 0:
            mu[c] = 0.0
            sigma_c = 1.0
        else:
            mu[c] = float(np.mean(allc))
            sigma_c = float(np.std(allc))
        sigma[c] = sigma_c if sigma_c > 1e-8 else 1.0

    # Transform each sequence
    out: List[np.ndarray] = []
    for probs in batch_probs:
        adj = probs.copy()
        for c in EVENT_CLASSES:
            l = _logit(adj[:, c])
            z = (l - mu[c]) / sigma[c]
            pc = _sigmoid(beta * z)
            adj[:, c] = pc.astype(adj.dtype, copy=False)
        # Clamp to avoid zeros
        np.clip(adj, 1e-12, 1 - 1e-12, out=adj)
        out.append(adj)
    return out



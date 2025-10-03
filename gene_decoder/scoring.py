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


def global_z_normalize_prob(batch_probs: List[np.ndarray], beta: float = 1.0) -> List[np.ndarray]:
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



def temperature_rescale_probs(probs: np.ndarray, t_ratio: float) -> np.ndarray:
    """Rescale softmax probabilities from temperature T_old to T_new using t_ratio = T_old / T_new.

    For each position (row) with class probabilities p, compute:
      p_new[i] = p[i] ** t_ratio / sum_j p[j] ** t_ratio

    This is equivalent to p = softmax(z / T_old) -> p_new = softmax(z / T_new),
    without needing the original logits z.

    Args:
        probs: Array of shape (L, C) with per-position class probabilities.
        t_ratio: Ratio T_old / T_new. >1.0 sharpens, <1.0 flattens.

    Returns:
        New probabilities with the same shape (L, C), each row summing to 1.
    """
    if probs is None:
        return probs
    # Ensure 2D shape (L, C)
    arr = np.asarray(probs)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim != 2:
        raise ValueError("temperature_rescale_probs expects probs with shape (L, C) or (C,)")

    tr = float(t_ratio)
    # Clip and use float64 for numerical stability during exponentiation
    p = np.clip(arr.astype(np.float64, copy=False), 1e-12, 1.0)
    # Equivalent to exp(tr * log p)
    scaled = np.exp(tr * np.log(p))
    # Normalize row-wise
    denom = np.sum(scaled, axis=-1, keepdims=True)
    # Avoid divide-by-zero; denom should be > 0 because of clipping
    denom = np.clip(denom, 1e-300, np.inf)
    out = (scaled / denom).astype(arr.dtype, copy=False)
    return out if probs.ndim == 2 else out[0]


def temperature_rescale_probs_T(probs: np.ndarray, T_old: float, T_new: float) -> np.ndarray:
    """Rescale probabilities from a known old temperature to a new temperature.

    Usage examples:
        # From T_old to no temperature (T_new = 1)
        p_new = temperature_rescale_probs_T(p, T_old=3.0, T_new=1.0)

        # General conversion via ratio
        p_new = temperature_rescale_probs_T(p, T_old=3.0, T_new=2.0)

    This computes t_ratio = T_old / T_new and applies `temperature_rescale_probs`.
    """
    return temperature_rescale_probs(probs, float(T_old) / float(T_new))



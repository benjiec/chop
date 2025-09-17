#!/usr/bin/env python3

from typing import List, Tuple, Sequence, Literal, Optional
import numpy as np


def compute_window_slices(seq_len: int, window: int, stride: int) -> List[Tuple[int, int]]:
    """
    Compute half-open slices [(start, end), ...] covering [0, seq_len),
    with step size `stride` and windows of length up to `window`.
    Ensures the last window reaches seq_len.
    """
    assert seq_len > 0 and window > 0 and stride > 0
    if seq_len <= window:
        return [(0, seq_len)]
    slices: List[Tuple[int, int]] = []
    start = 0
    while start + window < seq_len:
        end = start + window
        slices.append((start, end))
        start += stride
    # Ensure coverage of the tail
    if slices:
        last_start = max(0, seq_len - window)
        if not slices or slices[-1][1] < seq_len or slices[-1][0] != last_start:
            slices.append((last_start, seq_len))
    else:
        slices.append((0, seq_len))
    # Normalize to be strictly within bounds
    slices = [(max(0, s), min(seq_len, e)) for (s, e) in slices]
    return slices


def window_weights(length: int, mode: Literal['cosine', 'triangular'] = 'cosine', margin: Optional[int] = None) -> np.ndarray:
    """
    Return center-peaked weights of given length.
    - cosine: raised cosine over [0, L-1], peak at center, 0 at edges
    - triangular: linear ramp up then down, peak at center, 0 at edges
    If `margin` is provided, additionally down-weights the outer `margin` positions.
    """
    assert length > 0
    x = np.linspace(0.0, 1.0, num=length)
    if mode == 'cosine':
        w = 0.5 * (1.0 - np.cos(2.0 * np.pi * x))
    elif mode == 'triangular':
        mid = (length - 1) / 2.0
        w = 1.0 - (np.abs(np.arange(length) - mid) / (mid if mid > 0 else 1.0))
    else:
        raise ValueError(f"Unknown weight mode: {mode}")
    # Ensure strictly positive weights to avoid zero-weight positions at window edges
    w = np.clip(w, 1e-3, None)
    if margin is not None and margin > 0:
        mask = np.ones(length, dtype=np.float32)
        m = min(margin, length // 2)
        if m > 0:
            # Apply a simple taper to the outer margins
            taper = np.linspace(0.25, 1.0, num=m, dtype=np.float32)
            mask[:m] *= taper
            mask[-m:] *= taper[::-1]
        w = w * mask
        w = np.clip(w, 1e-3, None)
    return w.astype(np.float32)


def blend_logits(
    seq_len: int,
    slices: Sequence[Tuple[int, int]],
    window_logits: Sequence[np.ndarray],  # each (win_len, num_classes)
    weight_mode: Literal['cosine', 'triangular'] = 'cosine',
    margin: Optional[int] = None,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Weighted logit averaging across overlapping windows.
    Returns blended logits of shape (seq_len, num_classes).
    """
    assert len(slices) == len(window_logits)
    if not slices:
        raise ValueError("No slices provided for blending")
    num_classes = int(window_logits[0].shape[-1])
    sums = np.zeros((seq_len, num_classes), dtype=np.float32)
    weight_sums = np.zeros((seq_len,), dtype=np.float32)
    for (s, e), wl in zip(slices, window_logits):
        win_len = e - s
        if win_len <= 0:
            continue
        if wl.shape[0] != win_len:
            raise ValueError(f"Window logits length {wl.shape[0]} does not match slice length {win_len}")
        w = window_weights(win_len, mode=weight_mode, margin=margin).reshape(win_len, 1)
        sums[s:e, :] += (w * wl).astype(np.float32)
        weight_sums[s:e] += w.squeeze(1).astype(np.float32)
    # Avoid division by zero; if a position has zero weight (shouldn't), keep zeros
    weight_sums = np.maximum(weight_sums, eps)
    blended = sums / weight_sums[:, None]
    return blended



#!/usr/bin/env python3

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple


def compute_ssmd(pos_mean: float, pos_std: float, neg_mean: float, neg_std: float) -> float:
    """Compute SSMD = (m_p - m_n) / sqrt(s_p^2 + s_n^2) on scalar inputs.

    Inputs are expected to be span-level means and standard deviations.
    Guards small denominators to avoid division-by-zero.
    """
    mp = float(pos_mean)
    sp = float(max(pos_std, 0.0))
    mn = float(neg_mean)
    sn = float(max(neg_std, 0.0))
    denom = (sp * sp + sn * sn) ** 0.5
    if denom <= 1e-12:
        return 0.0
    return (mp - mn) / denom


@dataclass
class AlphaTrendState:
    ema_ssmd: float
    prev_ema_ssmd: float
    alpha: float


def alpha_from_ssmd_trend(
    *,
    ssmd_now: float,
    state: AlphaTrendState,
    ssmd_target: float = 2.0,
    k_p: float = 0.2,
    k_d: float = 0.1,
    beta: float = 0.8,
    alpha0: float = 1.0,
    alpha_min: float = 0.05,
    hysteresis: int = 1,
) -> Tuple[float, AlphaTrendState]:
    """Trend-aware scheduler for per-class alpha based on SSMD.

    Uses a PD-like update on an EMA of SSMD. Returns (new_alpha, new_state).
    """
    # EMA update
    ema_prev = float(state.ema_ssmd)
    ema = float(beta) * float(ssmd_now) + (1.0 - float(beta)) * ema_prev

    # Error (want SSMD close to/above target) and derivative (trend)
    e = float(ssmd_target) - ema
    d = ema - float(state.prev_ema_ssmd)

    raw = float(state.alpha) + float(k_p) * e - float(k_d) * d

    # Clip to [alpha_min, alpha0]
    a = max(float(alpha_min), min(float(alpha0), raw))

    # Hysteresis: optionally require stability across epochs before applying large swings
    # Here we implement a minimal variant by softly nudging toward the new value when hysteresis>1
    if int(hysteresis) > 1:
        blend = 1.0 / float(hysteresis)
        a = float(state.alpha) * (1.0 - blend) + float(a) * blend

    new_state = AlphaTrendState(ema_ssmd=ema, prev_ema_ssmd=ema_prev, alpha=a)
    return a, new_state



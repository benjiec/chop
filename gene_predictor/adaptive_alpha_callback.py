#!/usr/bin/env python3

import pytorch_lightning as pl
from typing import Dict, Optional

from utils.adaptive_alpha import AlphaTrendState, alpha_from_ssmd_trend


class AdaptiveAlphaCallback(pl.Callback):
    """Adjust per-class alpha weights at epoch end using SSMD from MetricsCallback.

    Updates the provided alpha_by_class dict in place, which is also used by the loss.
    """

    def __init__(
        self,
        *,
        alpha_by_class: Dict[int, float],
        metrics_cb,

	# ssmd_target: computed on windows, not blended, so value will be lower
	# then the final blended SSMD. 2.0 here is probably approaching 3 at
	# blended level
        ssmd_target: float = 2.0,

        k_p: float = 0.2,
        k_d: float = 0.1,
        beta: float = 0.8,
        alpha_min: float = 0.05,

    ) -> None:
        super().__init__()
        self._alpha_by_class = alpha_by_class
        self._initial_alpha_by_class = {k:v for k,v in alpha_by_class.items()}  # making a copy, not using reference
        self._metrics_cb = metrics_cb
        self._ssmd_target = float(ssmd_target)
        self._k_p = float(k_p)
        self._k_d = float(k_d)
        self._beta = float(beta)
        self._alpha_min = float(alpha_min)
        # Initialize per-class scheduler state with current alphas
        self._state_by_class: Dict[int, AlphaTrendState] = {}

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module) -> None:
        if trainer is not None and getattr(trainer, 'sanity_checking', False):
            return
        ssmd_by_class = getattr(self._metrics_cb, 'last_ssmd_by_class', None)
        if not isinstance(ssmd_by_class, dict) or not ssmd_by_class:
            return
        updated = {}
        for cls_id, ssmd in ssmd_by_class.items():
            alpha0 = float(self._initial_alpha_by_class.get(int(cls_id), 1.0))
            st = self._state_by_class.get(int(cls_id))
            if st is None:
                st = AlphaTrendState(ema_ssmd=float(ssmd), prev_ema_ssmd=float(ssmd), alpha=alpha0)
            ssmd_target = self._ssmd_target
            a, new_state = alpha_from_ssmd_trend(
                ssmd_now=float(ssmd),
                state=st,
                ssmd_target=ssmd_target,
                k_p=self._k_p,
                k_d=self._k_d,
                beta=self._beta,
                alpha0=alpha0,
                alpha_min=self._alpha_min,
                hysteresis=1,
            )
            self._state_by_class[int(cls_id)] = new_state
            self._alpha_by_class[int(cls_id)] = float(a)
            updated[int(cls_id)] = float(a)
            print(f"updated alpha for class {cls_id} from {st.alpha:.1f} to {a:.1f} based on ema ssmd {st.ema_ssmd:.2f} and cur ssmd {ssmd:.2f}")

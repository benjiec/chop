#!/usr/bin/env python3

import pytorch_lightning as pl
import torch
import numpy as np

from utils.metrics import calculate_generic_metrics
from utils.constants import GenePredictionClass as P


class F1Callback(pl.Callback):
    """Compute macro-average F1 over classes returned by calculate_generic_metrics on validation data.

    Also logs per-class F1 to the logger (not the progress bar) and can optionally
    print a compact per-class summary to stdout every N epochs.
    """

    def __init__(self, val_loader, print_per_class_every: int = 1):
        super().__init__()
        self.val_loader = val_loader
        self.print_per_class_every = int(print_per_class_every) if print_per_class_every is not None else 0

    @torch.no_grad()
    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module):
        pl_module.eval()
        device = next(pl_module.parameters()).device

        results_data = []

        for sequences, targets in self.val_loader:
            sequences = sequences.to(device)
            targets = targets.to(device)
            logits = pl_module.model(sequences)
            predictions = logits.argmax(dim=-1)

            batch_size = sequences.size(0)
            for b in range(batch_size):
                seq_tokens = sequences[b].detach().cpu().numpy()
                tgt = targets[b].detach().cpu().numpy()
                pred = predictions[b].detach().cpu().numpy()
                results_data.append({
                    'sequence_index': len(results_data),
                    'sequence_tokens': seq_tokens,
                    'targets': tgt,
                    'predictions': pred,
                    'probabilities': None,
                })

        # Pull class weights from model config if available
        try:
            class_weights = getattr(pl_module, 'config', {}).get('loss', {}).get('class_weights')
        except Exception:
            class_weights = None

        # Optional validity masks to exclude window edges using loss_window_margin_fraction
        valid_masks = None
        try:
            # If the module has model config, derive margin from model's max seq length
            max_len = int(getattr(pl_module, 'config', {}).get('model', {}).get('max_seq_length', 0) or 0)
            frac = float(getattr(pl_module, 'config', {}).get('loss', {}).get('loss_window_margin_fraction', 0.2) or 0.2)
            if max_len > 0 and frac > 0.0:
                margin = int(max(0, min(max_len // 2, round(frac * max_len))))
                valid_masks = []
                for r in results_data:
                    L = len(r['sequence_tokens'])
                    mask = [True] * L
                    if margin > 0 and L > 2 * margin:
                        for i in range(0, margin):
                            mask[i] = False
                        for i in range(L - margin, L):
                            mask[i] = False
                    valid_masks.append(mask)
        except Exception:
            valid_masks = None

        metrics_by_class = calculate_generic_metrics(results_data, class_weights=class_weights, min_weight=1.0, valid_masks=valid_masks)

        # Compute macro-average F1 across all classes returned
        f1_values = []
        per_class_f1 = {}
        for cls_idx, m in metrics_by_class.items():
            tp = float(m.get('tp', 0))
            fp = float(m.get('fp', 0))
            fn = float(m.get('fn', 0))
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            denom = precision + sensitivity
            f1 = (2.0 * precision * sensitivity / denom) if denom > 0.0 else 0.0
            f1_values.append(f1)
            cls_name = P.idx_to_cls.get(int(cls_idx), str(int(cls_idx)))
            per_class_f1[cls_name] = f1

        macro_f1 = float(np.median(f1_values)) if f1_values else 0.0

        # Progress bar: only macro F1
        pl_module.log('val_f1', macro_f1, prog_bar=True, on_epoch=True)

        # Logger-only: per-class F1
        for name, value in per_class_f1.items():
            pl_module.log(f"val_f1_classes/{name}", float(value), prog_bar=False, on_epoch=True)

        # Optional compact stdout summary
        should_print = self.print_per_class_every and (
            (trainer is None) or ((getattr(trainer, 'current_epoch', 0) + 1) % self.print_per_class_every == 0)
        )
        if should_print and per_class_f1:
            ordered = sorted(per_class_f1.items(), key=lambda kv: kv[0])
            summary = ' '.join([f"{k}={v:.2f}" for k, v in ordered])
            print(f"F1 per class: {summary}")



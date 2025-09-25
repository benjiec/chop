#!/usr/bin/env python3

import pytorch_lightning as pl
import torch
import numpy as np

from utils.metrics import calculate_generic_metrics
from utils.constants import GenePredictionClass as P

# Add Brier computation
from utils.metrics import compute_brier_scores


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
            probabilities = torch.softmax(logits, dim=-1)

            batch_size = sequences.size(0)
            for b in range(batch_size):
                seq_tokens = sequences[b].detach().cpu().numpy()
                tgt = targets[b].detach().cpu().numpy()
                pred = predictions[b].detach().cpu().numpy()
                probs = probabilities[b].detach().cpu().numpy()
                results_data.append({
                    'sequence_index': len(results_data),
                    'sequence_tokens': seq_tokens,
                    'targets': tgt,
                    'predictions': pred,
                    'probabilities': probs,
                })

        # Pull class weights from model config if available
        class_weights = pl_module.config.get('loss', {}).get('class_weights')

        # Optional validity masks to exclude window edges using bp margin
        valid_masks = None

        # If the module has config, read bp margin directly
        margin_bp = int(pl_module.config.get('loss', {}).get('loss_window_margin_bp', 200) or 0)
        if margin_bp > 0:
            valid_masks = []
            for r in results_data:
                L = len(r['sequence_tokens'])
                mask = [True] * L
                if L > 2 * margin_bp:
                    for i in range(0, margin_bp):
                        mask[i] = False
                    for i in range(L - margin_bp, L):
                        mask[i] = False
                valid_masks.append(mask)

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

        # Brier score (overall + per-class)
        brier = compute_brier_scores(results_data, class_weights=class_weights, min_weight=1.0, valid_masks=valid_masks)
        pl_module.log('val_brier', float(brier.get('brier', 0.0)), prog_bar=True, on_epoch=True)
        by_class = brier.get('brier_by_class', {})
        for cls_idx, val in by_class.items():
            name = P.idx_to_cls.get(int(cls_idx), str(int(cls_idx)))
            pl_module.log(f"val_brier_classes/{name}", float(val), prog_bar=False, on_epoch=True)

        # Optional compact stdout summary
        should_print = self.print_per_class_every and (
            (trainer is None) or (((trainer.current_epoch + 1) % self.print_per_class_every) == 0)
        )
        if should_print and per_class_f1:
            ordered = sorted(per_class_f1.items(), key=lambda kv: kv[0])
            summary = ' '.join([f"{k}={v:.2f}" for k, v in ordered])
            print(f"\nF1 per class: {summary}")


class DualMetricEarlyStopping(pl.Callback):
    """Stop if neither val_loss improves (min) nor val_f1 improves (max) for N epochs."""

    def __init__(self, patience: int = 8):
        super().__init__()
        self.patience = int(patience)
        self.best_loss = None
        self.best_f1 = None
        self.wait = 0

    def on_validation_end(self, trainer: pl.Trainer, pl_module):
        metrics = trainer.callback_metrics
        val_loss = float(metrics.get('val_loss', float('inf')))
        val_f1 = float(metrics.get('val_f1', float('-inf')))
        improved = False

        if trainer.current_epoch <= 1:  # first two epochs, just continue
            trainer.should_stop = False
            self.wait = 0
            return

        if self.best_loss is None or val_loss < self.best_loss - 1e-12:
            print("best_loss improved", self.best_loss, val_loss)
            self.best_loss = val_loss
            improved = True
        if self.best_f1 is None or val_f1 > self.best_f1 + 1e-12:
            print("best_f1 improved", self.best_f1, val_f1)
            self.best_f1 = val_f1
            improved = True
        if improved:
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                trainer.should_stop = True


class LossComponentsCallback(pl.Callback):
    """Aggregate and report validation loss components per epoch.

    Uses model._compute_adjusted_loss(..., components_out=...) to ensure
    components match training/validation logic exactly.
    """

    def on_validation_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._sum_total = 0.0
        self._sum_ce = 0.0
        self._sum_entropy = 0.0
        self._sum_fp = 0.0
        self._count = 0
        self._ce_sum_by_class = {}
        self._wt_sum_by_class = {}
        self._total_weighted_ce_sum = 0.0

    def on_validation_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs, batch, batch_idx: int, dataloader_idx: int = 0) -> None:
        try:
            sequences, targets = batch
            logits = pl_module.model(sequences)
            comp = {}
            _ = pl_module._compute_adjusted_loss(logits, targets, components_out=comp)
            self._sum_total += float(comp.get('total', 0.0))
            self._sum_ce += float(comp.get('ce', 0.0))
            self._sum_entropy += float(comp.get('entropy', 0.0))
            self._sum_fp += float(comp.get('fp_penalty', 0.0))
            self._total_weighted_ce_sum += float(comp.get('total_weighted_ce_sum', 0.0))
            for k, v in (comp.get('ce_weighted_sum_by_class', {}) or {}).items():
                self._ce_sum_by_class[int(k)] = self._ce_sum_by_class.get(int(k), 0.0) + float(v)
            for k, v in (comp.get('weight_sum_by_class', {}) or {}).items():
                self._wt_sum_by_class[int(k)] = self._wt_sum_by_class.get(int(k), 0.0) + float(v)
            self._count += 1
        except Exception:
            pass

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self._count <= 0:
            return
        mean_total = self._sum_total / self._count
        mean_ce = self._sum_ce / self._count
        mean_entropy = self._sum_entropy / self._count
        mean_fp = self._sum_fp / self._count
        pl_module.print(f"Loss components (val epoch {trainer.current_epoch}): total={mean_total:.4f} CE={mean_ce:.4f} entropy={mean_entropy:.4f} fp_penalty={mean_fp:.4f}")
        if self._wt_sum_by_class:
            pl_module.print("CE per class (weighted means) and share of total CE:")
            for k in sorted(self._ce_sum_by_class.keys()):
                denom = self._wt_sum_by_class.get(k, 0.0)
                mean_k = (self._ce_sum_by_class[k] / denom) if denom > 0 else float('nan')
                share_k = (self._ce_sum_by_class[k] / self._total_weighted_ce_sum) if self._total_weighted_ce_sum > 0 else 0.0
                name = P.idx_to_cls.get(int(k), str(int(k)))
                pl_module.print(f"  {name:>10s}: CE_mean={mean_k:.4f} CE_share={share_k:.2%}")

#!/usr/bin/env python3

import pytorch_lightning as pl
import csv
from pathlib import Path
import torch
import numpy as np

from utils.constants import GenePredictionClass as P


class F1Callback(pl.Callback):
    """Compute macro-average F1 over classes returned by calculate_generic_metrics on validation data.

    Also logs per-class F1 to the logger (not the progress bar) and can optionally
    print a compact per-class summary to stdout every N epochs.
    """

    def __init__(self, val_loader, print_per_class_every: int = 1, margin_bp: int = 0,
                 calculate_metrics_fn=None, compute_brier_fn=None):
        super().__init__()
        self.val_loader = val_loader
        self.print_per_class_every = int(print_per_class_every) if print_per_class_every is not None else 0
        self.margin_bp = int(margin_bp) if margin_bp is not None else 0
        self._calculate_metrics_fn = calculate_metrics_fn
        self._compute_brier_fn = compute_brier_fn

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

        # Optional validity masks to exclude window edges using bp margin
        valid_masks = None

        margin_bp = self.margin_bp
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

        metrics_fn = self._calculate_metrics_fn
        brier_fn = self._compute_brier_fn
        if metrics_fn is None or brier_fn is None:
            raise RuntimeError("F1Callback requires calculate_metrics_fn and compute_brier_fn to be provided")

        metrics_by_class = metrics_fn(results_data, min_weight=1.0, valid_masks=valid_masks)

        # Optional compact stdout summary
        should_print = self.print_per_class_every and (
            (trainer is None) or (((trainer.current_epoch + 1) % self.print_per_class_every) == 0)
        )

        # Compute macro-average F1 across all classes returned
        f1_values = []
        per_class_f1 = {}
        for cls_idx, m in metrics_by_class.items():
            precision = float(m.get('precision', 0.0))
            sensitivity = float(m.get('sensitivity', 0.0))
            denom = precision + sensitivity
            f1 = (2.0 * precision * sensitivity / denom) if denom > 0.0 else 0.0
            # Include only classes with any events (tp+fp+fn > 0)
            if int(m.get('tp', 0)) + int(m.get('fp', 0)) + int(m.get('fn', 0)) > 0:
                f1_values.append(f1)
            cls_name = P.idx_to_cls.get(int(cls_idx), str(int(cls_idx)))
            per_class_f1[cls_name] = f1
            if should_print:
                print(cls_name,"sen",sensitivity,"pre",precision,"f1",f1)

        macro_f1 = float(np.median(f1_values)) if f1_values else 0.0

        # Progress bar: only macro F1
        pl_module.log('val_f1', macro_f1, prog_bar=True, on_epoch=True)

        # Logger-only: per-class F1
        for name, value in per_class_f1.items():
            pl_module.log(f"val_f1_classes/{name}", float(value), prog_bar=False, on_epoch=True)

        # Brier score (overall + per-class)
        brier = brier_fn(results_data, valid_masks=valid_masks)
        pl_module.log('val_brier', float(brier.get('brier', 0.0)), prog_bar=True, on_epoch=True)
        by_class = brier.get('brier_by_class', {})
        for cls_idx, val in by_class.items():
            name = P.idx_to_cls.get(int(cls_idx), str(int(cls_idx)))
            pl_module.log(f"val_brier_classes/{name}", float(val), prog_bar=False, on_epoch=True)


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
    """Aggregate and report loss components for validation and training per epoch.

    For validation, reads components precomputed by pl_module.validation_step.
    For training, reuses components computed in training_step (no extra forward).
    """

    def __init__(self, report_train_components: bool = True, run_dir: Path | None = None):
        super().__init__()
        self.report_train_components = bool(report_train_components)
        self.run_dir = Path(run_dir) if run_dir is not None else None

    # ---- Validation aggregation ----
    def on_validation_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._v_sum_total = 0.0
        self._v_sum_ce = 0.0
        self._v_sum_entropy = 0.0
        self._v_sum_fp = 0.0
        self._v_count = 0
        self._v_ce_sum_by_class = {}
        self._v_wt_sum_by_class = {}
        self._v_total_weighted_ce_sum = 0.0

    def on_validation_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs, batch, batch_idx: int, dataloader_idx: int = 0) -> None:
        comp = getattr(pl_module, '_last_val_components', None)
        if not isinstance(comp, dict):
            return
        self._v_sum_total += float(comp.get('total', 0.0))
        self._v_sum_ce += float(comp.get('ce', 0.0))
        self._v_sum_entropy += float(comp.get('entropy', 0.0))
        self._v_sum_fp += float(comp.get('fp_penalty', 0.0))
        self._v_total_weighted_ce_sum += float(comp.get('total_weighted_ce_sum', 0.0))
        for k, v in (comp.get('ce_weighted_sum_by_class', {}) or {}).items():
            self._v_ce_sum_by_class[int(k)] = self._v_ce_sum_by_class.get(int(k), 0.0) + float(v)
        for k, v in (comp.get('weight_sum_by_class', {}) or {}).items():
            self._v_wt_sum_by_class[int(k)] = self._v_wt_sum_by_class.get(int(k), 0.0) + float(v)
        self._v_count += 1

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        # Compute means if available
        v_ce = v_ent = v_fp = v_total = None
        t_ce = t_ent = t_fp = t_total = None
        if self._v_count > 0:
            v_total = self._v_sum_total / self._v_count
            v_ce = self._v_sum_ce / self._v_count
            v_ent = self._v_sum_entropy / self._v_count
            v_fp = self._v_sum_fp / self._v_count
        if getattr(self, '_t_count', 0) > 0:
            t_total = self._t_sum_total / self._t_count
            t_ce = self._t_sum_ce / self._t_count
            t_ent = self._t_sum_entropy / self._t_count
            t_fp = self._t_sum_fp / self._t_count

        # Optionally write CSV row
        if self.run_dir is not None:
            # Tall format (epoch_summary.csv)
            self._write_epoch_csv_tall(trainer, pl_module, v_total, v_ce, v_ent, v_fp, t_total, t_ce, t_ent, t_fp)

    # ---- Training aggregation ----
    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if not self.report_train_components:
            return
        self._t_sum_total = 0.0
        self._t_sum_ce = 0.0
        self._t_sum_entropy = 0.0
        self._t_sum_fp = 0.0
        self._t_count = 0
        self._t_ce_sum_by_class = {}
        self._t_wt_sum_by_class = {}
        self._t_total_weighted_ce_sum = 0.0

    def on_train_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs, batch, batch_idx: int) -> None:
        if not self.report_train_components:
            return
        comp = getattr(pl_module, '_last_train_components', None)
        if not isinstance(comp, dict):
            return
        self._t_sum_total += float(comp.get('total', 0.0))
        self._t_sum_ce += float(comp.get('ce', 0.0))
        self._t_sum_entropy += float(comp.get('entropy', 0.0))
        self._t_sum_fp += float(comp.get('fp_penalty', 0.0))
        self._t_total_weighted_ce_sum += float(comp.get('total_weighted_ce_sum', 0.0))
        for k, v in (comp.get('ce_weighted_sum_by_class', {}) or {}).items():
            self._t_ce_sum_by_class[int(k)] = self._t_ce_sum_by_class.get(int(k), 0.0) + float(v)
        for k, v in (comp.get('weight_sum_by_class', {}) or {}).items():
            self._t_wt_sum_by_class[int(k)] = self._t_wt_sum_by_class.get(int(k), 0.0) + float(v)
        self._t_count += 1

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        # No console printing; values are written at validation end
        return

    # ---- Helpers ----
    def _mean_ce_for(self, ce_sum_by_class: dict, wt_sum_by_class: dict, cls_idx: int) -> float | None:
        if ce_sum_by_class is None or wt_sum_by_class is None:
            return None
        num = ce_sum_by_class.get(int(cls_idx))
        den = wt_sum_by_class.get(int(cls_idx))
        if num is None or den is None or den <= 0:
            return None
        return float(num) / float(den)

    def _write_epoch_csv_tall(self, trainer: pl.Trainer, pl_module: pl.LightningModule,
                              v_total, v_ce, v_ent, v_fp,
                              t_total, t_ce, t_ent, t_fp) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        csv_path = self.run_dir / 'epoch_summary.csv'
        exists = csv_path.exists()

        # Pull scalar metrics
        metrics = trainer.callback_metrics or {}
        val_f1 = metrics.get('val_f1')
        val_brier = metrics.get('val_brier')
        # Only use logged metrics; if missing, leave blank in CSV
        val_loss = metrics.get('val_loss')
        train_loss = metrics.get('train_loss')
        # Component means logged by the module (epoch-averaged by Lightning)
        val_ce = metrics.get('val_loss_ce')
        val_ent = metrics.get('val_loss_entropy')
        val_fp = metrics.get('val_loss_fp')
        tr_ce = metrics.get('train_loss_ce')
        tr_ent = metrics.get('train_loss_entropy')
        tr_fp = metrics.get('train_loss_fp')

        # Per-class CE means for key classes (use constants only here; model remains class-agnostic)
        v_start = self._mean_ce_for(self._v_ce_sum_by_class, self._v_wt_sum_by_class, P.START)
        v_stop = self._mean_ce_for(self._v_ce_sum_by_class, self._v_wt_sum_by_class, P.STOP)
        v_dss = self._mean_ce_for(self._v_ce_sum_by_class, self._v_wt_sum_by_class, P.DSS)
        v_ass = self._mean_ce_for(self._v_ce_sum_by_class, self._v_wt_sum_by_class, P.ASS)
        t_start = self._mean_ce_for(self._t_ce_sum_by_class, self._t_wt_sum_by_class, P.START) if self.report_train_components and hasattr(self, "_t_ce_sum_by_class") else None
        t_stop = self._mean_ce_for(self._t_ce_sum_by_class, self._t_wt_sum_by_class, P.STOP) if self.report_train_components and hasattr(self, "_t_ce_sum_by_class") else None
        t_dss = self._mean_ce_for(self._t_ce_sum_by_class, self._t_wt_sum_by_class, P.DSS) if self.report_train_components and hasattr(self, "_t_ce_sum_by_class") else None
        t_ass = self._mean_ce_for(self._t_ce_sum_by_class, self._t_wt_sum_by_class, P.ASS) if self.report_train_components and hasattr(self, "_t_ce_sum_by_class") else None

        # Prepare tall rows: (batch, metric_name -> value)
        rows = []
        def add(split, name, value):
            rows.append([int(trainer.current_epoch), split, name, _safe_float(value)])

        # Validation metrics
        add('val', 'f1', val_f1)
        add('val', 'brier', val_brier)
        add('val', 'loss', val_loss)
        add('val', 'loss_CE', val_ce)
        add('val', 'loss_entropy', val_ent)
        add('val', 'loss_fp_penalty', val_fp)
        add('val', 'loss_START', v_start)
        add('val', 'loss_STOP', v_stop)
        add('val', 'loss_DSS', v_dss)
        add('val', 'loss_ASS', v_ass)

        # Training metrics
        add('train', 'loss', train_loss)
        add('train', 'loss_CE', tr_ce)
        add('train', 'loss_entropy', tr_ent)
        add('train', 'loss_fp_penalty', tr_fp)
        add('train', 'loss_START', t_start)
        add('train', 'loss_STOP', t_stop)
        add('train', 'loss_DSS', t_dss)
        add('train', 'loss_ASS', t_ass)

        # Write
        with csv_path.open('a', newline='') as f:
            writer = csv.writer(f)
            if not exists:
                writer.writerow(['epoch', 'batch', 'metric', 'value'])
            writer.writerows(rows)


def _safe_float(x):
    try:
        if x is None:
            return ''
        return float(x)
    except Exception:
        try:
            return float(getattr(x, 'item')())
        except Exception:
            return ''

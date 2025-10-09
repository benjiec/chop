#!/usr/bin/env python3

import pytorch_lightning as pl
import csv
from pathlib import Path
import torch
import numpy as np

from utils.constants import GenePredictionClass as P


# Module-level CSV writer state and function (so it can be patched in tests)
RUN_DIR: Path | None = None

def write_epoch_csv_tall(trainer: pl.Trainer, macro_f1: float, overall_brier: float, per_class: dict) -> None:
    global RUN_DIR
    if RUN_DIR is None:
        return
    run_dir = RUN_DIR
    run_dir.mkdir(parents=True, exist_ok=True)
    csv_path = run_dir / 'epoch_summary.csv'
    exists = csv_path.exists()

    rows = []
    def add(name, value):
        rows.append([int(trainer.current_epoch) if trainer is not None else 0, 'val', name, _safe_float(value)])

    # Aggregates
    add('f1', macro_f1)
    add('brier', overall_brier)

    # Per-class metrics
    for cls_idx, vals in per_class.items():
        name = P.idx_to_cls.get(int(cls_idx), str(int(cls_idx)))
        add(f'f1_{name}', vals.get('f1'))
        add(f'sensitivity_{name}', vals.get('sensitivity'))
        add(f'precision_{name}', vals.get('precision'))
        add(f'brier_{name}', vals.get('brier'))

    with csv_path.open('a', newline='') as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(['epoch', 'batch', 'metric', 'value'])
        writer.writerows(rows)

class MetricsCallback(pl.Callback):
    """Compute validation metrics (macro F1, Brier) and per-class stats; optionally write TSV."""

    def __init__(self, val_loader, print_per_class_every: int = 1, margin_bp: int = 0,
                 calculate_metrics_fn=None, compute_brier_fn=None, run_dir: Path | None = None):
        super().__init__()
        self.val_loader = val_loader
        self.print_per_class_every = int(print_per_class_every) if print_per_class_every is not None else 0
        self.margin_bp = int(margin_bp) if margin_bp is not None else 0
        self._calculate_metrics_fn = calculate_metrics_fn
        self._compute_brier_fn = compute_brier_fn
        self.run_dir = Path(run_dir) if run_dir is not None else None
        # Set module-level run directory for CSV writer
        global RUN_DIR
        RUN_DIR = self.run_dir

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
        if self.margin_bp > 0:
            valid_masks = []
            for r in results_data:
                L = len(r['sequence_tokens'])
                mask = [True] * L
                if L > 2 * self.margin_bp:
                    for i in range(0, self.margin_bp):
                        mask[i] = False
                    for i in range(L - self.margin_bp, L):
                        mask[i] = False
                valid_masks.append(mask)

        metrics_fn = self._calculate_metrics_fn
        brier_fn = self._compute_brier_fn
        if metrics_fn is None or brier_fn is None:
            raise RuntimeError("MetricsCallback requires calculate_metrics_fn and compute_brier_fn to be provided")

        metrics_by_class = metrics_fn(results_data, min_weight=1.0, valid_masks=valid_masks)

        # Compute per-class F1, precision, sensitivity and macro F1
        f1_values = []
        per_class = {}
        for cls_idx, m in metrics_by_class.items():
            precision = float(m.get('precision', 0.0))
            sensitivity = float(m.get('sensitivity', 0.0))
            denom = precision + sensitivity
            f1 = (2.0 * precision * sensitivity / denom) if denom > 0.0 else 0.0
            # Include only classes with any events (tp+fp+fn > 0)
            if int(m.get('tp', 0)) + int(m.get('fp', 0)) + int(m.get('fn', 0)) > 0:
                f1_values.append(f1)
            per_class[int(cls_idx)] = {
                'precision': precision,
                'sensitivity': sensitivity,
                'f1': f1,
            }

        macro_f1 = float(np.median(f1_values)) if f1_values else 0.0
        pl_module.log('val_f1', macro_f1, prog_bar=True, on_epoch=True)

        # Brier score (overall + per-class)
        brier = brier_fn(results_data, valid_masks=valid_masks)
        overall_brier = float(brier.get('brier', 0.0))
        pl_module.log('val_brier', overall_brier, prog_bar=True, on_epoch=True)
        for cls_idx, val in (brier.get('brier_by_class', {}) or {}).items():
            if int(cls_idx) not in per_class:
                per_class[int(cls_idx)] = {}
            per_class[int(cls_idx)]['brier'] = float(val)

        # Log per-class metrics
        should_print = self.print_per_class_every and (
            (trainer is None) or (((trainer.current_epoch + 1) % self.print_per_class_every) == 0)
        )
        for cls_idx, vals in per_class.items():
            name = P.idx_to_cls.get(int(cls_idx), str(int(cls_idx)))
            if 'f1' in vals:
                pl_module.log(f"val_f1_classes/{name}", float(vals['f1']), prog_bar=False, on_epoch=True)
            if 'sensitivity' in vals:
                pl_module.log(f"val_sensitivity_classes/{name}", float(vals['sensitivity']), prog_bar=False, on_epoch=True)
            if 'precision' in vals:
                pl_module.log(f"val_precision_classes/{name}", float(vals['precision']), prog_bar=False, on_epoch=True)
            if 'brier' in vals:
                pl_module.log(f"val_brier_classes/{name}", float(vals['brier']), prog_bar=False, on_epoch=True)
            if should_print:
                print(name, 'sen', vals.get('sensitivity', 0.0), 'pre', vals.get('precision', 0.0), 'f1', vals.get('f1', 0.0), 'brier', vals.get('brier', 0.0))

        # Optionally write tall TSV of metrics
        if self.run_dir is not None:
            write_epoch_csv_tall(trainer, macro_f1, overall_brier, per_class)


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

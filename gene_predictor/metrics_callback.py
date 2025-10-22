#!/usr/bin/env python3

import pytorch_lightning as pl
import csv
from pathlib import Path
import torch
import numpy as np

from utils.constants import GenePredictionClass as P
from utils.metrics import SequenceResult
from utils.metrics import compute_event_span_mean_probability_metrics
from utils.events import build_class_logits_from_event_head_logits


def write_epoch_csv_tall(run_dir: Path | None, trainer: pl.Trainer, macro_f1: float, overall_brier: float, per_class: dict, val_loss: float | None = None) -> None:
    if run_dir is None:
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    csv_path = run_dir / 'epoch_summary.csv'
    exists = csv_path.exists()

    rows = []
    def add(metric_name, class_name, value):
        # Skip writing rows during sanity check
        if trainer is not None and getattr(trainer, 'sanity_checking', False):
            return
        rows.append([
            int(trainer.current_epoch) if trainer is not None else 0,
            ('sanity' if (trainer is not None and getattr(trainer, 'sanity_checking', False)) else 'val'),
            metric_name,
            class_name,
            _safe_float(value),
        ])

    # Aggregates
    add('f1', '', macro_f1)
    add('brier', '', overall_brier)
    if val_loss is not None:
        add('loss', '', val_loss)

    # Per-class metrics (write all provided keys)
    for cls_idx, vals in per_class.items():
        name = P.idx_to_cls.get(int(cls_idx), str(int(cls_idx)))
        for key, val in vals.items():
            add(str(key), name, val)

    with csv_path.open('a', newline='') as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(['epoch', 'batch', 'metric', 'class', 'value'])
        writer.writerows(rows)


class MetricsCallback(pl.Callback):
    """Compute validation metrics (macro F1, Brier) and per-class stats; optionally write TSV."""

    def __init__(self, val_loader, verbose: int = 1, margin_bp: int = 0,
                 calculate_metrics_fn=None, compute_brier_fn=None, run_dir: Path | None = None,
                 event_logits_conversion_fn=None, event_motifs_by_class=None, head_class_ids=None,
                 alpha_by_class: dict | None = None):
        super().__init__()
        self.val_loader = val_loader
        self.verbose = int(verbose) if verbose is not None else 0
        self.margin_bp = int(margin_bp) if margin_bp is not None else 0
        self._calculate_metrics_fn = calculate_metrics_fn
        self._compute_brier_fn = compute_brier_fn
        self.run_dir = Path(run_dir) if run_dir is not None else None
        self._event_logits_conversion_fn = event_logits_conversion_fn
        self._event_motifs_by_class = event_motifs_by_class
        self._head_class_ids = list(head_class_ids) if head_class_ids is not None else None
        # Initialize results accumulator so tests that call only epoch_end won't fail
        self._results_data = []
        # Optional shared alpha reference for printing
        self._alpha_by_class = alpha_by_class

    def on_validation_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._results_data = []

    @torch.no_grad()
    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module):
        pl_module.eval()

        results_data = self._results_data

        # Convert list[BatchResult] into SequenceResult entries using provided conversion fn when present
        batch_results = pl_module.validation_epoch_results
        for br in batch_results:
            seq = br.sequence_tokens_batch
            tgt = br.targets_batch
            logits = br.logits_batch
            ev = br.event_logits_batch
            if callable(self._event_logits_conversion_fn) and (ev is not None):
                logits_for_metrics = self._event_logits_conversion_fn(seq, ev)
            else:
                logits_for_metrics = logits
            sr_list = SequenceResult.from_batch(
                sequence_tokens_batch=seq,
                targets_batch=tgt,
                logits_batch=logits_for_metrics,
                sequence_index_start=int(br.sequence_index_start),
                mask_non_event_probs=False,
            )
            if results_data is None:
                results_data = []
            results_data.extend(sr_list)

        # Optional validity masks to exclude window edges using bp margin
        valid_masks = None
        if self.margin_bp > 0:
            valid_masks = []
            for r in results_data:
                L = len(r.sequence_tokens)
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

        macro_f1 = float(np.mean(f1_values)) if f1_values else 0.0
        pl_module.log('val_f1', macro_f1, prog_bar=True, on_epoch=True)

        # Brier score (overall + per-class)
        brier = brier_fn(results_data, valid_masks=valid_masks)
        overall_brier = float(brier.get('brier', 0.0))
        pl_module.log('val_brier', overall_brier, prog_bar=True, on_epoch=True)
        for cls_idx, val in (brier.get('brier_by_class', {}) or {}).items():
            if int(cls_idx) not in per_class:
                per_class[int(cls_idx)] = {}
            per_class[int(cls_idx)]['brier'] = float(val)

        # Add probability distribution summaries if motifs provided
        if self._event_motifs_by_class:
            fits = compute_event_span_mean_probability_metrics(results_data, self._event_motifs_by_class)
            for cls_idx, data in (fits or {}).items():
                tp = data.get('tp', {})
                tn = data.get('tn', {})
                if int(cls_idx) not in per_class:
                    per_class[int(cls_idx)] = {}
                per_class[int(cls_idx)]['pos_prob_mean'] = float(tp.get('mean', 0.0))
                per_class[int(cls_idx)]['pos_prob_std'] = float(tp.get('std', 0.0))
                per_class[int(cls_idx)]['neg_prob_mean'] = float(tn.get('mean', 0.0))
                per_class[int(cls_idx)]['neg_prob_std'] = float(tn.get('std', 0.0))

        # Aggregate loss via trainer metrics, map to per-class using head_class_ids when available
        val_loss = None
        if trainer is not None and hasattr(trainer, 'callback_metrics'):
            cbm = trainer.callback_metrics
            val_loss = float(cbm.get('val_loss')) if 'val_loss' in cbm else None
            # Collect aggregated per-head validation losses logged as val_loss_head_{i}
            for mk, mv in cbm.items():
                if isinstance(mk, str) and mk.startswith('val_loss_head_'):
                    # map to components key name used elsewhere
                    suffix = mk[len('val_'):]
                    # Also store into per_class when mapping available
                    try:
                        head_idx = int(suffix.split('_')[-1])
                    except Exception:
                        head_idx = None
                    if head_idx is not None and self._head_class_ids is not None and head_idx < len(self._head_class_ids):
                        cls_id = int(self._head_class_ids[head_idx])
                        if int(cls_id) not in per_class:
                            per_class[int(cls_id)] = {}
                        per_class[int(cls_id)]['event_head_loss'] = float(mv)

        if self.verbose:
            for cls_idx, vals in per_class.items():
                name = P.idx_to_cls.get(int(cls_idx), str(int(cls_idx)))
                print(name,
                      "%.4f" % vals.get('event_head_loss', 0.0),
                      "%d/%d-%d/%d" % (vals.get('pos_prob_mean', 0.0)*100, vals.get('pos_prob_std', 0.0)*100,
                                       vals.get('neg_prob_mean', 0.0*100), vals.get('neg_prob_std', 0.0)*100),
                      "%.4f" % vals.get('brier', 0.0),
                      "%d/%d" % (vals.get('sensitivity', 0.0)*100, vals.get('precision', 0.0)*100)
                )

        # Always print alphas used for loss at validation end if provided
        if isinstance(self._alpha_by_class, dict):
            # Print compact mapping in class-name order START, STOP, DSS, ASS when available
            order = [int(P.START), int(P.STOP), int(P.DSS), int(P.ASS)]
            parts = []
            for cid in order:
                name = P.idx_to_cls.get(int(cid), str(int(cid)))
                val = self._alpha_by_class.get(int(cid))
                if val is not None:
                    parts.append(f"{name}={float(val):.3f}")
            if parts:
                print("alpha_by_class:", ", ".join(parts))

        # Optionally write tall TSV of metrics 
        if self.run_dir is not None:
            write_epoch_csv_tall(self.run_dir, trainer, macro_f1, overall_brier, per_class, val_loss)


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

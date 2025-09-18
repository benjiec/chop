#!/usr/bin/env python3

import pytorch_lightning as pl
import torch
import numpy as np

from utils.metrics import calculate_generic_metrics


class F1Callback(pl.Callback):
    """Compute macro-average F1 over classes returned by calculate_generic_metrics on validation data."""

    def __init__(self, val_loader):
        super().__init__()
        self.val_loader = val_loader

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

        metrics_by_class = calculate_generic_metrics(results_data, class_weights=class_weights, min_weight=1.0)

        # Compute macro-average F1 across all classes returned
        f1_values = []
        for cls_idx, m in metrics_by_class.items():
            tp = float(m.get('tp', 0))
            fp = float(m.get('fp', 0))
            fn = float(m.get('fn', 0))
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            denom = precision + sensitivity
            f1 = (2.0 * precision * sensitivity / denom) if denom > 0.0 else 0.0
            f1_values.append(f1)

        macro_f1 = float(np.mean(f1_values)) if f1_values else 0.0

        pl_module.log('val_f1', macro_f1, prog_bar=True, on_epoch=True)



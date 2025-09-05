#!/usr/bin/env python3

import pytorch_lightning as pl
import torch
from utils.constants import GenePredictionClass as P


class StartSensitivityCallback(pl.Callback):
    """Compute ATG-only START Sensitivity on the validation loader and log it.

    Logs metric_name (default: 'val_start_sensitivity_atg') each validation epoch.
    Sensitivity = TP / (TP + FN) over positions where the underlying DNA has 'ATG'.
    """

    def __init__(self, val_loader, metric_name: str = 'val_start_sensitivity_atg'):
        super().__init__()
        self.val_loader = val_loader
        self.metric_name = metric_name

    @torch.no_grad()
    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module):
        pl_module.eval()
        device = next(pl_module.parameters()).device
        true_positives = 0
        false_negatives = 0

        # index->base mapping used throughout the project
        idx_to_nucleotide = {0: 'A', 1: 'T', 2: 'G', 3: 'C', 4: 'N'}

        for sequences, targets in self.val_loader:
            sequences = sequences.to(device)
            targets = targets.to(device)
            logits = pl_module.model(sequences)
            predictions = logits.argmax(dim=-1)

            batch_size = sequences.size(0)
            for b in range(batch_size):
                # Reconstruct sequence string to identify ATG positions
                token_ids = sequences[b].detach().cpu().tolist()
                dna = ''.join(idx_to_nucleotide.get(int(t), 'N') for t in token_ids)
                tgt = targets[b]
                pred = predictions[b]

                length = min(len(dna), tgt.numel(), pred.numel())
                for pos in range(0, length - 2):
                    if dna[pos:pos+3] != 'ATG':
                        continue
                    is_true_start = (tgt[pos].item() == P.START)
                    is_pred_start = (pred[pos].item() == P.START)
                    if is_true_start and is_pred_start:
                        true_positives += 1
                    elif is_true_start and not is_pred_start:
                        false_negatives += 1

        sensitivity = (true_positives / (true_positives + false_negatives)) if (true_positives + false_negatives) > 0 else 0.0
        pl_module.log(self.metric_name, sensitivity, prog_bar=True, on_epoch=True)



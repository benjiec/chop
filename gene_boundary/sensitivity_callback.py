#!/usr/bin/env python3

import pytorch_lightning as pl
import torch
from utils.constants import GenePredictionClass as P, DNAEmbed, ConventionalStopCodons as stop_codons


class BoundarySensitivityCallback(pl.Callback):
    """Compute triplet-aware START and STOP sensitivity on the validation loader.

    Logs per-epoch metrics:
      - val_start_sensitivity_atg
      - val_stop_sensitivity_taa_tag_tga
      - val_event_sensitivity (mean of START/STOP sensitivities)
    """

    def __init__(self, val_loader):
        super().__init__()
        self.val_loader = val_loader

    @torch.no_grad()
    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module):
        pl_module.eval()
        device = next(pl_module.parameters()).device

        start_tp = 0
        start_fn = 0
        stop_tp = 0
        stop_fn = 0

        idx_to_nucleotide = DNAEmbed.idx_to_bp

        for sequences, targets in self.val_loader:
            sequences = sequences.to(device)
            targets = targets.to(device)
            logits = pl_module.model(sequences)
            predictions = logits.argmax(dim=-1)

            batch_size = sequences.size(0)
            for b in range(batch_size):
                token_ids = sequences[b].detach().cpu().tolist()
                dna = ''.join(idx_to_nucleotide.get(int(t), 'N') for t in token_ids)
                tgt = targets[b]
                pred = predictions[b]

                length = min(len(dna), tgt.numel(), pred.numel())
                for pos in range(0, max(0, length - 2)):
                    triplet = dna[pos:pos+3]
                    if triplet == 'ATG':
                        true_start = ((tgt[pos:pos+3] == P.START).any().item())
                        pred_start = ((pred[pos:pos+3] == P.START).any().item())
                        if true_start and pred_start:
                            start_tp += 1
                        elif true_start and not pred_start:
                            start_fn += 1
                    elif triplet in stop_codons:
                        true_stop = ((tgt[pos:pos+3] == P.STOP).any().item())
                        pred_stop = ((pred[pos:pos+3] == P.STOP).any().item())
                        if true_stop and pred_stop:
                            stop_tp += 1
                        elif true_stop and not pred_stop:
                            stop_fn += 1

        start_sens = (start_tp / (start_tp + start_fn)) if (start_tp + start_fn) > 0 else 0.0
        stop_sens = (stop_tp / (stop_tp + stop_fn)) if (stop_tp + stop_fn) > 0 else 0.0
        event_sens = (start_sens + stop_sens) / 2.0

        pl_module.log('val_start_ss', start_sens, prog_bar=True, on_epoch=True)
        pl_module.log('val_stop_ss', stop_sens, prog_bar=True, on_epoch=True)
        pl_module.log('val_event_ss', event_sens, prog_bar=True, on_epoch=True)



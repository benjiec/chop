#!/usr/bin/env python3

import unittest
import tempfile
from pathlib import Path

import torch
import pytorch_lightning as pl

from dna_learner.trainer import train as train_fn
from dna_learner.model import create_base_config
from utils.dataset import GenomicSyntheticTestingDataset, RandomBasesGenerator, RandomUTR5Generator, AddATGGenerator
from utils.sequences import KOZAK_SEQUENCES, UTR5_REAL_SEQUENCES, IRES_SEQUENCES
from utils.constants import GenePredictionClass as P


class PhaseFlagCallback(pl.Callback):
    def __init__(self, phase_ref):
        super().__init__()
        self.phase_ref = phase_ref
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        self.phase_ref['phase'] = 'train'
    def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
        self.phase_ref['phase'] = 'val'


class TestTrainerCustomLossIntegration(unittest.TestCase):
    def test_trainer_passes_custom_loss_called_train_and_val(self):
        utr_choices = KOZAK_SEQUENCES + UTR5_REAL_SEQUENCES + IRES_SEQUENCES
        layouts = [
            RandomBasesGenerator(length=50, target=P.INTERGENIC),
            RandomUTR5Generator(choices=utr_choices, target=P.UTR5, mutation_prob=0.0),
            AddATGGenerator(),
            RandomBasesGenerator(length=50, target=P.INTERGENIC),
        ]
        dataset = GenomicSyntheticTestingDataset(
            max_sequence_length=200,
            num_contigs=8,
            layouts_per_contig=1,
            layouts=layouts,
        )

        config = create_base_config(
            max_seq_length=200,
            num_classes=3,
            class_names=['INTERGENIC', 'UTR5', 'START'],
            d_model=16,
            n_layers=1,
            n_heads=4,
            learning_rate=1e-3,
            max_epochs=1,
            batch_size=2,
            class_weights=[1.0, 1.0, 1.0],
            attention_masks={0: 2},
            kmer_size=0,
        )

        phase = {'phase': None}
        counts = {'train': 0, 'val': 0}

        def custom_loss(sequences, targets, logits, components_out):
            # Count calls by phase set by the callback
            p = phase.get('phase')
            if p in counts:
                counts[p] += 1
            # Return a constant tensor attached to graph
            return logits.sum() * 0.0 + torch.tensor(0.1, dtype=logits.dtype, device=logits.device)

        phase_cb = PhaseFlagCallback(phase)

        def cb_gen(val_loader):
            return [phase_cb]

        with tempfile.TemporaryDirectory() as tmpdir:
            train_fn(
                dataset,
                config,
                Path(tmpdir),
                cb_gen,
                monitor_metric='val_loss',
                monitor_mode='min',
                custom_loss_fn=custom_loss,
            )

        # Ensure the custom loss was invoked during both phases at least once
        self.assertGreater(counts['train'], 0)
        self.assertGreater(counts['val'], 0)


if __name__ == '__main__':
    unittest.main()



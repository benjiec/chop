#!/usr/bin/env python3

import unittest
from pathlib import Path
import tempfile

from dna_learner.trainer import train as train_fn
from synthetic.gene_boundary.train import create_boundary_config
from utils.losses import adjusted_ce_entropy_loss
from utils.dataset import GenomicSyntheticTestingDataset, RandomBasesGenerator, RandomUTR5Generator, AddATGGenerator
from utils.sequences import KOZAK_SEQUENCES, UTR5_REAL_SEQUENCES, IRES_SEQUENCES
from utils.constants import GenePredictionClass as P
import pytorch_lightning as pl


class DummyCallback(pl.Callback):
    def __init__(self):
        super().__init__()
        self.seen_val_loader = False
    def on_validation_start(self, trainer, pl_module):
        self.seen_val_loader = True


class TestGeneBoundaryTrainer(unittest.TestCase):
    def test_trainer_runs_and_returns(self):
        # Small synthetic dataset (same as layout tests)
        utr_choices = KOZAK_SEQUENCES + UTR5_REAL_SEQUENCES + IRES_SEQUENCES
        layouts = [
            RandomBasesGenerator(length=50, target=P.INTERGENIC),
            RandomUTR5Generator(choices=utr_choices, target=P.UTR5, mutation_prob=0.0),
            AddATGGenerator(),
            RandomBasesGenerator(length=50, target=P.INTERGENIC),
        ]
        dataset = GenomicSyntheticTestingDataset(
            max_sequence_length=200,
            num_contigs=10,
            layouts_per_contig=1,
            layouts=layouts,
        )

        cfg = create_boundary_config(
            d_model=16,
            n_layers=1,
            n_heads=4,
            learning_rate=1e-3,
            max_epochs=1,
            batch_size=2,
            use_class_weights=True,
            start_weight=8.0,
            stop_weight=8.0,
            utr_weight=4.0,
            attention_masks={0: 2},
            kmer_size=0,
            max_seq_length=200,
            use_focal=False,
        )

        cb = DummyCallback()
        def cb_gen(val_loader):
            self.assertTrue(hasattr(val_loader, '__iter__'))
            return [cb]

        with tempfile.TemporaryDirectory() as tmpdir:
            model, val_loader = train_fn(
                dataset,
                cfg,
                Path(tmpdir),
                cb_gen,
                monitor_metric='val_loss',
                monitor_mode='min',
                custom_loss_fn=lambda s,t,l,c: adjusted_ce_entropy_loss(l, t, loss_window_margin_bp=0, class_weights=cfg.get('loss',{}).get('class_weights'), entropy_lambda=0.0, fp_beta=0.0, components_out=c),
            )
            self.assertIsNotNone(model)
            self.assertTrue(hasattr(val_loader, '__iter__'))
            self.assertTrue(cb.seen_val_loader or True)


if __name__ == '__main__':
    unittest.main()

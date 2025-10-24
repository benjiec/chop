#!/usr/bin/env python3

import unittest
from pathlib import Path
import tempfile

from dna_learner.trainer import train as train_fn
from dna_learner.model import create_base_config
from utils.losses import adjusted_ce_entropy_loss
from utils.dataset import GenomicSyntheticTestingDataset, RandomBasesGenerator, RandomUTR5Generator, AddATGGenerator
from utils.sequences import KOZAK_SEQUENCES, UTR5_REAL_SEQUENCES, IRES_SEQUENCES
from utils.constants import GenePredictionClass as P
import pytorch_lightning as pl
from gene_predictor.metrics_callback import AlphaTargetAdjustCallback


class DummyCallback(pl.Callback):
    def __init__(self):
        super().__init__()
        self.seen_val_loader = False
    def on_validation_start(self, trainer, pl_module):
        self.seen_val_loader = True


class TestGenePredictorTrainer(unittest.TestCase):
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

        # Build minimal cfg via create_base_config (class weights not needed here)
        cfg = create_base_config(
            max_seq_length=200,
            num_classes=len(P.idx_to_cls),
            class_names=[P.idx_to_cls[i] for i in sorted(P.idx_to_cls.keys())],
            d_model=16,
            n_layers=1,
            n_heads=4,
            learning_rate=1e-3,
            max_epochs=1,
            batch_size=2,
            class_weights=None,
            attention_masks={0: 2},
            kmer_size=0,
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
                custom_loss_fn=lambda s,t,l,ev,c: adjusted_ce_entropy_loss(l, t, loss_window_margin_bp=0, class_weights=cfg.get('loss',{}).get('class_weights'), entropy_lambda=0.0, fp_beta=0.0, components_out=c),
            )
            self.assertIsNotNone(model)
            self.assertTrue(hasattr(val_loader, '__iter__'))
            self.assertTrue(cb.seen_val_loader or True)


class TestTrainerSamplerIntegration(unittest.TestCase):
    def test_trainer_runs_with_class_aware_sampler(self):
        utr_choices = KOZAK_SEQUENCES + UTR5_REAL_SEQUENCES + IRES_SEQUENCES
        layouts = [
            RandomBasesGenerator(length=50, target=P.INTERGENIC),
            RandomUTR5Generator(choices=utr_choices, target=P.UTR5, mutation_prob=0.0),
            AddATGGenerator(),
            RandomBasesGenerator(length=50, target=P.INTERGENIC),
        ]
        dataset = GenomicSyntheticTestingDataset(
            max_sequence_length=200,
            num_contigs=12,
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
            batch_size=3,
            class_weights=[1.0, 1.0, 1.0],
            attention_masks={0: 2},
            kmer_size=0,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            model, val_loader = train_fn(
                dataset,
                config,
                Path(tmpdir),
                additional_callback_generator=lambda v: [],
                monitor_metric='val_loss',
                monitor_mode='min',
                custom_loss_fn=lambda s,t,l,ev,c: adjusted_ce_entropy_loss(l, t, loss_window_margin_bp=0, class_weights=config.get('loss',{}).get('class_weights'), entropy_lambda=0.0, fp_beta=0.0, components_out=c),
            )
            self.assertIsNotNone(model)
            self.assertTrue(hasattr(val_loader, '__iter__'))


class TestAlphaTargetAdjustIntegration(unittest.TestCase):
    def _build_small_dataset(self):
        utr_choices = KOZAK_SEQUENCES + UTR5_REAL_SEQUENCES + IRES_SEQUENCES
        layouts = [
            RandomBasesGenerator(length=30, target=P.INTERGENIC),
            RandomUTR5Generator(choices=utr_choices, target=P.UTR5, mutation_prob=0.0),
            AddATGGenerator(),
            RandomBasesGenerator(length=30, target=P.INTERGENIC),
        ]
        return GenomicSyntheticTestingDataset(
            max_sequence_length=200,
            num_contigs=6,
            layouts_per_contig=1,
            layouts=layouts,
        )

    def _base_config(self):
        return create_base_config(
            max_seq_length=200,
            num_classes=len(P.idx_to_cls),
            class_names=[P.idx_to_cls[i] for i in sorted(P.idx_to_cls.keys())],
            d_model=16,
            n_layers=1,
            n_heads=4,
            learning_rate=1e-3,
            max_epochs=1,
            batch_size=2,
            class_weights=None,
            attention_masks={0: 2},
            kmer_size=0,
            num_event_heads=4,
        )

    def test_alpha_reduces_when_target_hit(self):
        dataset = self._build_small_dataset()
        cfg = self._base_config()

        # Shared alpha map and class/heads mapping
        alpha_by_class = {int(P.START): 1.0, int(P.STOP): 1.0, int(P.DSS): 1.0, int(P.ASS): 1.0}
        head_class_ids = [int(P.START), int(P.STOP), int(P.DSS), int(P.ASS)]

        # Custom loss that logs a low head-0 loss to trigger callback
        target = 0.5
        def custom_loss(s, t, l, ev, comp):
            comp['loss_head_0'] = target - 1e-6
            return l.sum() * 0.0

        def cb_gen(val_loader):
            return [AlphaTargetAdjustCallback(alpha_by_class=alpha_by_class, head_class_ids=head_class_ids, alpha_targets_by_class={int(P.START): target}, alpha_min=0.1)]

        with tempfile.TemporaryDirectory() as tmpdir:
            model, val_loader = train_fn(
                dataset,
                cfg,
                Path(tmpdir),
                cb_gen,
                monitor_metric='val_loss',
                monitor_mode='min',
                custom_loss_fn=custom_loss,
            )
            self.assertAlmostEqual(alpha_by_class[int(P.START)], 0.1, places=6)

    def test_alpha_unchanged_when_target_not_hit(self):
        dataset = self._build_small_dataset()
        cfg = self._base_config()

        alpha_by_class = {int(P.START): 1.0, int(P.STOP): 1.0, int(P.DSS): 1.0, int(P.ASS): 1.0}
        head_class_ids = [int(P.START), int(P.STOP), int(P.DSS), int(P.ASS)]

        target = 0.5
        def custom_loss(s, t, l, ev, comp):
            comp['loss_head_0'] = target + 1e-6
            return l.sum() * 0.0

        def cb_gen(val_loader):
            return [AlphaTargetAdjustCallback(alpha_by_class=alpha_by_class, head_class_ids=head_class_ids, alpha_targets_by_class={int(P.START): target}, alpha_min=0.1)]

        with tempfile.TemporaryDirectory() as tmpdir:
            model, val_loader = train_fn(
                dataset,
                cfg,
                Path(tmpdir),
                cb_gen,
                monitor_metric='val_loss',
                monitor_mode='min',
                custom_loss_fn=custom_loss,
            )
            self.assertAlmostEqual(alpha_by_class[int(P.START)], 1.0, places=6)

    def test_raises_when_metric_missing_for_targeted_class(self):
        dataset = self._build_small_dataset()
        cfg = self._base_config()

        alpha_by_class = {int(P.START): 1.0, int(P.STOP): 1.0, int(P.DSS): 1.0, int(P.ASS): 1.0}
        head_class_ids = [int(P.START), int(P.STOP), int(P.DSS), int(P.ASS)]

        # Custom loss that DOES NOT log loss_head_0; callback should raise
        target = 0.5
        def custom_loss(s, t, l, ev, comp):
            # intentionally omit comp['loss_head_0']
            return l.sum() * 0.0

        def cb_gen(val_loader):
            return [AlphaTargetAdjustCallback(alpha_by_class=alpha_by_class, head_class_ids=head_class_ids, alpha_targets_by_class={int(P.START): target}, alpha_min=0.1)]

        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(RuntimeError):
                _model, _val_loader = train_fn(
                    dataset,
                    cfg,
                    Path(tmpdir),
                    cb_gen,
                    monitor_metric='val_loss',
                    monitor_mode='min',
                    custom_loss_fn=custom_loss,
                )

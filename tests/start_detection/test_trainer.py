#!/usr/bin/env python3

import unittest
from pathlib import Path
import tempfile

import torch

from synthetic.start_detection.train import create_utr_start_config
from dna_learner.model import GenePredictorModule


class TestTrainer(unittest.TestCase):
    def test_utr_weight_passes_to_module(self):
        # Verify utr_weight affects class_weights in the module's criterion
        cfg = create_utr_start_config(
            d_model=16,
            n_layers=1,
            n_heads=4,
            learning_rate=1e-3,
            max_epochs=1,
            batch_size=2,
            use_class_weights=True,
            start_weight=8.0,
            utr_weight=4.0,
            attention_masks={0: 2},
            kmer_size=0,
            max_seq_length=128,
            use_focal=False,
        )
        module = GenePredictorModule(cfg)
        w = module.criterion.weight.detach().cpu().numpy().tolist()
        self.assertEqual(w, [1.0, 4.0, 8.0])

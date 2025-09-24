import unittest
import torch
import numpy as np

from dna_learner.model import GenePredictorModule, create_base_config


class TestEntropyPenalty(unittest.TestCase):
    def _make_module(self, num_classes=3, entropy_lambda=1e-3):
        cfg = create_base_config(
            max_seq_length=4,
            num_classes=num_classes,
            class_names=[f'C{i}' for i in range(num_classes)],
            d_model=24,
            n_layers=1,
            n_heads=3,
            learning_rate=1e-3,
            max_epochs=1,
            batch_size=1,
            class_weights=[1.0]*num_classes,
            loss_window_margin_bp=0,
            attention_masks=None,
            kmer_size=0,
        )
        cfg['loss']['use_focal'] = False
        cfg['loss']['entropy_lambda'] = entropy_lambda
        cfg['loss']['fp_beta'] = 0.0
        mod = GenePredictorModule(cfg)
        return mod

    def test_entropy_term_effect(self):
        torch.manual_seed(0)
        mod = self._make_module(num_classes=3, entropy_lambda=1e-2)
        # Build (B=1, L=4, C=3) logits and targets
        logits = torch.tensor([
            [[5.0, -2.0, -3.0],  # very confident class 0
             [2.0,  1.8,  1.7],  # moderately confident class 0
             [0.1,  0.0, -0.1],  # flat
             [1.0,  1.0,  1.0],  # uniform
            ]
        ], dtype=torch.float32)
        targets = torch.tensor([[0, 0, 1, 2]], dtype=torch.long)
        # CE-only over included tokens (all here):
        ce_only = torch.nn.functional.cross_entropy(logits.view(-1, 3), targets.view(-1), reduction='mean')
        loss_with_entropy = mod._compute_adjusted_loss(logits, targets)
        # With entropy regularization subtracting lambda*H, the loss should be <= CE-only (since entropy >= 0)
        self.assertLessEqual(float(loss_with_entropy), float(ce_only) + 1e-6)


if __name__ == '__main__':
    unittest.main()

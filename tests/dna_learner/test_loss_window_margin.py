#!/usr/bin/env python3

import unittest
import torch

from dna_learner.model import GenePredictorModule, create_base_config


class TestLossWindowMargin(unittest.TestCase):
    def test_edge_masking_reduces_loss(self):
        # Sequence length and classes
        L = 20
        C = 3
        class_names = [f'C{i}' for i in range(C)]

        # Build tokens and targets: all class 0
        sequences = torch.zeros(1, L, dtype=torch.long)
        targets = torch.zeros(1, L, dtype=torch.long)

        # Logits: edges predict class 1 strongly (wrong), center predicts class 0 strongly (correct)
        logits = torch.zeros(1, L, C)
        # Wrong at edges
        logits[:, :5, 1] = 5.0
        logits[:, -5:, 1] = 5.0
        # Correct center
        logits[:, 5:-5, 0] = 5.0

        # Helper to compute validation loss for a given margin in bp
        def compute_loss(margin_bp: int) -> float:
            cfg = create_base_config(
                max_seq_length=L,
                num_classes=C,
                class_names=class_names,
                d_model=8,
                n_layers=1,
                n_heads=1,
                learning_rate=1e-3,
                batch_size=1,
                loss_window_margin_bp=margin_bp,
            )
            mod = GenePredictorModule(cfg)
            # Replace model with a dummy nn.Module returning fixed logits
            class DummyModel(torch.nn.Module):
                def __init__(self, logits):
                    super().__init__()
                    self._logits = logits
                def forward(self, x, return_attention: bool = False):
                    return self._logits
            mod.model = DummyModel(logits)
            loss = mod.validation_step((sequences, targets), 0)
            return float(loss.detach().cpu().item())

        loss_no_margin = compute_loss(0)
        loss_with_margin = compute_loss(5)  # masks 5 tokens per side for L=20

        # Sanity: without margin, edge errors dominate and loss should be high
        self.assertGreater(loss_no_margin, 1.0)
        # With margin masking edges, loss should drop near zero (only correct center remains)
        self.assertLess(loss_with_margin, 0.1)
        # And strictly lower than without margin
        self.assertLess(loss_with_margin, loss_no_margin)


if __name__ == '__main__':
    unittest.main()



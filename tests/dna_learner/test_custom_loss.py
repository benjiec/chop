#!/usr/bin/env python3

import unittest
import torch

from dna_learner.model import GenePredictorModule, create_base_config


class TestCustomLossHook(unittest.TestCase):
    def test_custom_loss_called_in_training_and_validation(self):
        cfg = create_base_config(
            max_seq_length=64,
            num_classes=3,
            class_names=['A', 'B', 'C'],
            d_model=16,
            n_layers=1,
            n_heads=4,
            learning_rate=1e-3,
            max_epochs=1,
            batch_size=2,
            attention_masks={0: 2},
            kmer_size=0,
        )

        call_counts = {"count": 0}

        def custom_loss(sequences, targets, logits, components_out):
            # Record that custom loss was invoked
            call_counts["count"] += 1
            # Optionally populate components dictionary
            components_out["custom_marker"] = 1.0
            # Return a constant loss tied to the graph
            return logits.sum() * 0.0 + torch.tensor(0.42, dtype=logits.dtype, device=logits.device)

        module = GenePredictorModule(cfg, custom_loss_fn=custom_loss)
        module.eval()

        # Small synthetic batch
        B, L, C = 2, 30, 3
        sequences = torch.randint(0, 5, (B, L))
        targets = torch.randint(0, C, (B, L))

        # Training step should call custom loss once
        loss_train = module.training_step((sequences, targets), 0)
        self.assertEqual(call_counts["count"], 1)
        self.assertTrue(torch.is_tensor(loss_train))
        self.assertAlmostEqual(float(loss_train.detach().cpu().item()), 0.42, places=6)

        # Validation step should call custom loss again
        loss_val = module.validation_step((sequences, targets), 0)
        self.assertEqual(call_counts["count"], 2)
        self.assertTrue(torch.is_tensor(loss_val))
        self.assertAlmostEqual(float(loss_val.detach().cpu().item()), 0.42, places=6)


if __name__ == '__main__':
    unittest.main()



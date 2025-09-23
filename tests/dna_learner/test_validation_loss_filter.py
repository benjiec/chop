import unittest
import torch

from dna_learner.model import GenePredictorModule, create_base_config


class TestValidationLossFilter(unittest.TestCase):
    def _make_module(self, class_weights, entropy_lambda=0.0):
        cfg = create_base_config(
            max_seq_length=4,
            num_classes=3,
            class_names=['A','B','C'],
            d_model=12,
            n_layers=1,
            n_heads=3,
            learning_rate=1e-3,
            max_epochs=1,
            batch_size=1,
            class_weights=class_weights,
            loss_window_margin_fraction=0.0,
            attention_masks=None,
            kmer_size=0,
        )
        cfg['loss']['use_focal'] = False
        cfg['loss']['entropy_lambda'] = entropy_lambda
        mod = GenePredictorModule(cfg)
        return mod

    def test_val_loss_filters_by_weight_gt_one(self):
        # class weights: only class 1 has weight>1
        cw = [1.0, 2.0, 0.0]
        mod = self._make_module(cw, entropy_lambda=0.0)
        # Craft logits so CE differs by class
        # batch=1, L=4, C=3
        logits = torch.tensor([
            [[5.0,  -2.0,  -2.0],  # y=0 confident
             [-2.0,  5.0,  -2.0],  # y=1 confident
             [-2.0, -2.0,   5.0],  # y=2 confident
             [-2.0,  5.0,  -2.0],  # y=1 confident
            ]
        ], dtype=torch.float32)
        targets = torch.tensor([[0,1,2,1]], dtype=torch.long)
        # Monkeypatch model forward to return our logits
        def _fwd(x):
            return logits
        mod.model.forward = _fwd
        loss = mod.validation_step((torch.zeros_like(targets), targets), 0)
        # Expected CE only for class 1 positions (2 positions)
        import math
        p = math.exp(5.0) / (math.exp(5.0) + math.exp(-2.0) + math.exp(-2.0))
        ce = -math.log(p)
        expected = ce  # averaged across the two identical class-1 positions
        self.assertAlmostEqual(float(loss), expected, places=3)


if __name__ == '__main__':
    unittest.main()

import unittest
import torch

from dna_learner.model import GenePredictorModule, create_base_config
from utils.losses import adjusted_ce_entropy_loss


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
            attention_masks=None,
            kmer_size=0,
        )
        cfg['loss']['use_focal'] = False
        cfg['loss']['entropy_lambda'] = entropy_lambda
        mod = GenePredictorModule(
            cfg,
            custom_loss_fn=lambda s,t,l,c: adjusted_ce_entropy_loss(
                l,
                t,
                loss_window_margin_bp=0,
                class_weights=class_weights,
                entropy_lambda=entropy_lambda,
                fp_beta=0.0,
                components_out=c,
            ),
        )
        return mod

    def test_val_loss_weighted_all_center_tokens(self):
        # class weights: only class 1 has weight>1; class 2 weight=0 (ignored)
        cw = [1.0, 2.0, 0.0]
        mod = self._make_module(cw, entropy_lambda=0.0)
        # batch=1, L=4, C=3, each position confident for its true class
        logits = torch.tensor([
            [[5.0,  -2.0,  -2.0],  # y=0
             [-2.0,  5.0,  -2.0],  # y=1
             [-2.0, -2.0,   5.0],  # y=2
             [-2.0,  5.0,  -2.0],  # y=1
            ]
        ], dtype=torch.float32)
        targets = torch.tensor([[0,1,2,1]], dtype=torch.long)
        def _fwd(x):
            return logits
        mod.model.forward = _fwd
        loss = mod.validation_step((torch.zeros_like(targets), targets), 0)
        import math
        # CE is identical for each correctly classified token given identical margins
        p = math.exp(5.0) / (math.exp(5.0) + math.exp(-2.0) + math.exp(-2.0))
        ce = -math.log(p)
        # Weighted mean over tokens (weights by target class): [1,2,0,2]
        expected = (1*ce + 2*ce + 0*ce + 2*ce) / (1+2+0+2)
        self.assertAlmostEqual(float(loss), expected, places=3)


if __name__ == '__main__':
    unittest.main()

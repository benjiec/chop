import unittest
import torch
import torch.nn.functional as F

from dna_learner.model import GenePredictorModule, create_base_config
from utils.losses import adjusted_ce_entropy_loss


class TestLossComponents(unittest.TestCase):
    def test_compute_adjusted_loss_emits_components_correctly(self):
        # Minimal config with no margins and no auxiliary penalties
        cfg = create_base_config(
            max_seq_length=8,
            num_classes=3,
            class_names=['A', 'B', 'C'],
            d_model=16,
            n_layers=1,
            n_heads=4,
            learning_rate=1e-3,
            max_epochs=1,
            batch_size=1,
        )

        module = GenePredictorModule(cfg, custom_loss_fn=lambda s,t,l,c: adjusted_ce_entropy_loss(l, t, loss_window_margin_bp=0, class_weights=None, entropy_lambda=0.0, fp_beta=0.0, components_out=c))
        module.eval()

        # Construct a tiny batch: (B=1, L=4, C=3)
        # Targets use all classes at least once so per-class buckets are populated
        targets = torch.tensor([[0, 1, 2, 1]], dtype=torch.long)
        logits = torch.tensor([
            [  # L=4 rows, C=3 columns
                [2.0, 0.0, -1.0],
                [0.0, 1.5, -0.5],
                [-0.5, 0.0, 2.5],
                [0.2, 1.0, -0.2],
            ]
        ], dtype=torch.float32)

        # Compute expected CE and entropy (no margins, unit weights)
        lf = logits.view(-1, logits.size(-1))  # (N, C)
        tf = targets.view(-1)                   # (N,)
        ce_vec = F.cross_entropy(lf, tf, reduction='none')
        probs = torch.softmax(lf, dim=-1)
        ent_vec = -(probs * torch.log(torch.clamp(probs, min=1e-12))).sum(dim=-1)

        expected_ce_mean = ce_vec.mean().item()
        expected_entropy_mean = ent_vec.mean().item()
        expected_total = expected_ce_mean  # lambda=0, beta=0

        components = {}
        loss = adjusted_ce_entropy_loss(
            logits,
            targets,
            loss_window_margin_bp=0,
            class_weights=None,
            entropy_lambda=0.0,
            fp_beta=0.0,
            components_out=components,
        )

        # Top-level components
        self.assertAlmostEqual(components['total'], expected_total, places=6)
        self.assertAlmostEqual(components['ce'], expected_ce_mean, places=6)
        self.assertAlmostEqual(components['entropy'], expected_entropy_mean, places=6)
        self.assertAlmostEqual(components['fp_penalty'], 0.0, places=8)
        # And the returned loss should equal total
        self.assertAlmostEqual(loss.item(), expected_total, places=6)

        # Per-class weighted sums: with unit weights, sums reduce to unweighted sums
        total_weighted_ce_sum = ce_vec.sum().item()
        self.assertAlmostEqual(components['total_weighted_ce_sum'], total_weighted_ce_sum, places=6)

        ce_ws = components['ce_weighted_sum_by_class']
        wt_ws = components['weight_sum_by_class']
        # Check each present class bucket
        for k in [0, 1, 2]:
            mask_k = (tf == k)
            if mask_k.any():
                expected_num_k = ce_vec[mask_k].sum().item()
                expected_den_k = float(mask_k.sum().item())
                self.assertAlmostEqual(ce_ws[int(k)], expected_num_k, places=6)
                self.assertAlmostEqual(wt_ws[int(k)], expected_den_k, places=6)



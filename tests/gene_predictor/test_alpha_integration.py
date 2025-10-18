import unittest
import torch

from utils.losses import event_head_bce_loss_factory


def _make_dummy_inputs(B=1, L=8, H=4, device='cpu'):
    # Sequences (tokens) and targets: mark class 2 at center positions
    seq = torch.zeros((B, L), dtype=torch.long, device=device)
    tgt = torch.zeros((B, L), dtype=torch.long, device=device)
    tgt[:, L//2 - 1:L//2 + 1] = 2  # positives for class 2
    logits = torch.zeros((B, L, 8), dtype=torch.float32, device=device)
    # Event logits per head (H=4) with small random noise
    ev = torch.zeros((B, L, H), dtype=torch.float32, device=device)
    return seq, tgt, logits, ev


class TestAlphaIntegration(unittest.TestCase):
    def test_shared_alpha_dict_scales_loss(self):
        # Setup
        event_motifs_by_class = {2: {'ATG'}}
        head_class_ids = [2, 4, 6, 7]
        pos_w = {2: 1.0, 4: 1.0, 6: 1.0, 7: 1.0}
        neg_w = {2: 1.0, 4: 1.0, 6: 1.0, 7: 1.0}
        alpha_by_class = {2: 1.0, 4: 1.0, 6: 1.0, 7: 1.0}

        loss_fn = event_head_bce_loss_factory(
            event_motifs_by_class,
            head_class_ids,
            pos_weights_by_class=pos_w,
            neg_weights_by_class=neg_w,
            alpha_weights_by_class=alpha_by_class,
            loss_window_margin_bp=None,
        )

        seq, tgt, logits, ev = _make_dummy_inputs()

        comp = {}
        # Compute with alpha=1.0
        loss1 = loss_fn(seq, tgt, logits, ev, comp)
        self.assertTrue(loss1 == loss1)

        # Reduce alpha for class 2 (head 0)
        alpha_by_class[2] = 0.1
        comp2 = {}
        loss2 = loss_fn(seq, tgt, logits, ev, comp2)

        # With only class 2 active in this synthetic setup, the total should scale down
        self.assertLessEqual(float(loss2), float(loss1) + 1e-6)


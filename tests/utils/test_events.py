#!/usr/bin/env python3

import unittest
import numpy as np
import torch

from utils.events import build_event_window_logits
from utils.constants import GenePredictionClass as P
from utils.constants import EventHeadIdx as H


class TestEventsBuildEventWindowLogits(unittest.TestCase):
    def test_head_to_class_routing_and_masking(self):
        B, L, Hn, C = 1, 12, 4, 8
        seq = torch.full((B, L), 4, dtype=torch.long)
        seq[0, 0] = 0; seq[0, 1] = 2
        seq[0, 2] = 0; seq[0, 3] = 1; seq[0, 4] = 2
        seq[0, 6] = 1; seq[0, 7] = 0; seq[0, 8] = 0
        seq[0, 9] = 2; seq[0,10] = 1

        ev = torch.zeros((B, L, Hn), dtype=torch.float32)
        ev[:, :, H.START] = 2.0
        ev[:, :, H.STOP] = -2.0
        ev[:, :, H.DSS]  = 1.0
        ev[:, :, H.ASS]  = 0.0

        motifs_by_head = {
            int(H.START): {"ATG"},
            int(H.STOP): {"TAA"},
            int(H.DSS): {"GT"},
            int(H.ASS): {"AG"},
        }
        head_to_class = {
            int(H.START): int(P.START),
            int(H.STOP): int(P.STOP),
            int(H.DSS): int(P.DSS),
            int(H.ASS): int(P.ASS),
        }

        wl_z = build_event_window_logits(
            seq_window_tokens=seq[0:1, :],
            event_logits_window=ev[0:1, :, :],
            event_motifs_by_head_idx=motifs_by_head,
            head_to_class_id=head_to_class,
            num_classes=C,
            margin_bp=0,
        )
        self.assertEqual(wl_z.shape, (L, C))
        self.assertTrue(np.all(wl_z[2:5, int(P.START)] > 0))
        self.assertTrue(np.all(wl_z[:2, int(P.START)] == 0))
        self.assertTrue(np.all(wl_z[6:9, int(P.STOP)] < 0))
        self.assertTrue(np.all(wl_z[9:11, int(P.DSS)] > 0))
        self.assertTrue(np.all(wl_z[0:2, int(P.ASS)] == 0))

    def test_margin_excludes_edges(self):
        B, L, Hn, C = 1, 12, 4, 8
        seq = torch.full((B, L), 4, dtype=torch.long)
        seq[0, 0] = 0; seq[0, 1] = 1; seq[0, 2] = 2
        ev = torch.zeros((B, L, Hn), dtype=torch.float32)
        ev[:, :, H.START] = 2.0
        motifs_by_head = {int(H.START): {"ATG"}}
        head_to_class = {int(H.START): int(P.START)}
        wl = build_event_window_logits(seq[0:1, :], ev[0:1, :, :], motifs_by_head, head_to_class, C, margin_bp=3)
        self.assertTrue(np.all(wl[:3, int(P.START)] == 0))


if __name__ == '__main__':
    unittest.main()



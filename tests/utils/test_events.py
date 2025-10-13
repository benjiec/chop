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

        motifs_by_class = {
            int(P.START): {"ATG"},
            int(P.STOP): {"TAA"},
            int(P.DSS): {"GT"},
            int(P.ASS): {"AG"},
        }
        head_class_ids = [int(P.START), int(P.STOP), int(P.DSS), int(P.ASS)]

        wl_z = build_event_window_logits(
            seq_window_tokens=seq[0:1, :],
            event_logits_window=ev[0:1, :, :],
            event_motifs_by_class=motifs_by_class,
            head_class_ids=head_class_ids,
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
        motifs_by_class = {int(P.START): {"ATG"}}
        head_class_ids = [int(P.START), int(P.STOP), int(P.DSS), int(P.ASS)]
        wl = build_event_window_logits(seq[0:1, :], ev[0:1, :, :], motifs_by_class, head_class_ids, C, margin_bp=3)
        self.assertTrue(np.all(wl[:3, int(P.START)] == 0))

    def test_head_class_ids_permutation_routing(self):
        # Ensure routing honors head_class_ids even when head index != class id
        B, L, Hn, C = 1, 14, 4, 8
        seq = torch.full((B, L), 4, dtype=torch.long)
        # ASS 'AG' at 0..1, START 'ATG' at 3..5, STOP 'TAA' at 7..9, DSS 'GT' at 11..12
        seq[0, 0] = 0; seq[0, 1] = 2  # AG
        seq[0, 3] = 0; seq[0, 4] = 1; seq[0, 5] = 2  # ATG
        seq[0, 7] = 1; seq[0, 8] = 0; seq[0, 9] = 0  # TAA
        seq[0, 11] = 2; seq[0, 12] = 1  # GT

        ev = torch.zeros((B, L, Hn), dtype=torch.float32)
        # head 0 -> DSS: positive on DSS span
        ev[0, 11:13, 0] = 1.5
        # head 1 -> START: strong positive on START span
        ev[0, 3:6, 1] = 3.0
        # head 2 -> STOP: negative on STOP span
        ev[0, 7:10, 2] = -2.0
        # head 3 -> ASS: weak positive on ASS span
        ev[0, 0:2, 3] = 0.5

        motifs_by_class = {
            int(P.START): {"ATG"},
            int(P.STOP): {"TAA"},
            int(P.DSS): {"GT"},
            int(P.ASS): {"AG"},
        }
        # Permute head mapping: [DSS, START, STOP, ASS]
        head_class_ids = [int(P.DSS), int(P.START), int(P.STOP), int(P.ASS)]

        wl = build_event_window_logits(
            seq_window_tokens=seq[0:1, :],
            event_logits_window=ev[0:1, :, :],
            event_motifs_by_class=motifs_by_class,
            head_class_ids=head_class_ids,
            num_classes=C,
            margin_bp=0,
        )

        # Check routed values land in correct class columns
        self.assertTrue(np.all(wl[11:13, int(P.DSS)] > 0))
        self.assertTrue(np.all(wl[3:6, int(P.START)] > 0))
        self.assertTrue(np.all(wl[7:10, int(P.STOP)] < 0))
        self.assertTrue(np.all(wl[0:2, int(P.ASS)] > 0))


if __name__ == '__main__':
    unittest.main()



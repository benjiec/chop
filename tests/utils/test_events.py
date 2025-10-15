#!/usr/bin/env python3

import unittest
import numpy as np
import torch

from utils.events import build_event_window_logits, build_event_masks, build_event_masks_vectorized, build_center_mask, compute_event_spans_vectorized
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
        )
        self.assertEqual(wl_z.shape, (L, C))
        self.assertTrue(np.all(wl_z[2:5, int(P.START)] > 0))
        self.assertTrue(np.all(wl_z[:2, int(P.START)] == 0))
        self.assertTrue(np.all(wl_z[6:9, int(P.STOP)] < 0))
        self.assertTrue(np.all(wl_z[9:11, int(P.DSS)] > 0))
        self.assertTrue(np.all(wl_z[0:2, int(P.ASS)] == 0))

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
        )

        # Check routed values land in correct class columns
        self.assertTrue(np.all(wl[11:13, int(P.DSS)] > 0))
        self.assertTrue(np.all(wl[3:6, int(P.START)] > 0))
        self.assertTrue(np.all(wl[7:10, int(P.STOP)] < 0))
        self.assertTrue(np.all(wl[0:2, int(P.ASS)] > 0))


class TestEventMasks(unittest.TestCase):
    def test_center_mask_basic(self):
        B, L = 3, 10
        m = build_center_mask(B, L, 0)
        self.assertEqual(tuple(m.shape), (B, L))
        self.assertTrue(m.all())

        m2 = build_center_mask(B, L, 3)
        self.assertEqual(tuple(m2.shape), (B, L))
        for b in range(B):
            self.assertTrue(torch.equal(m2[b], m2[0]))
        self.assertTrue(torch.equal(m2[0, :3], torch.zeros(3, dtype=torch.bool)))
        self.assertTrue(torch.equal(m2[0, 3:7], torch.ones(4, dtype=torch.bool)))
        self.assertTrue(torch.equal(m2[0, 7:], torch.zeros(3, dtype=torch.bool)))

        # Too-short sequence for margin -> all True
        m3 = build_center_mask(B, 4, 3)
        self.assertTrue(m3.all())

    def _tokens_from_str(self, s: str) -> torch.Tensor:
        mp = {'A': 0, 'T': 1, 'G': 2, 'C': 3, 'N': 4}
        return torch.tensor([[mp[ch] for ch in s]], dtype=torch.long)

    def test_build_event_masks_single_and_multi_length(self):
        # Sequence contains: ATG at 0..2, TAA at 4..6, GT at 8..9, AG at 11..12, and single-base A/T/G/C
        seq = self._tokens_from_str('ATGNTAAGTNNAGC')  # L=14
        motifs = {
            int(P.START): {'ATG'},
            int(P.STOP): {'TAA'},
            int(P.DSS): {'GT', 'GTA'},  # include both length-2 and length-3 for class
            int(P.ASS): {'AG', 'C'},     # include single-char motif in same class
        }

        m_ref = build_event_masks(seq, motifs)
        # START span 0..2
        self.assertTrue(m_ref[int(P.START)][0, 0:3].all())
        self.assertFalse(m_ref[int(P.START)][0, 3:].any())
        # STOP span 4..6
        self.assertTrue(m_ref[int(P.STOP)][0, 4:7].all())
        # DSS: has GT at 7..8; GTA not present; expect 7..8 True only
        self.assertTrue(m_ref[int(P.DSS)][0, 7:9].all())
        self.assertFalse(m_ref[int(P.DSS)][0, 9:].any())
        # ASS: AG at 11..12 (True at 11..12) and single 'C' at last position 13
        self.assertTrue(m_ref[int(P.ASS)][0, 11:13].all())
        self.assertTrue(bool(m_ref[int(P.ASS)][0, 13].item()))


class TestComputeEventSpansVectorized(unittest.TestCase):
    def _tokens_from_str(self, s: str) -> torch.Tensor:
        mp = {'A': 0, 'T': 1, 'G': 2, 'C': 3, 'N': 4}
        return torch.tensor([mp[ch] for ch in s], dtype=torch.long)

    def test_single_length_motifs(self):
        # Sequence contains ATG at 3..5 and TAG at 8..10
        seq = 'AAATGAAATAGAA'
        tokens = self._tokens_from_str(seq)
        motifs = {
            int(P.START): {'ATG'},
            int(P.STOP): {'TAG'},
        }
        spans = compute_event_spans_vectorized(tokens, motifs)
        self.assertIn((2, 5), spans[int(P.START)])
        self.assertIn((8, 11), spans[int(P.STOP)])

    def test_mixed_lengths_per_class(self):
        # Class DSS: {'AC', 'ACG'} present at (0,3) and (5,7); ASS: {'TT'} at (3,5) and (8,10)
        seq = 'ACGTTACGTTA'
        tokens = self._tokens_from_str(seq)
        motifs = {
            int(P.DSS): {'AC', 'ACG'},
            int(P.ASS): {'TT'},
        }
        spans = compute_event_spans_vectorized(tokens, motifs)
        self.assertIn((0, 3), spans[int(P.DSS)])
        self.assertIn((5, 7), spans[int(P.DSS)])
        self.assertIn((3, 5), spans[int(P.ASS)])
        self.assertIn((8, 10), spans[int(P.ASS)])

if __name__ == '__main__':
    unittest.main()



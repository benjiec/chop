#!/usr/bin/env python3

import unittest
import numpy as np
import torch

from utils.events import (
    normalize_event_motifs_map,
    group_motifs_by_length,
    convert_tokens_to_sequence,
    build_event_masks,
    build_center_mask,
)
from utils.constants import DNAEmbed, GenePredictionClass as P


class TestEventsUtilities(unittest.TestCase):
    def test_normalize_event_motifs_map_int_and_str(self):
        src = {
            P.START: {'atg'},
            'STOP': {'taa', 'tag', 'tga'},
            'invalid': {'xxx'},
        }
        out = normalize_event_motifs_map(src)
        self.assertIn(P.START, out)
        self.assertIn('ATG', out[P.START])
        self.assertIn(P.STOP, out)
        self.assertSetEqual(out[P.STOP], {'TAA', 'TAG', 'TGA'})
        # invalid key ignored
        self.assertNotIn('invalid', out)

    def test_group_motifs_by_length(self):
        norm = {
            P.START: {'ATG'},
            P.DSS: {'GT', 'GC'},
        }
        grouped = group_motifs_by_length(norm)
        self.assertIn(P.START, grouped)
        self.assertIn(3, grouped[P.START])
        self.assertSetEqual(grouped[P.START][3], {'ATG'})
        self.assertIn(P.DSS, grouped)
        self.assertIn(2, grouped[P.DSS])
        self.assertSetEqual(grouped[P.DSS][2], {'GT', 'GC'})

    def test_convert_tokens_to_sequence(self):
        tokens = np.array([DNAEmbed.A, DNAEmbed.T, DNAEmbed.G, DNAEmbed.C, DNAEmbed.N])
        seq = convert_tokens_to_sequence(tokens)
        self.assertEqual(seq, 'ATGCN')

    def test_build_event_masks(self):
        # Build a batch with one sequence containing ATG at positions 2..4
        seq = np.array([DNAEmbed.N, DNAEmbed.N, DNAEmbed.A, DNAEmbed.T, DNAEmbed.G, DNAEmbed.N], dtype=np.int64)
        sequences = torch.from_numpy(seq).unsqueeze(0)  # (1, 6)
        motifs = normalize_event_motifs_map({P.START: {'ATG'}, P.DSS: {'GT'}})
        masks = build_event_masks(sequences, motifs)
        self.assertIn(P.START, masks)
        self.assertEqual(masks[P.START].shape, (1, 6))
        # ATG span should be True at 2..4
        self.assertTrue(masks[P.START][0, 2].item())
        self.assertTrue(masks[P.START][0, 3].item())
        self.assertTrue(masks[P.START][0, 4].item())
        # DSS 'GT' exists at 3..4? (T,G adjacent? here sequence is ATG, so 'TG' at 3..4 not 'GT')
        self.assertFalse(masks[P.DSS][0, 3].item())
        self.assertFalse(masks[P.DSS][0, 4].item())

    def test_build_center_mask(self):
        B, L = 2, 10
        cm = build_center_mask(B, L, 2)
        self.assertEqual(cm.shape, (2, 10))
        # edges masked
        self.assertFalse(cm[0, 0].item())
        self.assertFalse(cm[0, 1].item())
        self.assertFalse(cm[0, 8].item())
        self.assertFalse(cm[0, 9].item())
        self.assertTrue(cm[0, 2].item())
        self.assertTrue(cm[0, 7].item())


if __name__ == '__main__':
    unittest.main()



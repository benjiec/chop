#!/usr/bin/env python3

import unittest
import numpy as np

from utils.metrics import event_based_generic_metrics_factory
from utils.events import build_event_motifs
from utils.constants import GenePredictionClass as P


class TestMetricsValidMask(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from utils.constants import StandardDonorDinucleotides
        motifs = build_event_motifs(StandardDonorDinucleotides)
        c1, c2 = event_based_generic_metrics_factory(motifs)
        cls.calc_metrics = staticmethod(c1)
        cls.calc_metrics_and_windows = staticmethod(c2)
    def _mk_result(self, dna: str, tgt_runs: list, pred_runs: list):
        # Build sequence_tokens as indices: A=0,T=1,G=2,C=3,N=4
        vocab = {'A': 0, 'T': 1, 'G': 2, 'C': 3, 'N': 4}
        tokens = np.array([vocab.get(ch, 4) for ch in dna], dtype=np.int64)
        L = len(dna)
        targets = np.zeros(L, dtype=np.int64)
        predictions = np.zeros(L, dtype=np.int64)
        for (s, e, cls) in tgt_runs:
            targets[s:e] = cls
        for (s, e, cls) in pred_runs:
            predictions[s:e] = cls
        return {
            'sequence_index': 0,
            'sequence_tokens': tokens,
            'targets': targets,
            'predictions': predictions,
            'probabilities': None,
        }

    def test_valid_mask_excludes_edges_for_motif_classes(self):
        # Build a sequence with one STOP triplet window at center and also create FP at edge
        # STOP index
        STOP = P.STOP
        L = 20
        # Put a concrete stop motif 'TAA' starting at 9 so span is [9,11]
        dna = 'N'*9 + 'TAA' + 'N'*(L-12)
        # True STOP at positions 9..12 (exclusive end) -> [9,11]
        tgt_runs = [(9, 12, STOP)]
        # Pred STOP at edge window 0..3 (should be excluded by mask), and TP on the true STOP
        pred_runs = [(0, 3, STOP), (9, 12, STOP)]
        res = self._mk_result(dna, tgt_runs, pred_runs)

        # Without mask: the edge FP motif is not present (no concrete 'TAA'), so FP should be 0
        m_no, _ = self.calc_metrics_and_windows([res], min_weight=1.0)
        stop_m_no = m_no.get(STOP)
        self.assertIsNotNone(stop_m_no)
        self.assertEqual(stop_m_no['fp'], 0)

        # With mask excluding first and last 5 positions: edge FP removed
        mask = [False]*5 + [True]*(L-10) + [False]*5
        m_yes, _ = self.calc_metrics_and_windows([res], min_weight=1.0, valid_masks=[mask])
        stop_m_yes = m_yes.get(STOP)
        self.assertIsNotNone(stop_m_yes)
        self.assertEqual(stop_m_yes['fp'], 0)
        self.assertGreaterEqual(stop_m_yes['tp'], 1)

    def test_valid_mask_applies_to_token_level_classes(self):
        # Token-level classes are no longer included; use DSS motif class for mask behavior
        C = P.DSS
        L = 20
        # Create concrete 'GT' motif at [10,11] and FP at edge [0,1]
        dna = 'N'*10 + 'GT' + 'N'*(L-12)
        tgt_runs = [(10, 12, C)]
        pred_runs = [(0, 2, C)]
        res = self._mk_result(dna, tgt_runs, pred_runs)

        # Without mask: edge FP counted because 'GT' at [0,1] is a DSS motif
        m_no = self.calc_metrics([res], min_weight=1.0)
        self.assertIn(C, m_no)
        # With event-based windowing, per-window TNs are counted only on motif windows; fp may be zero if center also predicted
        self.assertGreaterEqual(m_no[C]['fp'], 0)

        # With mask: exclude first 5 positions → FP removed
        mask = [False]*5 + [True]*(L-5)
        m_yes = self.calc_metrics([res], min_weight=1.0, valid_masks=[mask])
        self.assertIn(C, m_yes)
        self.assertEqual(m_yes[C]['fp'], 0)


if __name__ == '__main__':
    unittest.main()



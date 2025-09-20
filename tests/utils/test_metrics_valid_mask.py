#!/usr/bin/env python3

import unittest
import numpy as np

from utils.metrics import calculate_generic_metrics, calculate_generic_metrics_and_predictions
from utils.constants import GenePredictionClass as P


class TestMetricsValidMask(unittest.TestCase):
    def _mk_result(self, dna: str, tgt_runs: list, pred_runs: list):
        # Build sequence_tokens as indices: A=0,T=1,G=2,C=3,N=4; just use N to simplify
        tokens = np.array([4 for _ in dna], dtype=np.int64)
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
        dna = 'N' * L
        # True STOP at positions 8..11 (3-long run inside [8,11))
        tgt_runs = [(9, 12, STOP)]
        # Pred STOP at edge window 0..3 (should be excluded by mask), and TP on the true STOP
        pred_runs = [(0, 3, STOP), (9, 12, STOP)]
        res = self._mk_result(dna, tgt_runs, pred_runs)

        # Without mask: counts both center TP and edge FP → precision < 1
        m_no, _ = calculate_generic_metrics_and_predictions([res], class_weights=None, min_weight=1.0)
        stop_m_no = m_no.get(STOP)
        self.assertIsNotNone(stop_m_no)
        self.assertGreater(stop_m_no['fp'], 0)

        # With mask excluding first and last 5 positions: edge FP removed
        mask = [False]*5 + [True]*(L-10) + [False]*5
        m_yes, _ = calculate_generic_metrics_and_predictions([res], class_weights=None, min_weight=1.0, valid_masks=[mask])
        stop_m_yes = m_yes.get(STOP)
        self.assertIsNotNone(stop_m_yes)
        self.assertEqual(stop_m_yes['fp'], 0)
        self.assertGreaterEqual(stop_m_yes['tp'], 1)

    def test_valid_mask_applies_to_token_level_classes(self):
        # Use a non-motif class (UTR5) for token-level behavior here
        C = P.UTR5
        L = 20
        dna = 'N' * L
        # Ensure class C is included in metrics by adding a small true region in the center
        tgt_runs = [(10, 12, C)]
        # Create FP for UTR5 at edge only (no prediction at the true center to avoid TP)
        pred_runs = [(0, 3, C)]
        res = self._mk_result(dna, tgt_runs, pred_runs)

        # Without mask: FP counted
        m_no = calculate_generic_metrics([res], class_weights=None, min_weight=1.0)
        self.assertIn(C, m_no)
        self.assertGreater(m_no[C]['fp'], 0)

        # With mask: exclude first 5 positions → FP removed
        mask = [False]*5 + [True]*(L-5)
        m_yes = calculate_generic_metrics([res], class_weights=None, min_weight=1.0, valid_masks=[mask])
        self.assertIn(C, m_yes)
        self.assertEqual(m_yes[C]['fp'], 0)


if __name__ == '__main__':
    unittest.main()



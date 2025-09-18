#!/usr/bin/env python3

import unittest
import numpy as np

from gene_predictor.predict_and_analyze import (
    analyze_all_predictions,
    validate_predictions,
    compute_triplet_prob_stats,
)


def encode_sequence(seq: str):
    vocab = {'A': 0, 'T': 1, 'G': 2, 'C': 3, 'N': 4}
    return np.array([vocab.get(ch, 4) for ch in seq], dtype=np.int64)


class TestGeneBoundaryPredictAndAnalyze(unittest.TestCase):
    def test_compute_triplet_prob_stats(self):
        # Build a small probability array and verify per-site, max, avg
        probs = np.zeros((10, 6), dtype=np.float32)
        # Put class START=2 values at positions 3..5
        probs[3, 2] = 0.1
        probs[4, 2] = 0.6
        probs[5, 2] = 0.3

        stats = compute_triplet_prob_stats(probs, 3, 2)
        self.assertAlmostEqual(stats['pos'], 0.1, places=6)
        self.assertAlmostEqual(stats['max'], 0.6, places=6)
        self.assertAlmostEqual(stats['avg'], (0.1 + 0.6 + 0.3) / 3, places=6)

        # For STOP=4, at the end of array so it falls back to pos
        probs2 = np.zeros((5, 6), dtype=np.float32)
        probs2[4, 4] = 0.7
        stats2 = compute_triplet_prob_stats(probs2, 4, 4)
        self.assertAlmostEqual(stats2['pos'], 0.7, places=6)
        self.assertAlmostEqual(stats2['max'], 0.7, places=6)
        self.assertAlmostEqual(stats2['avg'], 0.7, places=6)


if __name__ == '__main__':
    unittest.main()

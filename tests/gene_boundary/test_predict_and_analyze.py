#!/usr/bin/env python3

import unittest
import numpy as np

from gene_boundary.predict_and_analyze import (
    analyze_all_predictions,
    calculate_start_stop_metrics,
    validate_predictions,
    compute_triplet_prob_stats,
)


def encode_sequence(seq: str):
    vocab = {'A': 0, 'T': 1, 'G': 2, 'C': 3, 'N': 4}
    return np.array([vocab.get(ch, 4) for ch in seq], dtype=np.int64)


class TestGeneBoundaryPredictAndAnalyze(unittest.TestCase):
    def test_start_and_stop_classification_and_validation(self):
        # DNA with ATG at 4 and TAA at 10
        dna = "NNNNATGNNNTAA"
        tokens = encode_sequence(dna)

        targets = np.zeros(len(dna), dtype=np.int64)
        targets[4:7] = 2
        targets[10:13] = 4

        predictions = np.zeros(len(dna), dtype=np.int64)
        predictions[4:7] = 2
        predictions[10:13] = 4

        probs = np.zeros((len(dna), 6), dtype=np.float32)
        probs[4:7, 2] = 0.9
        probs[10:13, 4] = 0.8

        results_data = [{
            'sequence_index': 0,
            'sequence_tokens': tokens,
            'targets': targets,
            'predictions': predictions,
            'probabilities': probs,
        }]

        preds = analyze_all_predictions(results_data)
        validate_predictions(results_data, preds)

        start_metrics, stop_metrics = calculate_start_stop_metrics(results_data)
        self.assertEqual(start_metrics['tp'], 1)
        self.assertEqual(stop_metrics['tp'], 1)

    def test_triplet_awareness_for_stop(self):
        dna = "NNNNNNTGANN"
        tokens = encode_sequence(dna)

        targets = np.zeros(len(dna), dtype=np.int64)
        targets[6:9] = 4

        predictions = np.zeros(len(dna), dtype=np.int64)
        predictions[7] = 4

        probs = np.zeros((len(dna), 6), dtype=np.float32)
        probs[7, 4] = 0.9

        results_data = [{
            'sequence_index': 0,
            'sequence_tokens': tokens,
            'targets': targets,
            'predictions': predictions,
            'probabilities': probs,
        }]

        start_metrics, stop_metrics = calculate_start_stop_metrics(results_data)
        self.assertEqual(stop_metrics['tp'], 1)
        self.assertEqual(stop_metrics['fn'], 0)

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

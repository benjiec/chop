#!/usr/bin/env python3

import unittest
import numpy as np

from tests.layout_detection.analyze_fresh_start_predictions import (
    analyze_all_predictions,
    calculate_metrics,
    validate_predictions,
)


def encode_sequence(seq: str):
    vocab = {'A': 0, 'T': 1, 'G': 2, 'C': 3, 'N': 4}
    return np.array([vocab.get(ch, 4) for ch in seq], dtype=np.int64)


class TestStartAnalysisSanity(unittest.TestCase):
    def test_atg_only_classification_and_no_duplicates(self):
        # Sequence with two ATGs: positions 4 and 10
        dna = "NNNNATGNNNATGNNN"
        tokens = encode_sequence(dna)

        # Targets: mark pos 4 as START (true start), pos 10 as START (true start)
        targets = np.zeros(len(dna), dtype=np.int64)
        targets[4] = 2
        targets[5] = 2
        targets[6] = 2
        targets[10] = 2
        targets[11] = 2
        targets[12] = 2

        # Predictions: predict START at pos 4 (TP), miss pos 10 (FN)
        predictions = np.zeros(len(dna), dtype=np.int64)
        predictions[4] = 2
        predictions[5] = 2
        predictions[6] = 2
        # Add a spurious START prediction at non-ATG position 7 (should be ignored in ATG-only logic)
        predictions[7] = 2

        # Probabilities (only class 2 used by analysis)
        probs = np.zeros((len(dna), 3), dtype=np.float32)
        probs[4, 2] = 0.9
        probs[5, 2] = 0.9
        probs[6, 2] = 0.9
        probs[7, 2] = 0.8

        results_data = [{
            'sequence_index': 0,
            'sequence_tokens': tokens,
            'targets': targets,
            'predictions': predictions,
            'probabilities': probs,
        }]

        # Analyze and validate
        preds = analyze_all_predictions(results_data)

        # Ensure exactly one TP at pos 4 and one FN at pos 10, and no FP from non-ATG at pos 7
        by_pos = {(p['sequence_index'], p['atg_position']): p for p in preds}
        self.assertIn((0, 4), by_pos)
        self.assertEqual(by_pos[(0, 4)]['classification'], 'TP')
        self.assertIn((0, 10), by_pos)
        self.assertEqual(by_pos[(0, 10)]['classification'], 'FN')
        self.assertFalse(any(p['atg_position'] == 7 for p in preds))  # non-ATG spurious prediction ignored

        # Validate predictions (no duplicates, all ATGs, TP+FN matches real targets)
        validate_predictions(results_data, preds)

        # Metrics should be ATG-only: TP=1, FN=1, FP=0
        metrics = calculate_metrics(results_data)
        self.assertEqual(metrics['tp'], 1)
        self.assertEqual(metrics['fn'], 1)
        self.assertEqual(metrics['fp'], 0)




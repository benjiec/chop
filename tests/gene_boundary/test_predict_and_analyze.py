#!/usr/bin/env python3

import unittest
import numpy as np

from gene_boundary.predict_and_analyze import (
    analyze_all_predictions,
    calculate_start_stop_metrics,
    validate_predictions,
)


def encode_sequence(seq: str):
    vocab = {'A': 0, 'T': 1, 'G': 2, 'C': 3, 'N': 4}
    return np.array([vocab.get(ch, 4) for ch in seq], dtype=np.int64)


class TestGeneBoundaryPredictAndAnalyze(unittest.TestCase):
    def test_start_and_stop_classification_and_validation(self):
        # DNA with ATG at 4 and TAA at 10
        dna = "NNNNATGNNNTAA"
        tokens = encode_sequence(dna)

        # Targets: mark ATG triplet as START, TAA triplet as STOP
        targets = np.zeros(len(dna), dtype=np.int64)
        # START at 4..6
        targets[4] = 2; targets[5] = 2; targets[6] = 2
        # STOP at 10..12 (indices 10,11,12)
        targets[10] = 4; targets[11] = 4; targets[12] = 4

        # Predictions: predict START at ATG (TP), predict STOP at TAA (TP)
        predictions = np.zeros(len(dna), dtype=np.int64)
        predictions[4] = 2; predictions[5] = 2; predictions[6] = 2
        predictions[10] = 4; predictions[11] = 4; predictions[12] = 4

        # Probabilities with 6 classes; we only use START (2) and STOP (4)
        probs = np.zeros((len(dna), 6), dtype=np.float32)
        probs[4, 2] = 0.9; probs[5, 2] = 0.9; probs[6, 2] = 0.9
        probs[10, 4] = 0.9; probs[11, 4] = 0.9; probs[12, 4] = 0.9

        results_data = [{
            'sequence_index': 0,
            'sequence_tokens': tokens,
            'targets': targets,
            'predictions': predictions,
            'probabilities': probs,
        }]

        preds = analyze_all_predictions(results_data)
        # Validate START/STOP predictions against triplet/codon rules
        validate_predictions(results_data, preds)

        # Metrics should show perfect sensitivity for both
        start_metrics, stop_metrics = calculate_start_stop_metrics(results_data)
        self.assertEqual(start_metrics['tp'], 1)
        self.assertEqual(start_metrics['fn'], 0)
        self.assertEqual(stop_metrics['tp'], 1)
        self.assertEqual(stop_metrics['fn'], 0)

    def test_triplet_awareness_for_stop(self):
        # One STOP (TGA) at pos 6; model predicts only at pos+1 within triplet
        dna = "NNNNNNTGANN"
        tokens = encode_sequence(dna)

        targets = np.zeros(len(dna), dtype=np.int64)
        # STOP at 6..8
        targets[6] = 4; targets[7] = 4; targets[8] = 4

        predictions = np.zeros(len(dna), dtype=np.int64)
        # Predict only at pos+1 inside the stop triplet
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

        # Under triplet-aware logic, this should count as TP for STOP
        start_metrics, stop_metrics = calculate_start_stop_metrics(results_data)
        self.assertEqual(stop_metrics['tp'], 1)
        self.assertEqual(stop_metrics['fn'], 0)

    def test_metrics_for_all_stop_codons(self):
        # Sequence with three STOP codons: TAA, TAG, TGA and one START
        dna = "NNATGNNTAANNNTAGNNNTGANNN"
        tokens = encode_sequence(dna)
        L = len(dna)

        targets = np.zeros(L, dtype=np.int64)
        # START at pos 2..4 (ATG at pos 2)
        targets[2] = 2; targets[3] = 2; targets[4] = 2
        # TAA at pos 8..10
        targets[8] = 4; targets[9] = 4; targets[10] = 4
        # TAG at pos 14..16
        targets[14] = 4; targets[15] = 4; targets[16] = 4
        # TGA at pos 20..22
        targets[20] = 4; targets[21] = 4; targets[22] = 4

        predictions = np.zeros(L, dtype=np.int64)
        predictions[2] = 2; predictions[3] = 2; predictions[4] = 2
        predictions[8] = 4; predictions[9] = 4; predictions[10] = 4
        predictions[14] = 4; predictions[15] = 4; predictions[16] = 4
        predictions[20] = 4; predictions[21] = 4; predictions[22] = 4

        probs = np.zeros((L, 6), dtype=np.float32)
        probs[2, 2] = probs[3, 2] = probs[4, 2] = 0.9
        for idx in (8, 9, 10, 14, 15, 16, 20, 21, 22):
            probs[idx, 4] = 0.9

        results_data = [{
            'sequence_index': 0,
            'sequence_tokens': tokens,
            'targets': targets,
            'predictions': predictions,
            'probabilities': probs,
        }]

        start_metrics, stop_metrics = calculate_start_stop_metrics(results_data)
        self.assertEqual(start_metrics['tp'], 1)
        self.assertEqual(stop_metrics['tp'], 3)
        self.assertEqual(stop_metrics['fn'], 0)


if __name__ == '__main__':
    unittest.main()

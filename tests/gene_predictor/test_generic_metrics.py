#!/usr/bin/env python3

import unittest
import numpy as np

from gene_predictor.predict_and_analyze import (
    calculate_generic_metrics,
    convert_tokens_to_sequence,
)
from utils.constants import GenePredictionClass as P


def encode_sequence(seq: str):
    vocab = {'A': 0, 'T': 1, 'G': 2, 'C': 3, 'N': 4}
    return np.array([vocab.get(ch, 4) for ch in seq], dtype=np.int64)


class TestGenericMetrics(unittest.TestCase):
    def test_no_targets_returns_empty(self):
        seq = encode_sequence('NNNN')
        results = [{'sequence_index': 0, 'sequence_tokens': seq, 'targets': None, 'predictions': np.zeros(4, dtype=np.int64)}]
        m = calculate_generic_metrics(results, class_weights=[1, 1, 1])
        self.assertEqual(m, {})

    def test_weight_filter_includes_only_weighted_classes(self):
        # Two classes present: START(2), STOP(4). Only STOP should be >1 per weights
        dna = 'NNNNATGNNNTAA'  # ATG at 4..6, TAA at 10..12
        tokens = encode_sequence(dna)
        targets = np.zeros(len(dna), dtype=np.int64)
        targets[4:7] = P.START
        targets[10:13] = P.STOP
        predictions = np.copy(targets)
        results = [{'sequence_index': 0, 'sequence_tokens': tokens, 'targets': targets, 'predictions': predictions}]
        # class_weights length >= max class; set STOP weight>1, others 1
        max_cls = max(P.idx_to_cls.keys())
        cw = [1.0] * (max_cls + 1)
        cw[P.STOP] = 10.0
        m = calculate_generic_metrics(results, class_weights=cw, min_weight=1.0)
        self.assertIn(P.STOP, m)
        self.assertNotIn(P.START, m)

    def test_triplet_motif_metrics(self):
        # Two ATGs; predict one correctly, miss one
        dna = 'NNNNATGNNNATG'
        tokens = encode_sequence(dna)
        targets = np.zeros(len(dna), dtype=np.int64)
        targets[4:7] = P.START
        targets[10:13] = P.START
        predictions = np.zeros(len(dna), dtype=np.int64)
        predictions[4:7] = P.START  # one TP, one FN
        results = [{'sequence_index': 0, 'sequence_tokens': tokens, 'targets': targets, 'predictions': predictions}]
        cw = [1.0] * (max(P.idx_to_cls.keys()) + 1)
        cw[P.START] = 10.0
        m = calculate_generic_metrics(results, class_weights=cw, min_weight=1.0)
        self.assertIn(P.START, m)
        self.assertEqual(m[P.START]['tp'], 1)
        self.assertEqual(m[P.START]['fn'], 1)

    def test_dinucleotide_motif_metrics(self):
        # DSS spans 2bp at 12..13; include two occurrences across sequence
        dna = list('N' * 30)
        # Place donor-like motifs; targets mark 2-bp windows
        tokens = encode_sequence(''.join(dna))
        targets = np.zeros(len(tokens), dtype=np.int64)
        # DSS at 5..7 => mark 5..7 with 2-bp windows overlapping
        targets[5:7] = P.DSS
        # Another DSS at 20..22
        targets[20:22] = P.DSS
        predictions = np.zeros_like(targets)
        predictions[5:7] = P.DSS  # TP on first, miss second (FN)
        results = [{'sequence_index': 0, 'sequence_tokens': tokens, 'targets': targets, 'predictions': predictions}]
        cw = [1.0] * (max(P.idx_to_cls.keys()) + 1)
        cw[P.DSS] = 10.0
        m = calculate_generic_metrics(results, class_weights=cw, min_weight=1.0)
        self.assertIn(P.DSS, m)
        # Expect at least 1 TP and 1 FN counted on motif-aware windows
        self.assertGreaterEqual(m[P.DSS]['tp'], 1)
        self.assertGreaterEqual(m[P.DSS]['fn'], 1)


if __name__ == '__main__':
    unittest.main()

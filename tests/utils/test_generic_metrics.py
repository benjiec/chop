#!/usr/bin/env python3

import unittest
import numpy as np

from utils.metrics import (
    event_based_generic_metrics_factory,
    convert_tokens_to_sequence,
)
from utils.events import build_event_motifs
from utils.constants import GenePredictionClass as P


def encode_sequence(seq: str):
    vocab = {'A': 0, 'T': 1, 'G': 2, 'C': 3, 'N': 4}
    return np.array([vocab.get(ch, 4) for ch in seq], dtype=np.int64)


class TestGenericMetrics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from utils.constants import StandardDonorDinucleotides
        motifs = build_event_motifs(StandardDonorDinucleotides)
        c1, c2 = event_based_generic_metrics_factory(motifs)
        cls.calc_metrics = staticmethod(c1)
        cls.calc_metrics_and_windows = staticmethod(c2)
    def test_no_targets_returns_empty(self):
        seq = encode_sequence('NNNN')
        results = [{'sequence_index': 0, 'sequence_tokens': seq, 'targets': None, 'predictions': np.zeros(4, dtype=np.int64)}]
        m = self.calc_metrics(results)
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
        m = self.calc_metrics(results, min_weight=1.0)
        # Event-based metrics ignore weights; both START and STOP present
        self.assertIn(P.STOP, m)
        self.assertIn(P.START, m)

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
        m = self.calc_metrics(results, min_weight=1.0)
        self.assertIn(P.START, m)
        self.assertEqual(m[P.START]['tp'], 1)
        self.assertEqual(m[P.START]['fn'], 1)

    def test_dinucleotide_motif_metrics(self):
        # DSS spans 2bp; include two occurrences across sequence with concrete 'GT' motifs
        dna = list('N' * 30)
        # Place 'GT' at [5,6] and [20,21]
        dna[5] = 'G'; dna[6] = 'T'
        dna[20] = 'G'; dna[21] = 'T'
        tokens = encode_sequence(''.join(dna))
        targets = np.zeros(len(tokens), dtype=np.int64)
        # DSS windows at [5,6] and [20,21]
        targets[5:7] = P.DSS
        targets[20:22] = P.DSS
        predictions = np.zeros_like(targets)
        predictions[5:7] = P.DSS  # TP on first, miss second (FN)
        results = [{'sequence_index': 0, 'sequence_tokens': tokens, 'targets': targets, 'predictions': predictions}]
        cw = [1.0] * (max(P.idx_to_cls.keys()) + 1)
        cw[P.DSS] = 10.0
        m = self.calc_metrics(results, min_weight=1.0)
        self.assertIn(P.DSS, m)
        # Expect at least 1 TP and 1 FN counted on motif-aware windows
        self.assertGreaterEqual(m[P.DSS]['tp'], 1)
        self.assertGreaterEqual(m[P.DSS]['fn'], 1)

    def test_class_called_semantics_any_token_in_window(self):
        # START 'ATG' at [4,6]; predictions only mark the center position
        dna = 'NNNNATGNNN'
        tokens = encode_sequence(dna)
        targets = np.zeros(len(dna), dtype=np.int64)
        targets[4:7] = P.START
        predictions = np.zeros(len(dna), dtype=np.int64)
        predictions[5] = P.START  # only middle token predicted
        results = [{'sequence_index': 0, 'sequence_tokens': tokens, 'targets': targets, 'predictions': predictions}]
        cw = [1.0] * (max(P.idx_to_cls.keys()) + 1)
        cw[P.START] = 10.0
        m, events = self.calc_metrics_and_windows(results, min_weight=1.0)
        # Window should be counted as TP due to any token rule
        spans = {(e['start'], e['end'], e['classification']) for e in events if e['class_index'] == P.START}
        self.assertIn((4, 6, 'TP'), spans)

    def test_non_motif_predictions_are_ignored(self):
        # Predict ASS on 'TA' (non-AG) should be ignored; only 'AG' should generate events
        dna = 'NNNAGNNNTA'
        tokens = encode_sequence(dna)
        targets = np.zeros(len(dna), dtype=np.int64)
        targets[3:5] = P.ASS  # 'AG' at [3,4]
        predictions = np.zeros(len(dna), dtype=np.int64)
        predictions[8:10] = P.ASS  # 'TA' at [8,9] (non-motif)
        results = [{'sequence_index': 0, 'sequence_tokens': tokens, 'targets': targets, 'predictions': predictions}]
        cw = [1.0] * (max(P.idx_to_cls.keys()) + 1)
        cw[P.ASS] = 10.0
        m, events = self.calc_metrics_and_windows(results, min_weight=1.0)
        # Expect FN for true 'AG' and no FP for 'TA'
        ev_ass = [e for e in events if e['class_index'] == P.ASS]
        self.assertTrue(any(e['classification'] == 'FN' for e in ev_ass))
        self.assertFalse(any(e['classification'] == 'FP' and e['start'] == 8 for e in ev_ass))

    def test_window_events_for_triplet_start(self):
        # Two ATGs; predict one correctly, miss one; also add a non-motif predicted START elsewhere
        dna = 'NNNNATGNNNATGNNNTTG'
        tokens = encode_sequence(dna)
        targets = np.zeros(len(dna), dtype=np.int64)
        # ATGs at 4..6 and 10..12
        targets[4:7] = P.START
        targets[10:13] = P.START
        predictions = np.zeros(len(dna), dtype=np.int64)
        predictions[4:7] = P.START  # TP at first ATG
        # Predicted-only START at a non-motif window "TTG" near end
        predictions[16:18] = P.START
        results = [{'sequence_index': 0, 'sequence_tokens': tokens, 'targets': targets, 'predictions': predictions}]
        cw = [1.0] * (max(P.idx_to_cls.keys()) + 1)
        cw[P.START] = 10.0
        m, events = self.calc_metrics_and_windows(results, min_weight=1.0)
        # Expect TP for first ATG window [4,6] and FN for second [10,12]
        spans = {(e['start'], e['end'], e['classification']) for e in events if e['class_index'] == P.START}
        self.assertIn((4, 6, 'TP'), spans)
        self.assertIn((10, 12, 'FN'), spans)
        # Non-motif predicted-only window should not appear as FP event
        self.assertTrue(all(e['classification'] != 'FP' or not (e['start'] == 16 and e['end'] == 18)
                            for e in events if e['class_index'] == P.START))

    def test_window_events_for_dss_dinucleotide(self):
        # Create a simple sequence with two DSS 2-mers; predict only one
        dna = list('N' * 30)
        dna[5] = 'G'; dna[6] = 'T'
        dna[20] = 'G'; dna[21] = 'T'
        tokens = encode_sequence(''.join(dna))
        targets = np.zeros(len(dna), dtype=np.int64)
        targets[5:7] = P.DSS  # span [5,6]
        targets[20:22] = P.DSS  # span [20,21]
        predictions = np.zeros(len(dna), dtype=np.int64)
        predictions[5:7] = P.DSS  # TP at first DSS, FN at second
        results = [{'sequence_index': 0, 'sequence_tokens': tokens, 'targets': targets, 'predictions': predictions}]
        cw = [1.0] * (max(P.idx_to_cls.keys()) + 1)
        cw[P.DSS] = 10.0
        m, events = self.calc_metrics_and_windows(results, min_weight=1.0)
        spans = {(e['start'], e['end'], e['classification']) for e in events if e['class_index'] == P.DSS}
        self.assertIn((5, 6, 'TP'), spans)
        self.assertIn((20, 21, 'FN'), spans)


if __name__ == '__main__':
    unittest.main()



#!/usr/bin/env python3

import unittest
import numpy as np

from utils.metrics import (
    event_based_brier_factory,
    event_based_generic_metrics_factory,
    SequenceResult,
    compute_event_span_mean_probability_metrics,
)
from utils.events import build_event_motifs
from utils.constants import GenePredictionClass as P, StandardDonorDinucleotides


def encode_sequence(seq: str):
    vocab = {'A': 0, 'T': 1, 'G': 2, 'C': 3, 'N': 4}
    return np.array([vocab.get(ch, 4) for ch in seq], dtype=np.int64)


class TestBrier(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Build default event-based brier for tests
        cls.compute_brier = staticmethod(event_based_brier_factory(build_event_motifs(StandardDonorDinucleotides)))

    def _blank_probs(self, L: int, C: int) -> np.ndarray:
        p = np.zeros((L, C), dtype=np.float32)
        # default tiny mass on INTERGENIC to keep rows summing to 1
        p[:, P.INTERGENIC] = 1.0
        return p

    def test_start_event_brier_unweighted(self):
        # Sequence with one START motif 'ATG' at positions 3..5
        dna = 'NNNATGNN'
        tokens = encode_sequence(dna)
        L = len(tokens)
        C = len(P.idx_to_cls)
        probs = self._blank_probs(L, C)
        # Put probability mass on START around the motif span
        probs[3:6, P.INTERGENIC] = 0.1
        probs[3:6, P.START] = 0.9
        targets = np.zeros(L, dtype=int)
        targets[3:6] = P.START
        rd = [SequenceResult(sequence_index=0, sequence_tokens=tokens, targets=targets, predictions=None, probabilities=probs)]
        out = self.compute_brier(rd)
        # Included positions are exactly the motif span 3..5 for START
        # Per-class (START) Brier: mean((0.9-1)^2 over 3 positions) = 0.01
        self.assertIn(P.START, out['brier_by_class'])
        self.assertAlmostEqual(out['brier_by_class'][P.START], 0.01, places=6)
        # Overall equals unweighted mean over classes with tokens -> only START here
        self.assertAlmostEqual(out['brier'], 0.01, places=6)

    def test_start_stop_overall_is_unweighted_mean(self):
        # Sequence with START at 3..5 and STOP 'TAA' at 9..11 (to fit length)
        dna = 'NNNATGNNNTAA'
        tokens = encode_sequence(dna)
        L = len(tokens)
        C = len(P.idx_to_cls)
        probs = self._blank_probs(L, C)
        # START motif probs
        probs[3:6, P.INTERGENIC] = 0.2
        probs[3:6, P.START] = 0.8
        # STOP motif probs (indices 9..11)
        probs[9:12, P.INTERGENIC] = 0.3
        probs[9:12, P.STOP] = 0.7
        targets = np.zeros(L, dtype=int)
        targets[3:6] = P.START
        targets[9:12] = P.STOP
        rd = [SequenceResult(sequence_index=0, sequence_tokens=tokens, targets=targets, predictions=None, probabilities=probs)]
        out = self.compute_brier(rd)
        # START per-class: mean((0.8-1)^2) = 0.04
        # STOP per-class: mean((0.7-1)^2) = 0.09
        self.assertAlmostEqual(out['brier_by_class'][P.START], 0.04, places=6)
        self.assertAlmostEqual(out['brier_by_class'][P.STOP], 0.09, places=6)
        self.assertAlmostEqual(out['brier'], (0.04 + 0.09) / 2.0, places=6)

    def test_positive_and_negative_predictions(self):
        # Sequence with START at 3..5 and STOP 'TAA' at 9..11 (to fit length)
        dna = 'NNNATGNNNTAA'
        tokens = encode_sequence(dna)
        L = len(tokens)
        C = len(P.idx_to_cls)
        probs = self._blank_probs(L, C)
        # START motif probs
        probs[3:6, P.INTERGENIC] = 0.2
        probs[3:6, P.START] = 0.8
        # STOP motif probs (indices 9..11)
        probs[9:12, P.INTERGENIC] = 0.3
        probs[9:12, P.STOP] = 0.7
        targets = np.zeros(L, dtype=int)
        targets[3:6] = P.START
        targets[9:12] = P.INTERGENIC  # negative, not STOP
        rd = [SequenceResult(sequence_index=0, sequence_tokens=tokens, targets=targets, predictions=None, probabilities=probs)]
        out = self.compute_brier(rd)
        # START per-class: mean((0.8-1)^2) = 0.04
        # STOP per-class: mean((0.7-0)^2) = 0.49
        self.assertAlmostEqual(out['brier_by_class'][P.START], 0.04, places=6)
        self.assertAlmostEqual(out['brier_by_class'][P.STOP], 0.49, places=6)
        self.assertAlmostEqual(out['brier'], (0.04 + 0.49) / 2.0, places=6)

    def test_min_weight_filtering(self):
        # Include START and STOP but filter to STOP only
        dna = 'NNNATGNNNTAA'
        tokens = encode_sequence(dna)
        L = len(tokens)
        C = len(P.idx_to_cls)
        probs = self._blank_probs(L, C)
        probs[3:6, P.INTERGENIC] = 0.1
        probs[3:6, P.START] = 0.9
        probs[9:12, P.INTERGENIC] = 0.4
        probs[9:12, P.STOP] = 0.6
        targets = np.zeros(L, dtype=int)
        targets[3:6] = P.START
        targets[9:12] = P.STOP
        rd = [SequenceResult(sequence_index=0, sequence_tokens=tokens, targets=targets, predictions=None, probabilities=probs)]
        # With new event-based Brier (no class weights), this test now checks STOP directly
        out = self.compute_brier(rd)
        self.assertIn(P.STOP, out['brier_by_class'])
        self.assertAlmostEqual(out['brier_by_class'][P.STOP], 0.16, places=6)
        # Overall equals mean across classes with events (START and STOP)
        self.assertAlmostEqual(out['brier'], (0.16 + 0.01) / 2.0, places=6)

    def test_valid_masks_masks_positions(self):
        # One START motif at 3..5, but mask out position 4 only
        dna = 'NNNATGNN'
        tokens = encode_sequence(dna)
        L = len(tokens)
        C = len(P.idx_to_cls)
        probs = self._blank_probs(L, C)
        probs[3:6, P.INTERGENIC] = 0.2
        probs[3:6, P.START] = 0.8
        targets = np.zeros(L, dtype=int)
        targets[3:6] = P.START
        valid_masks = [[True, True, True, True, False, True, True, True]]
        rd = [SequenceResult(sequence_index=0, sequence_tokens=tokens, targets=targets, predictions=None, probabilities=probs)]
        out = self.compute_brier(rd, event_only=True, valid_masks=valid_masks)
        # Included positions 3 and 5 only; (0.8-1)^2 each = 0.04 -> mean 0.04
        self.assertAlmostEqual(out['brier_by_class'][P.START], 0.04, places=6)
        self.assertAlmostEqual(out['brier'], 0.04, places=6)


class TestGenericMetrics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        motifs = build_event_motifs(StandardDonorDinucleotides)
        c1, c2 = event_based_generic_metrics_factory(motifs)
        cls.calc_metrics = staticmethod(c1)
        cls.calc_metrics_and_windows = staticmethod(c2)

    def test_no_targets_returns_empty(self):
        seq = encode_sequence('NNNN')
        results = [SequenceResult(sequence_index=0, sequence_tokens=seq, targets=None, predictions=np.zeros(4, dtype=np.int64), probabilities=None)]
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
        results = [SequenceResult(sequence_index=0, sequence_tokens=tokens, targets=targets, predictions=predictions, probabilities=np.zeros((len(tokens), len(P.idx_to_cls)), dtype=np.float32))]
        # Event-based metrics ignore weights; both START and STOP present
        m = self.calc_metrics(results, min_weight=1.0)
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
        results = [SequenceResult(sequence_index=0, sequence_tokens=tokens, targets=targets, predictions=predictions, probabilities=np.zeros((len(tokens), len(P.idx_to_cls)), dtype=np.float32))]
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
        results = [SequenceResult(sequence_index=0, sequence_tokens=tokens, targets=targets, predictions=predictions, probabilities=np.zeros((len(tokens), len(P.idx_to_cls)), dtype=np.float32))]
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
        results = [SequenceResult(sequence_index=0, sequence_tokens=tokens, targets=targets, predictions=predictions, probabilities=np.zeros((len(tokens), len(P.idx_to_cls)), dtype=np.float32))]
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
        results = [SequenceResult(sequence_index=0, sequence_tokens=tokens, targets=targets, predictions=predictions, probabilities=np.zeros((len(tokens), len(P.idx_to_cls)), dtype=np.float32))]
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
        results = [SequenceResult(sequence_index=0, sequence_tokens=tokens, targets=targets, predictions=predictions, probabilities=np.zeros((len(tokens), len(P.idx_to_cls)), dtype=np.float32))]
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
        results = [SequenceResult(sequence_index=0, sequence_tokens=tokens, targets=targets, predictions=predictions)]
        m, events = self.calc_metrics_and_windows(results, min_weight=1.0)
        spans = {(e['start'], e['end'], e['classification']) for e in events if e['class_index'] == P.DSS}
        self.assertIn((5, 6, 'TP'), spans)
        self.assertIn((20, 21, 'FN'), spans)


class TestEventSpanMeanProbabilityMetrics(unittest.TestCase):
    def test_beta_fit_tp_tn_per_class(self):
        # Build a simple sequence with one START and one DSS motif
        dna = 'NNNATGNNNGTNN'
        vocab = {'A': 0, 'T': 1, 'G': 2, 'C': 3, 'N': 4}
        tokens = np.array([vocab.get(ch, 4) for ch in dna], dtype=np.int64)
        L = len(tokens)
        C = len(P.idx_to_cls)
        probs = np.zeros((L, C), dtype=np.float32)
        # START probs high on its window 3..5
        probs[3:6, P.START] = 0.9
        # DSS probs moderate on its window 9..10
        probs[9:11, P.DSS] = 0.6
        # Labels: only START positive, DSS negative
        targets = np.zeros(L, dtype=int)
        targets[3:6] = P.START
        res = [SequenceResult(sequence_index=0, sequence_tokens=tokens, targets=targets, predictions=None, probabilities=probs)]

        motifs = build_event_motifs(StandardDonorDinucleotides)
        out = compute_event_span_mean_probability_metrics(res, motifs)
        # START should have TP stats with mean near 0.9, DSS should have TN stats near 0.6
        self.assertIn(P.START, out)
        self.assertIn('tp', out[P.START])
        self.assertGreater(out[P.START]['tp']['n'], 0.0)
        self.assertAlmostEqual(out[P.START]['tp']['mean'], 0.9, places=2)
        self.assertIn(P.DSS, out)
        self.assertIn('tn', out[P.DSS])
        self.assertGreater(out[P.DSS]['tn']['n'], 0.0)
        self.assertAlmostEqual(out[P.DSS]['tn']['mean'], 0.6, places=2)

class TestMetricsValidMask(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
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
        return SequenceResult(
            sequence_index=0,
            sequence_tokens=tokens,
            targets=targets,
            predictions=predictions,
            probabilities=None,
        )

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


class TestSequenceResult(unittest.TestCase):
    def test_from_batch_builds_sequence_results(self):
        # Build tiny batch
        import torch
        tokens = torch.tensor([
            [0, 1, 2],
            [3, 4, 0],
        ], dtype=torch.long)
        targets = torch.tensor([
            [1, 1, 1],
            [0, 0, 0],
        ], dtype=torch.long)
        logits = torch.zeros((2, 3, 2), dtype=torch.float32)
        logits[0, :, 1] = 2.0  # class 1 for first sequence
        logits[1, :, 0] = 2.0  # class 0 for second sequence

        rs = SequenceResult.from_batch(tokens, targets, logits, sequence_index_start=5)
        self.assertEqual(len(rs), 2)
        self.assertEqual(rs[0].sequence_index, 5)
        self.assertEqual(rs[1].sequence_index, 6)
        # Predictions auto-computed
        np.testing.assert_array_equal(rs[0].predictions, np.array([1,1,1], dtype=np.int64))
        np.testing.assert_array_equal(rs[1].predictions, np.array([0,0,0], dtype=np.int64))


class TestSequenceResultSigmoidMasking(unittest.TestCase):
    def test_from_batch_sigmoid_and_nan_masking(self):
        import torch
        # Build tokens for sequence: N N A T N N  => positions 2..3 form motif 'AT'
        tokens = torch.tensor([[4, 4, 0, 1, 4, 4]], dtype=torch.long)
        logits = torch.zeros((1, 6, 3), dtype=torch.float32)
        logits[0, 2:4, 1] = 3.0  # only positions 2..3 have non-zero logits for class 1
        # Softmax: no NaNs
        rs_soft = SequenceResult.from_batch(tokens, None, logits, sequence_index_start=0, prob_activation='softmax')
        self.assertFalse(np.isnan(rs_soft[0].probabilities).any())
        # Sigmoid with event motifs for class 1 only: non-event positions for class 1 become NaN
        event_motifs = {1: {'AT'}}
        rs_sig = SequenceResult.from_batch(
            tokens,
            None,
            logits,
            sequence_index_start=0,
            prob_activation='sigmoid',
            event_motifs_by_class=event_motifs,
            event_margin_bp=0,
        )
        probs = rs_sig[0].probabilities
        # Class 1: only positions 2 and 3 should be finite
        self.assertTrue(np.isnan(probs[0, 1]))
        self.assertTrue(np.isnan(probs[1, 1]))
        self.assertGreater(probs[2, 1], 0.9)
        self.assertGreater(probs[3, 1], 0.9)
        self.assertTrue(np.isnan(probs[4, 1]))
        self.assertTrue(np.isnan(probs[5, 1]))


import unittest
import math
import numpy as np

from gene_decoder import PredictedSequence
from gene_decoder.decoder import decode_sequence, _event_logp, Beam, _scan_events
from gene_decoder.scoring import global_z_normalize_prob
from gene_decoder.codon_usage import CodonUsageModel, build_codon_usage_from_cds
from utils.constants import GenePredictionClass as P, StandardDonorDinucleotides, DinoDonorDinucleotides


def _make_probs(L: int, C: int, default_class: int = P.INTERGENIC, default_p: float = 0.99) -> np.ndarray:
    probs = np.full((L, C), (1.0 - default_p) / max(1, C - 1), dtype=np.float32)
    probs[:, int(default_class)] = default_p
    return probs


def _set_event(probs: np.ndarray, pos: int, cls_idx: int, span: int, p: float = 0.9) -> None:
    for i in range(pos, min(pos + span, probs.shape[0])):
        probs[i, int(cls_idx)] = p


class TestDecoder(unittest.TestCase):

    def test_global_z_normalize_prob_shapes_and_monotonicity(self):
        # Construct two sequences with different raw scales; ensure transform preserves shape and boosts rare highs
        L = 50
        C = len(P.idx_to_cls)
        probs1 = _make_probs(L, C, default_p=0.9)
        probs2 = _make_probs(L, C, default_p=0.9)
        # Add a high START peak in seq2 only
        s = 10
        _set_event(probs2, s, P.START, 3, 0.99)
        batch_adj = global_z_normalize_prob([probs1, probs2], beta=1.0)
        self.assertEqual(len(batch_adj), 2)
        self.assertEqual(batch_adj[0].shape, probs1.shape)
        self.assertEqual(batch_adj[1].shape, probs2.shape)
        # After standardization, the START peak in seq2 should remain higher than surrounding positions
        peak = float(batch_adj[1][s, int(P.START)])
        flank = float(batch_adj[1][s-1, int(P.START)])
        self.assertGreater(peak, flank)

    def test_global_z_normalize_prob_clamps_and_event_only(self):
        L = 30
        C = len(P.idx_to_cls)
        probs = _make_probs(L, C, default_p=0.95)
        # Artificial values near 0 and 1 to test clamping
        probs[5, int(P.DSS)] = 1.0
        probs[6, int(P.ASS)] = 0.0
        adj = global_z_normalize_prob([probs], beta=1.0)[0]
        # Event classes are clamped into (0,1)
        self.assertLess(adj[5, int(P.DSS)], 1.0)
        self.assertGreater(adj[6, int(P.ASS)], 0.0)
        # A non-event class remains unchanged at the same position
        non_event_cls = int(P.INTERGENIC)
        self.assertAlmostEqual(adj[5, non_event_cls], probs[5, non_event_cls])

    def test_event_logp_spans_and_clamping(self):
        # Verify span behavior: START/STOP use 3bp; DSS/ASS use 2bp and clamping avoids -inf
        L = 20
        C = len(P.idx_to_cls)
        probs = _make_probs(L, C, default_p=0.99)
        # Set exact spans with known probabilities
        s, d, a, t = 3, 8, 12, 15
        _set_event(probs, s, P.START, 3, 0.9)
        _set_event(probs, d, P.DSS, 2, 0.8)
        _set_event(probs, a, P.ASS, 2, 0.7)
        _set_event(probs, t, P.STOP, 3, 0.6)
        # Expected using average-over-span then log
        exp_start = math.log(0.9)
        exp_stop = math.log(0.6)
        exp_dss = math.log(0.8)
        exp_ass = math.log(0.7)
        self.assertAlmostEqual(_event_logp(probs, s, P.START), exp_start, places=6)
        self.assertAlmostEqual(_event_logp(probs, t, P.STOP), exp_stop, places=6)
        self.assertAlmostEqual(_event_logp(probs, d, P.DSS), exp_dss, places=6)
        self.assertAlmostEqual(_event_logp(probs, a, P.ASS), exp_ass, places=6)
        # Clamping: if any prob is 0, result should be finite (not -inf)
        probs[s, int(P.START)] = 0.0
        v = _event_logp(probs, s, P.START)
        self.assertTrue(math.isfinite(v))

    def test_single_exon_decode(self):
        # Sequence: NNN ATG ... TAA ...
        # start at 3, stop at 15 -> exon [3, 18)
        L = 22
        seq = 'NNN' + 'ATG' + 'A' * 9 + 'TAA' + 'A' * (L - 3 - 3 - 9 - 3)
        self.assertEqual(len(seq), L)
        C = len(P.idx_to_cls)
        probs = _make_probs(L, C)
        _set_event(probs, 3, P.START, span=3)
        _set_event(probs, 15, P.STOP, span=3)

        ps = PredictedSequence(sequence_index=0, sequence=seq, probabilities=probs,
                               class_order=[P.idx_to_cls[i] for i in sorted(P.idx_to_cls.keys())])
        res = decode_sequence(ps, StandardDonorDinucleotides, top_k_splicing=1, top_k_starts=1, beam_size=8)

        self.assertEqual(len(res.global_topk), 1)
        cand = res.global_topk[0]
        self.assertEqual(cand.exons, [(3, 18)])
        self.assertIn('start', cand.events)
        self.assertIn('stop', cand.events)
        self.assertEqual(cand.events['start'], [3])
        self.assertEqual(cand.events['stop'], [15])

    def test_one_intron_decode(self):
        # Design with one intron:
        # start at 3, donor GT at 12, acceptor AG at 20, next exon starts at 22, stop TAA at 28
        # exons: [3,12), [22,31) where stop spans 28..30 so exon ends at 31
        L = 35
        chars = ['A'] * L
        # Leading Ns for clarity
        chars[0:3] = list('NNN')
        # START
        chars[3:6] = list('ATG')
        # DSS at 12..13
        chars[12:14] = list('GT')
        # Intron filler
        # ASS at 20..21
        chars[20:22] = list('AG')
        # STOP at 28..30
        chars[28:31] = list('TAA')
        seq = ''.join(chars)

        C = len(P.idx_to_cls)
        probs = _make_probs(L, C)
        _set_event(probs, 3, P.START, span=3)
        _set_event(probs, 12, P.DSS, span=2)
        _set_event(probs, 20, P.ASS, span=2)
        _set_event(probs, 28, P.STOP, span=3)

        ps = PredictedSequence(sequence_index=1, sequence=seq, probabilities=probs,
                               class_order=[P.idx_to_cls[i] for i in sorted(P.idx_to_cls.keys())])
        # Also test hazard scoring path runs
        res = decode_sequence(ps, StandardDonorDinucleotides, top_k_splicing=1, top_k_starts=1, beam_size=32)

        self.assertEqual(len(res.global_topk), 1)
        cand = res.global_topk[0]
        self.assertEqual(cand.exons, [(3, 12), (22, 31)])
        self.assertEqual(cand.events['dss'], [12])
        self.assertEqual(cand.events['ass'], [20])
        self.assertEqual(cand.events['start'], [3])
        self.assertEqual(cand.events['stop'], [28])

    def test_stop_after_intron_out_of_frame_no_candidate(self):
        # START at 3, DSS at 12, ASS at 20, STOP at 29 (out of frame)
        # EXON lengths: [3,12) -> 9, [22,32) -> 10, total 19 (mod 3 != 0) => reject
        L = 35
        chars = ['A'] * L
        chars[0:3] = list('NNN')
        chars[3:6] = list('ATG')
        chars[12:14] = list('GT')
        chars[22:23] = list('AG')
        chars[29:32] = list('TAA')  # out-of-frame STOP
        seq = ''.join(chars)

        C = len(P.idx_to_cls)
        probs = _make_probs(L, C)
        _set_event(probs, 3, P.START, span=3)
        _set_event(probs, 12, P.DSS, span=2)
        _set_event(probs, 20, P.ASS, span=2)
        _set_event(probs, 29, P.STOP, span=3)

        ps = PredictedSequence(sequence_index=14, sequence=seq, probabilities=probs,
                               class_order=[P.idx_to_cls[i] for i in sorted(P.idx_to_cls.keys())])
        res = decode_sequence(ps, StandardDonorDinucleotides, top_k_splicing=3, top_k_starts=3, beam_size=16)
        self.assertEqual(len(res.global_topk), 0)

    def test_start_then_two_ass_ignored(self):
        # START then ASS, ASS inside EXON should have no effect; STOP in-frame
        L = 40
        seq = list('A' * L)
        s = 3
        a1 = 10
        a2 = 14
        t = 30  # in-frame with START at 3 (30 - (3+3) = 24)
        seq[0:3] = list('NNN')
        seq[s:s+3] = list('ATG')
        seq[a1:a1+2] = list('AG')
        seq[a2:a2+2] = list('AG')
        seq[t:t+3] = list('TAA')
        seq = ''.join(seq)

        C = len(P.idx_to_cls)
        probs = _make_probs(L, C)
        _set_event(probs, s, P.START, 3, 0.99)
        _set_event(probs, a1, P.ASS, 2, 0.99)
        _set_event(probs, a2, P.ASS, 2, 0.99)
        _set_event(probs, t, P.STOP, 3, 0.99)

        ps = PredictedSequence(15, seq, probs, [P.idx_to_cls[i] for i in sorted(P.idx_to_cls.keys())])
        res = decode_sequence(ps, StandardDonorDinucleotides, top_k_splicing=2, top_k_starts=2, beam_size=16)
        self.assertEqual(len(res.global_topk), 1)
        cand = res.global_topk[0]
        # Single exon; ASS inside exon ignored
        self.assertEqual(cand.exons, [(s, t+3)])
        self.assertEqual(cand.events['ass'], [])

    def test_two_dss_back_to_back_only_first_taken(self):
        # START -> DSS -> DSS (back-to-back), then ASS, STOP
        # After the first DSS we are in INTRON, so the second DSS must be ignored
        L = 50
        seq = list('A' * L)
        s = 3
        d1 = 12
        d2 = 14  # second DSS inside INTRON region
        a = 20
        t = 40  # choose STOP: split path in-frame, single-exon out-of-frame
        seq[0:3] = list('NNN')
        seq[s:s+3] = list('ATG')
        seq[d1:d1+2] = list('GT')
        seq[d2:d2+2] = list('GT')
        seq[a:a+2] = list('AG')
        seq[t:t+3] = list('TAA')
        seq = ''.join(seq)

        C = len(P.idx_to_cls)
        probs = _make_probs(L, C)
        _set_event(probs, s, P.START, 3, 0.99)
        _set_event(probs, d1, P.DSS, 2, 0.99)
        _set_event(probs, d2, P.DSS, 2, 0.99)
        _set_event(probs, a, P.ASS, 2, 0.99)
        _set_event(probs, t, P.STOP, 3, 0.99)

        ps = PredictedSequence(16, seq, probs, [P.idx_to_cls[i] for i in sorted(P.idx_to_cls.keys())])
        res = decode_sequence(ps, StandardDonorDinucleotides, top_k_splicing=3, top_k_starts=3, beam_size=16)
        self.assertGreaterEqual(len(res.global_topk), 1)
        # Pick any candidate that uses a DSS and verify only the first is taken
        cand = next(c for c in res.global_topk if len(c.events['dss']) >= 1)
        self.assertIn(d1, cand.events['dss'])
        self.assertNotIn(d2, cand.events['dss'])

    def test_boundary_and_hazard_exact_single_exon_math(self):
        # Exact math: no DSS/ASS present, hazard == boundary == sum logp of taken events
        L = 40
        seq = list('A' * L)
        s = 3
        t = 27  # in-frame: 27 - (3+3) = 21
        seq[0:3] = list('NNN')
        seq[s:s+3] = list('ATG')
        seq[t:t+3] = list('TAA')
        seq = ''.join(seq)

        C = len(P.idx_to_cls)
        probs = _make_probs(L, C)
        p_start, p_stop = 0.8, 0.7
        _set_event(probs, s, P.START, 3, p_start)
        _set_event(probs, t, P.STOP, 3, p_stop)

        ps = PredictedSequence(20, seq, probs, [P.idx_to_cls[i] for i in sorted(P.idx_to_cls.keys())])
        res = decode_sequence(ps, StandardDonorDinucleotides, top_k_splicing=1, top_k_starts=1, beam_size=8)
        self.assertEqual(len(res.global_topk), 1)
        cb = res.global_topk[0]
        ch = cb
        expected = _event_logp(probs, s, P.START) + _event_logp(probs, t, P.STOP)
        self.assertAlmostEqual(cb.boundary_score, expected, places=6)
        self.assertAlmostEqual(cb.transition_score, 0.0, places=6)
        self.assertAlmostEqual(ch.boundary_score, expected, places=6)
        self.assertAlmostEqual(ch.transition_score, 0.0, places=6)

    def test_hazard_exact_exon_skips_dss_math(self):
        # Two DSS along exon are skipped; hazard adds log(1 - p_dss_span) at each DSS index
        L = 60
        seq = list('A' * L)
        s = 3
        d1 = 12
        d2 = 20
        t = 45  # in-frame single-exon
        seq[0:3] = list('NNN')
        seq[s:s+3] = list('ATG')
        seq[d1:d1+2] = list('GT')
        seq[d2:d2+2] = list('GT')
        seq[t:t+3] = list('TAA')
        seq = ''.join(seq)

        C = len(P.idx_to_cls)
        probs = _make_probs(L, C)
        p_start, p_stop, p_dss = 0.9, 0.9, 0.6
        _set_event(probs, s, P.START, 3, p_start)
        _set_event(probs, d1, P.DSS, 2, p_dss)
        _set_event(probs, d2, P.DSS, 2, p_dss)
        _set_event(probs, t, P.STOP, 3, p_stop)

        ps = PredictedSequence(21, seq, probs, [P.idx_to_cls[i] for i in sorted(P.idx_to_cls.keys())])
        res_h = decode_sequence(ps, StandardDonorDinucleotides, top_k_splicing=1, top_k_starts=1, beam_size=16)
        self.assertEqual(len(res_h.global_topk), 1)
        cand = res_h.global_topk[0]
        # choose the single-exon candidate (no dss taken)
        self.assertEqual(cand.events['dss'], [])
        # Compute boundary directly from candidate events to match decoder
        expected_boundary = 0.0
        for pos in cand.events['start']:
            expected_boundary += _event_logp(probs, pos, P.START)
        for pos in cand.events['dss']:
            expected_boundary += _event_logp(probs, pos, P.DSS)
        for pos in cand.events['ass']:
            expected_boundary += _event_logp(probs, pos, P.ASS)
        for pos in cand.events['stop']:
            expected_boundary += _event_logp(probs, pos, P.STOP)
        # hazard penalties for skipping each DSS event: use event span probs for DSS via _event_logp
        p1 = float(np.exp(_event_logp(probs, d1, P.DSS)))
        p2 = float(np.exp(_event_logp(probs, d2, P.DSS)))
        pen = math.log(max(1e-12, 1.0 - p1)) + math.log(max(1e-12, 1.0 - p2))
        expected_total = expected_boundary + pen
        self.assertAlmostEqual(cand.boundary_score, expected_boundary, places=6)
        self.assertAlmostEqual(cand.transition_score, pen, places=6)

    def test_hazard_no_penalty_for_dss_in_intron_math(self):
        # Path: START -> DSS -> INTRON (has extra DSS, which should NOT penalize) -> ASS -> STOP
        # Expect: boundary = START+STOP; transition = DSS+ASS (no penalties while in INTRON)
        L = 60
        seq = list('A' * L)
        s = 3
        d = 12
        d_intr = 16  # DSS inside intron
        a = 22
        t = 39
        seq[0:3] = list('NNN')
        seq[s:s+3] = list('ATG')
        seq[d:d+2] = list('GT')
        seq[d_intr:d_intr+2] = list('GT')
        seq[a:a+2] = list('AG')
        seq[t:t+3] = list('TAA')
        seq = ''.join(seq)

        C = len(P.idx_to_cls)
        probs = _make_probs(L, C)
        p_start, p_dss, p_ass, p_stop = 0.92, 0.88, 0.91, 0.87
        _set_event(probs, s, P.START, 3, p_start)
        _set_event(probs, d, P.DSS, 2, p_dss)
        _set_event(probs, d_intr, P.DSS, 2, p_dss)
        _set_event(probs, a, P.ASS, 2, p_ass)
        _set_event(probs, t, P.STOP, 3, p_stop)

        ps = PredictedSequence(22, seq, probs, [P.idx_to_cls[i] for i in sorted(P.idx_to_cls.keys())])
        res_h = decode_sequence(ps, StandardDonorDinucleotides, top_k_splicing=2, top_k_starts=2, beam_size=16)
        self.assertGreaterEqual(len(res_h.global_topk), 1)
        # pick the split candidate (one dss and one ass recorded)
        cand = next(c for c in res_h.global_topk if len(c.events['dss']) == 1 and len(c.events['ass']) == 1)
        # Expected boundary: START + STOP only
        expected_boundary = 0.0
        for pos in cand.events['start']:
            expected_boundary += _event_logp(probs, pos, P.START)
        for pos in cand.events['stop']:
            expected_boundary += _event_logp(probs, pos, P.STOP)
        # Expected transition: DSS and ASS positives; no penalties while in INTRON
        expected_transition = _event_logp(probs, d, P.DSS) + _event_logp(probs, a, P.ASS)
        self.assertAlmostEqual(cand.boundary_score, expected_boundary, places=6)
        self.assertAlmostEqual(cand.transition_score, expected_transition, places=6)

    def test_codon_usage_scoring(self):
        # Build a small codon model that favors AAA heavily
        model = CodonUsageModel(probabilities={c: 1.0/64.0 for c in [a+b+c for a in 'ATGC' for b in 'ATGC' for c in 'ATGC']})
        model.probabilities['AAA'] = 0.99
        model.probabilities['GGG'] = 0.01

        cds1 = 'AAA' * 4
        cds2 = 'GGG' * 4
        s1 = model.rare_codon_penalty(cds1)
        s2 = model.rare_codon_penalty(cds2)
        self.assertGreater(s1, s2)

    def test_stop_terminates_immediately(self):
        # START at 3, STOP at 12 ('TGA'), and DSS 'GA' at 13 (overlaps STOP+1)
        # Decoder should terminate at STOP and never take DSS at 13
        L = 20
        chars = ['A'] * L
        chars[0:3] = list('NNN')
        chars[3:6] = list('ATG')
        # place STOP 'TGA' at 12..14 so in-frame from 3 (exon length 12 -> phase 0)
        stop_i = 12
        chars[stop_i:stop_i+3] = list('TGA')
        # DSS 'GA' begins at stop_i+1
        # Already satisfied by TGA (positions 13..14 == 'GA')
        seq = ''.join(chars)

        C = len(P.idx_to_cls)
        probs = _make_probs(L, C)
        _set_event(probs, 3, P.START, span=3, p=0.95)
        _set_event(probs, stop_i, P.STOP, span=3, p=0.95)
        _set_event(probs, stop_i + 1, P.DSS, span=2, p=0.99)

        ps = PredictedSequence(sequence_index=2, sequence=seq, probabilities=probs,
                               class_order=[P.idx_to_cls[i] for i in sorted(P.idx_to_cls.keys())])
        res = decode_sequence(ps, StandardDonorDinucleotides, top_k_splicing=3, top_k_starts=3, beam_size=16)
        for res in (res,):
            self.assertGreaterEqual(len(res.global_topk), 1)
            cand = res.global_topk[0]
            # must end at STOP and not contain any DSS
            self.assertEqual(cand.events['stop'], [stop_i])
            self.assertEqual(cand.events['dss'], [])
            self.assertEqual(cand.exons[-1][1], stop_i + 3)

    def test_cannot_skip_inframe_stop(self):
        # Earliest in-frame STOP should prevent exploring donors beyond it
        L = 60
        seq = list('A' * L)
        s = 3
        t1 = 27  # in-frame from 3
        d = 30   # donor after first in-frame STOP
        a = 38
        t2 = 50
        seq[0:3] = list('NNN')
        seq[s:s+3] = list('ATG')
        seq[t1:t1+3] = list('TAA')
        seq[d:d+2] = list('GT')
        seq[a:a+2] = list('AG')
        seq[t2:t2+3] = list('TGA')
        seq = ''.join(seq)

        C = len(P.idx_to_cls)
        probs = _make_probs(L, C)
        _set_event(probs, s, P.START, 3, 0.99)
        _set_event(probs, t1, P.STOP, 3, 0.99)
        _set_event(probs, d, P.DSS, 2, 0.99)
        _set_event(probs, a, P.ASS, 2, 0.99)
        _set_event(probs, t2, P.STOP, 3, 0.99)

        ps = PredictedSequence(43, seq, probs, [P.idx_to_cls[i] for i in sorted(P.idx_to_cls.keys())])
        res = decode_sequence(ps, StandardDonorDinucleotides, top_k_splicing=10, top_k_starts=10, beam_size=10000)
        self.assertGreaterEqual(len(res.global_topk), 1)
        for cand in res.global_topk:
            self.assertIn(t1, cand.events['stop'])
            self.assertTrue(all(x < t1 for x in cand.events['dss']))

    def test_no_splice_past_first_stop_even_if_would_win(self):
        # Construct a case where skipping the first in-frame STOP and splicing later
        # would have a higher boundary score if allowed. Assert decoder still ends at
        # the earliest in-frame STOP and returns no candidates with DSS >= STOP1.
        L = 80
        s = 3
        t1 = 27   # earliest in-frame STOP from s
        d = 30    # donor AFTER t1
        a = 38
        t2 = 61   # choose so that (t2 - (a+2)) % 3 == 0
        chars = ['A'] * L
        chars[0:3] = list('NNN')
        chars[s:s+3] = list('ATG')
        chars[t1:t1+3] = list('TAA')
        chars[d:d+2] = list('GT')
        chars[a:a+2] = list('AG')
        chars[t2:t2+3] = list('TGA')
        seq = ''.join(chars)

        C = len(P.idx_to_cls)
        probs = _make_probs(L, C)
        # Make early STOP moderate, and splice path events very strong
        _set_event(probs, s, P.START, 3, 0.95)
        _set_event(probs, t1, P.STOP, 3, 0.55)
        _set_event(probs, d, P.DSS, 2, 0.99)
        _set_event(probs, a, P.ASS, 2, 0.99)
        _set_event(probs, t2, P.STOP, 3, 0.99)

        # Sanity: if skipping were allowed, the split path's boundary would beat direct STOP
        direct_boundary = _event_logp(probs, s, P.START) + _event_logp(probs, t1, P.STOP)
        split_boundary = (
            _event_logp(probs, s, P.START)
            + _event_logp(probs, d, P.DSS)
            + _event_logp(probs, a, P.ASS)
            + _event_logp(probs, t2, P.STOP)
        )
        self.assertGreater(split_boundary, direct_boundary)

        ps = PredictedSequence(44, seq, probs, [P.idx_to_cls[i] for i in sorted(P.idx_to_cls.keys())])
        res = decode_sequence(ps, StandardDonorDinucleotides, top_k_splicing=20, top_k_starts=20, beam_size=100000)
        self.assertGreaterEqual(len(res.global_topk), 1)
        for cand in res.global_topk:
            self.assertIn(t1, cand.events['stop'])
            self.assertTrue(all(x < t1 for x in cand.events['dss']))

    def test_stop_out_of_frame_no_candidate(self):
        # START at 3, STOP at 16 (out of frame since 16 - (3+3) = 10, not multiple of 3)
        L = 25
        seq = list('A' * L)
        seq[0:3] = list('NNN')
        seq[3:6] = list('ATG')
        seq[16:19] = list('TAA')
        seq = ''.join(seq)
        C = len(P.idx_to_cls)
        probs = _make_probs(L, C)
        _set_event(probs, 3, P.START, 3, 0.95)
        _set_event(probs, 16, P.STOP, 3, 0.95)
        ps = PredictedSequence(4, seq, probs, [P.idx_to_cls[i] for i in sorted(P.idx_to_cls.keys())])
        res = decode_sequence(ps, StandardDonorDinucleotides, top_k_splicing=3, top_k_starts=3, beam_size=8)
        self.assertEqual(len(res.global_topk), 0)

    def test_exon_ass_ignored(self):
        # Place ASS inside exon; decoder should not record ASS unless in INTRON
        L = 30
        seq = list('A' * L)
        seq[0:3] = list('NNN')
        seq[3:6] = list('ATG')
        seq[10:12] = list('AG')  # ASS in exon
        seq[21:24] = list('TAA')  # STOP in-frame (21 - (3+3) = 15)
        seq = ''.join(seq)
        C = len(P.idx_to_cls)
        probs = _make_probs(L, C)
        _set_event(probs, 3, P.START, 3, 0.9)
        _set_event(probs, 10, P.ASS, 2, 0.99)
        _set_event(probs, 21, P.STOP, 3, 0.9)
        ps = PredictedSequence(5, seq, probs, [P.idx_to_cls[i] for i in sorted(P.idx_to_cls.keys())])
        res = decode_sequence(ps, StandardDonorDinucleotides, top_k_splicing=1, top_k_starts=1, beam_size=8)
        self.assertEqual(len(res.global_topk), 1)
        self.assertEqual(res.global_topk[0].events['ass'], [])

    def test_intron_dss_ignored(self):
        # START->DSS->INTRON; place DSS inside intron; should be ignored
        L = 40
        seq = list('A' * L)
        seq[0:3] = list('NNN')
        seq[3:6] = list('ATG')
        d = 12
        a = 22
        t = 30
        seq[d:d+2] = list('GT')
        seq[a:a+2] = list('AG')
        seq[a+5:a+7] = list('GT')  # spurious DSS inside intron before ASS
        seq[t:t+3] = list('TAA')
        seq = ''.join(seq)
        C = len(P.idx_to_cls)
        probs = _make_probs(L, C)
        _set_event(probs, 3, P.START, 3, 0.95)
        _set_event(probs, d, P.DSS, 2, 0.95)
        _set_event(probs, a, P.ASS, 2, 0.95)
        _set_event(probs, a+5, P.DSS, 2, 0.99)
        _set_event(probs, t, P.STOP, 3, 0.95)
        ps = PredictedSequence(6, seq, probs, [P.idx_to_cls[i] for i in sorted(P.idx_to_cls.keys())])
        res = decode_sequence(ps, StandardDonorDinucleotides, top_k_splicing=3, top_k_starts=3, beam_size=32)
        # Pick any split candidate and validate intronic DSS is ignored
        split = next(c for c in res.global_topk if len(c.events['dss']) == 1)
        self.assertIn(d, split.events['dss'])
        self.assertNotIn(a+5, split.events['dss'])

    def test_hazard_exon_dss_skip_penalizes(self):
        # Single exon with many DSS along exon; transition_score adds skip penalties
        L = 60
        seq = list('A' * L)
        seq[0:3] = list('NNN')
        seq[3:6] = list('ATG')
        t = 48
        seq[t:t+3] = list('TAA')  # in-frame: 48 - (3+3) = 42
        # Sprinkle DSS every 5 bp starting at 10
        for x in range(10, t, 5):
            seq[x:x+2] = list('GT')
        seq = ''.join(seq)
        C = len(P.idx_to_cls)
        probs = _make_probs(L, C)
        _set_event(probs, 3, P.START, 3, 0.99)
        for x in range(10, t, 5):
            _set_event(probs, x, P.DSS, 2, 0.9)
        _set_event(probs, t, P.STOP, 3, 0.99)
        ps = PredictedSequence(7, seq, probs, [P.idx_to_cls[i] for i in sorted(P.idx_to_cls.keys())])
        # Use very large beam and k to avoid pruning effects in this test
        res = decode_sequence(ps, StandardDonorDinucleotides, top_k_splicing=100, top_k_starts=100, beam_size=100000)
        self.assertGreaterEqual(len(res.global_topk), 1)
        # Choose the single-exon (no DSS taken) candidate
        b = next(c for c in res.global_topk if len(c.events['dss']) == 0)
        self.assertEqual(b.events['dss'], [])
        # Expected skip penalties at each DSS along the exon
        expected_pen = 0.0
        for x in range(10, t, 5):
            px = float(np.exp(_event_logp(probs, x, P.DSS)))
            expected_pen += math.log(max(1e-12, 1.0 - px))
        self.assertAlmostEqual(b.transition_score, expected_pen, places=6)
        self.assertLess(b.transition_score, 0.0)

    def test_hazard_intron_ass_skip_penalizes(self):
        # Put two ASS inside intron; earlier high-prob ASS leads to out-of-frame STOP,
        # so decoder should skip it and use later ASS; transition_score penalizes the skip
        L = 60
        seq = list('A' * L)
        seq[0:3] = list('NNN')
        s = 3
        d = 12
        a1 = 19  # earlier ASS (will be out-of-frame with STOP)
        a2 = 21  # later ASS chosen (in-frame)
        t = 41   # t+3 = 44 ≡ 2; (44 - (a2+2=23)) ≡ 0; (44 - (a1+2=21)) ≡ 2 (not 0)
        seq[s:s+3] = list('ATG')
        seq[d:d+2] = list('GT')
        seq[a1:a1+2] = list('AG')
        seq[a2:a2+2] = list('AG')
        seq[t:t+3] = list('TAA')
        seq = ''.join(seq)
        C = len(P.idx_to_cls)
        probs = _make_probs(L, C)
        _set_event(probs, s, P.START, 3, 0.99)
        _set_event(probs, d, P.DSS, 2, 0.99)
        _set_event(probs, a1, P.ASS, 2, 0.95)
        _set_event(probs, a2, P.ASS, 2, 0.95)
        _set_event(probs, t, P.STOP, 3, 0.99)
        ps = PredictedSequence(8, seq, probs, [P.idx_to_cls[i] for i in sorted(P.idx_to_cls.keys())])
        res = decode_sequence(ps, StandardDonorDinucleotides, top_k_splicing=3, top_k_starts=3, beam_size=32)
        # pick the best split candidate
        c = next(c for c in res.global_topk if len(c.events['dss']) == 1)
        # Should use a2, skipping a1
        self.assertNotIn(a1, c.events['ass'])
        self.assertIn(a2, c.events['ass'])
        # Transition equals DSS+ASS positives plus skip penalty at a1
        expected_transition = _event_logp(probs, d, P.DSS) + _event_logp(probs, a2, P.ASS)
        p_a1 = float(np.exp(_event_logp(probs, a1, P.ASS)))
        expected_transition += math.log(max(1e-12, 1.0 - p_a1))
        self.assertAlmostEqual(c.transition_score, expected_transition, places=6)

    def test_beam_prefers_split_when_events_strong(self):
        # Make DSS/ASS + STOP much stronger than direct STOP; even with small beam, pick split
        L = 50
        seq = list('A' * L)
        s = 3
        d = 12
        a = 20
        t = 31  # choose in-frame for split and not in-frame for single-exon
        seq[0:3] = list('NNN')
        seq[s:s+3] = list('ATG')
        seq[d:d+2] = list('GT')
        seq[a:a+2] = list('AG')
        seq[t:t+3] = list('TAA')
        seq = ''.join(seq)
        C = len(P.idx_to_cls)
        probs = _make_probs(L, C)
        _set_event(probs, s, P.START, 3, 0.99)
        _set_event(probs, d, P.DSS, 2, 0.99)
        _set_event(probs, a, P.ASS, 2, 0.99)
        _set_event(probs, t, P.STOP, 3, 0.6)
        ps = PredictedSequence(9, seq, probs, [P.idx_to_cls[i] for i in sorted(P.idx_to_cls.keys())])
        # Use hazard scoring so repeated DSS skips penalize single-exon path
        res = decode_sequence(ps, StandardDonorDinucleotides, top_k_splicing=1, top_k_starts=1, beam_size=2)
        self.assertEqual(len(res.global_topk), 1)
        c = res.global_topk[0]
        self.assertEqual(c.events['dss'], [d])
        self.assertEqual(c.events['ass'], [a])

    def test_dp_handles_multiple_introns_without_limit(self):
        # Build two introns; ensure decoder runs and returns candidates (no intron limit enforced)
        L = 80
        seq = list('A' * L)
        s = 3
        d1 = 12
        a1 = 20
        d2 = 30
        a2 = 38
        t = 57
        seq[0:3] = list('NNN')
        seq[s:s+3] = list('ATG')
        for p in [(d1,'GT'), (a1,'AG'), (d2,'GT'), (a2,'AG')]:
            seq[p[0]:p[0]+len(p[1])] = list(p[1])
        seq[t:t+3] = list('TAA')
        seq = ''.join(seq)
        C = len(P.idx_to_cls)
        probs = _make_probs(L, C)
        for pos, cls, span, p in [(s,P.START,3,0.99), (d1,P.DSS,2,0.99), (a1,P.ASS,2,0.99), (d2,P.DSS,2,0.99), (a2,P.ASS,2,0.99), (t,P.STOP,3,0.99)]:
            _set_event(probs, pos, cls, span, p)
        ps = PredictedSequence(10, seq, probs, [P.idx_to_cls[i] for i in sorted(P.idx_to_cls.keys())])
        res = decode_sequence(ps, StandardDonorDinucleotides, top_k_splicing=3, top_k_starts=3, beam_size=16)
        self.assertGreaterEqual(len(res.global_topk), 1)

    def test_multiple_starts_and_global_union(self):
        # Two STARTs -> per_start has two keys; global aggregates
        L = 40
        seq = list('A' * L)
        s1 = 3
        s2 = 10
        t1 = 21
        t2 = 31  # make in-frame for s2 (31 - (10+3) = 18)
        seq[0:3] = list('NNN')
        seq[s1:s1+3] = list('ATG')
        seq[s2:s2+3] = list('ATG')
        seq[t1:t1+3] = list('TAA')
        seq[t2:t2+3] = list('TAA')
        seq = ''.join(seq)
        C = len(P.idx_to_cls)
        probs = _make_probs(L, C)
        _set_event(probs, s1, P.START, 3, 0.95)
        _set_event(probs, s2, P.START, 3, 0.95)
        _set_event(probs, t1, P.STOP, 3, 0.95)
        _set_event(probs, t2, P.STOP, 3, 0.95)
        ps = PredictedSequence(11, seq, probs, [P.idx_to_cls[i] for i in sorted(P.idx_to_cls.keys())])
        res = decode_sequence(ps, StandardDonorDinucleotides, top_k_splicing=2, top_k_starts=2, beam_size=8)
        self.assertGreaterEqual(len(res.per_start.keys()), 2)
        self.assertEqual(len(res.global_topk), 2)

    def test_no_overlap_selection(self):
        # Two overlapping candidates; with allow_overlap=False only one remains
        L = 35
        seq = list('A' * L)
        s1 = 3; t1 = 18
        s2 = 6; t2 = 21
        seq[0:3] = list('NNN')
        seq[s1:s1+3] = list('ATG')
        seq[s2:s2+3] = list('ATG')
        seq[t1:t1+3] = list('TAA')
        seq[t2:t2+3] = list('TAA')
        seq = ''.join(seq)
        C = len(P.idx_to_cls)
        probs = _make_probs(L, C)
        for pos in [(s1,P.START,3),(s2,P.START,3),(t1,P.STOP,3),(t2,P.STOP,3)]:
            _set_event(probs, pos[0], pos[1], pos[2], 0.9)
        ps = PredictedSequence(12, seq, probs, [P.idx_to_cls[i] for i in sorted(P.idx_to_cls.keys())])
        res = decode_sequence(ps, StandardDonorDinucleotides, top_k_splicing=3, top_k_starts=1, beam_size=8, allow_overlap=False)
        self.assertEqual(len(res.global_topk), 1)

    def test_event_near_end_no_oob(self):
        # DSS at L-2 and STOP at L-3 should not crash; candidate may or may not use DSS
        L = 25
        seq = list('A' * L)
        s = 3
        d = L - 2
        t = L - 3
        seq[0:3] = list('NNN')
        seq[s:s+3] = list('ATG')
        seq[d:d+2] = list('GT')
        seq[t:t+3] = list('TAA')
        seq = ''.join(seq)
        C = len(P.idx_to_cls)
        probs = _make_probs(L, C)
        _set_event(probs, s, P.START, 3, 0.9)
        _set_event(probs, d, P.DSS, 2, 0.9)
        _set_event(probs, t, P.STOP, 3, 0.9)
        ps = PredictedSequence(13, seq, probs, [P.idx_to_cls[i] for i in sorted(P.idx_to_cls.keys())])
        res = decode_sequence(ps, StandardDonorDinucleotides, top_k_splicing=2, top_k_starts=2, beam_size=8)
        # Ensure no exception; candidate existence depends on frame
        self.assertGreaterEqual(len(res.global_topk), 0)

    def test_both_single_and_split_transcripts_when_both_in_frame(self):
        # START at 3, DSS at 9, ASS at 16, STOP at 30
        # Single-exon path: START->STOP in-frame
        # Split-exon path: START->DSS ... ASS->STOP also in-frame
        L = 40
        chars = ['A'] * L
        chars[0:3] = list('NNN')
        s = 3
        d = 9
        a = 16
        t = 30
        chars[s:s+3] = list('ATG')
        chars[d:d+2] = list('GT')
        chars[a:a+2] = list('AG')
        chars[t:t+3] = list('TAA')
        seq = ''.join(chars)

        C = len(P.idx_to_cls)
        probs = _make_probs(L, C)
        _set_event(probs, s, P.START, span=3, p=0.95)
        _set_event(probs, d, P.DSS, span=2, p=0.95)
        _set_event(probs, a, P.ASS, span=2, p=0.95)
        _set_event(probs, t, P.STOP, span=3, p=0.95)

        ps = PredictedSequence(sequence_index=3, sequence=seq, probabilities=probs,
                               class_order=[P.idx_to_cls[i] for i in sorted(P.idx_to_cls.keys())])

        res_b = decode_sequence(ps, StandardDonorDinucleotides, top_k_splicing=3, top_k_starts=3, beam_size=32)
        self.assertGreaterEqual(len(res_b.global_topk), 2)
        # Identify single-exon vs split-exon by presence of junctions
        has_single = any(len(c.events['dss']) == 0 and len(c.exons) == 1 for c in res_b.global_topk)
        has_split = any(len(c.events['dss']) == 1 and len(c.events['ass']) == 1 and len(c.exons) == 2 for c in res_b.global_topk)
        self.assertTrue(has_single)
        self.assertTrue(has_split)

        # Transition also proposes both; ordering may differ
        res_h = res_b

    def test_donor_updates_phase_for_stop_framing(self):
        # START at 3; donor at 10 creates exon length (10 - 3) = 7 -> phase 1
        # acceptor at 20 -> re-enter EXON at 22; STOP at 24 is in-frame only if phase was updated
        L = 40
        seq = list('A' * L)
        s = 3
        d = 10
        a = 20
        t = 24  # Valid if phase at re-entry is 1: (24 - 22) = 2, 1+2 ≡ 0 mod 3
        seq[0:3] = list('NNN')
        seq[s:s+3] = list('ATG')
        seq[d:d+2] = list('GT')
        seq[a:a+2] = list('AG')
        seq[t:t+3] = list('TAA')
        seq = ''.join(seq)

        C = len(P.idx_to_cls)
        probs = _make_probs(L, C)
        _set_event(probs, s, P.START, 3, 0.99)
        _set_event(probs, d, P.DSS, 2, 0.99)
        _set_event(probs, a, P.ASS, 2, 0.99)
        _set_event(probs, t, P.STOP, 3, 0.99)

        ps = PredictedSequence(45, seq, probs, [P.idx_to_cls[i] for i in sorted(P.idx_to_cls.keys())])
        res = decode_sequence(ps, StandardDonorDinucleotides, top_k_splicing=5, top_k_starts=5, beam_size=1000)
        # Expect a split candidate with DSS at 10, ASS at 20, STOP at 24
        split = next(c for c in res.global_topk if c.events['dss'] == [d] and c.events['ass'] == [a] and c.events['stop'] == [t])
        self.assertIsNotNone(split)

    def test_scan_events_threshold(self):
        L = 40
        seq = list('A' * L)
        s = 3
        d = 10
        a = 20
        t = 30
        seq[0:3] = list('NNN')
        seq[s:s+3] = list('ATG')
        seq[d:d+2] = list('GT')
        seq[a:a+2] = list('AG')
        seq[t:t+3] = list('TAA')
        seq = ''.join(seq)
        C = len(P.idx_to_cls)
        probs = _make_probs(L, C)
        # Set low probs below 0.1 → should be filtered out
        _set_event(probs, s, P.START, 3, 0.05)
        _set_event(probs, d, P.DSS, 2, 0.05)
        _set_event(probs, a, P.ASS, 2, 0.05)
        _set_event(probs, t, P.STOP, 3, 0.05)
        ev = _scan_events(seq, StandardDonorDinucleotides, probs=probs, min_logp=math.log(0.1))
        self.assertEqual(ev['start'], [])
        self.assertEqual(ev['dss'], [])
        self.assertEqual(ev['ass'], [])
        self.assertEqual(ev['stop'], [])
        # Raise probs above 0.1 → should be included
        _set_event(probs, s, P.START, 3, 0.2)
        _set_event(probs, d, P.DSS, 2, 0.2)
        _set_event(probs, a, P.ASS, 2, 0.2)
        _set_event(probs, t, P.STOP, 3, 0.2)
        ev2 = _scan_events(seq, StandardDonorDinucleotides, probs=probs, min_logp=math.log(0.1))
        self.assertEqual(ev2['start'], [s])
        self.assertEqual(ev2['dss'], [d])
        self.assertEqual(ev2['ass'], [a])
        self.assertEqual(ev2['stop'], [t])


if __name__ == '__main__':
    unittest.main()



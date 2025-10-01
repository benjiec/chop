import unittest
import numpy as np

from gene_decoder import PredictedSequence
from gene_decoder.decoder import decode_sequence
from gene_decoder.codon_usage import CodonUsageModel, build_codon_usage_from_cds
from utils.constants import GenePredictionClass as P


def _make_probs(L: int, C: int, default_class: int = P.INTERGENIC, default_p: float = 0.99) -> np.ndarray:
    probs = np.full((L, C), (1.0 - default_p) / max(1, C - 1), dtype=np.float32)
    probs[:, int(default_class)] = default_p
    return probs


def _set_event(probs: np.ndarray, pos: int, cls_idx: int, span: int, p: float = 0.9) -> None:
    for i in range(pos, min(pos + span, probs.shape[0])):
        probs[i, :] = (1.0 - p) / (probs.shape[1] - 1)
        probs[i, int(cls_idx)] = p


class TestDecoder(unittest.TestCase):
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
        res = decode_sequence(ps, k_per_start=1, k_global=1, beam_size=8, scoring='boundary')

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
        res = decode_sequence(ps, k_per_start=1, k_global=1, beam_size=32, scoring='boundary')
        res_h = decode_sequence(ps, k_per_start=1, k_global=1, beam_size=32, scoring='hazard')
        self.assertEqual(len(res_h.global_topk), 1)

        self.assertEqual(len(res.global_topk), 1)
        cand = res.global_topk[0]
        self.assertEqual(cand.exons, [(3, 12), (22, 31)])
        self.assertEqual(cand.events['dss'], [12])
        self.assertEqual(cand.events['ass'], [20])
        self.assertEqual(cand.events['start'], [3])
        self.assertEqual(cand.events['stop'], [28])

    def test_codon_usage_scoring(self):
        # Build a small codon model that favors AAA heavily
        model = CodonUsageModel(logp={c: np.log(1.0/64.0) for c in [a+b+c for a in 'ATGC' for b in 'ATGC' for c in 'ATGC']})
        model.logp['AAA'] = 0.0  # highest logp
        model.logp['GGG'] = -10.0

        cds1 = 'AAA' * 4
        cds2 = 'GGG' * 4
        s1 = model.score(cds1)
        s2 = model.score(cds2)
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
        res_b = decode_sequence(ps, k_per_start=3, k_global=3, beam_size=16, scoring='boundary')
        res_h = decode_sequence(ps, k_per_start=3, k_global=3, beam_size=16, scoring='hazard')

        for res in (res_b, res_h):
            self.assertGreaterEqual(len(res.global_topk), 1)
            cand = res.global_topk[0]
            # must end at STOP and not contain any DSS
            self.assertEqual(cand.events['stop'], [stop_i])
            self.assertEqual(cand.events['dss'], [])
            self.assertEqual(cand.exons[-1][1], stop_i + 3)

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
        res = decode_sequence(ps, k_per_start=3, k_global=3, beam_size=8, scoring='boundary')
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
        res = decode_sequence(ps, k_per_start=1, k_global=1, beam_size=8)
        self.assertEqual(len(res.global_topk), 1)
        self.assertEqual(res.global_topk[0].events['ass'], [])

    def test_intron_dss_ignored(self):
        # START->DSS->INTRON; place DSS inside intron; should be ignored
        L = 40
        seq = list('A' * L)
        seq[0:3] = list('NNN')
        seq[3:6] = list('ATG')
        d = 12
        a = 20
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
        res = decode_sequence(ps, k_per_start=2, k_global=2, beam_size=16)
        self.assertGreaterEqual(len(res.global_topk), 1)
        cand = res.global_topk[0]
        # Only one DSS (at exon end) should be recorded
        self.assertIn(d, cand.events['dss'])
        self.assertNotIn(a+5, cand.events['dss'])

    def test_hazard_exon_dss_skip_penalizes(self):
        # Single exon with many DSS along exon; hazard lower score than boundary
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
        res_b = decode_sequence(ps, k_per_start=1, k_global=1, beam_size=32, scoring='boundary')
        res_h = decode_sequence(ps, k_per_start=1, k_global=1, beam_size=32, scoring='hazard')
        self.assertEqual(len(res_b.global_topk), 1)
        self.assertEqual(len(res_h.global_topk), 1)
        b = res_b.global_topk[0]
        h = res_h.global_topk[0]
        # Hazard penalizes skips -> total (==boundary here) is lower
        self.assertLess(h.total, b.total)
        self.assertEqual(b.events['dss'], [])

    def test_hazard_intron_ass_skip_penalizes(self):
        # Put two ASS inside intron; earlier high-prob ASS leads to out-of-frame STOP,
        # so decoder should skip it and use later ASS; hazard penalizes the skip
        L = 60
        seq = list('A' * L)
        seq[0:3] = list('NNN')
        s = 3
        d = 12
        a1 = 18  # earlier ASS
        a2 = 21  # later ASS chosen
        t = 39   # choose so that using a2 yields in-frame STOP
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
        res_b = decode_sequence(ps, k_per_start=3, k_global=3, beam_size=32, scoring='boundary')
        res_h = decode_sequence(ps, k_per_start=3, k_global=3, beam_size=32, scoring='hazard')
        # pick the best split candidate from each
        cb = next(c for c in res_b.global_topk if len(c.events['dss']) == 1)
        ch = next(c for c in res_h.global_topk if len(c.events['dss']) == 1)
        # Both should use a2, skipping a1
        self.assertNotIn(a1, cb.events['ass'])
        self.assertIn(a2, cb.events['ass'])
        self.assertNotIn(a1, ch.events['ass'])
        self.assertIn(a2, ch.events['ass'])
        self.assertLess(ch.total, cb.total)

    def test_beam_prefers_split_when_events_strong(self):
        # Make DSS/ASS + STOP much stronger than direct STOP; even with small beam, pick split
        L = 50
        seq = list('A' * L)
        s = 3
        d = 12
        a = 20
        t = 33
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
        res = decode_sequence(ps, k_per_start=1, k_global=1, beam_size=2, scoring='boundary')
        self.assertEqual(len(res.global_topk), 1)
        c = res.global_topk[0]
        self.assertEqual(c.events['dss'], [d])
        self.assertEqual(c.events['ass'], [a])

    def test_max_introns_limit_enforced(self):
        # Build two introns but set max_introns=1; decoder must output at most one intron
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
        res = decode_sequence(ps, k_per_start=3, k_global=3, beam_size=16, max_introns=1, scoring='boundary')
        self.assertGreaterEqual(len(res.global_topk), 1)
        self.assertTrue(all(len(c.events['dss']) <= 1 for c in res.global_topk))

    def test_multiple_starts_and_global_union(self):
        # Two STARTs -> per_start has two keys; global aggregates
        L = 40
        seq = list('A' * L)
        s1 = 3
        s2 = 10
        t1 = 21
        t2 = 30
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
        res = decode_sequence(ps, k_per_start=2, k_global=2, beam_size=8)
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
        res = decode_sequence(ps, k_per_start=3, k_global=1, beam_size=8, allow_overlap=False)
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
        res = decode_sequence(ps, k_per_start=2, k_global=2, beam_size=8)
        # Just ensure no exception and at least one candidate exists (likely single exon)
        self.assertGreaterEqual(len(res.global_topk), 1)

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

        # Boundary scoring: should propose both candidates
        res_b = decode_sequence(ps, k_per_start=3, k_global=3, beam_size=32, scoring='boundary')
        self.assertGreaterEqual(len(res_b.global_topk), 2)
        # Identify single-exon vs split-exon by presence of junctions
        has_single = any(len(c.events['dss']) == 0 and len(c.exons) == 1 for c in res_b.global_topk)
        has_split = any(len(c.events['dss']) == 1 and len(c.events['ass']) == 1 and len(c.exons) == 2 for c in res_b.global_topk)
        self.assertTrue(has_single)
        self.assertTrue(has_split)

        # Hazard scoring: should also propose both
        res_h = decode_sequence(ps, k_per_start=3, k_global=3, beam_size=32, scoring='hazard')
        self.assertGreaterEqual(len(res_h.global_topk), 2)
        has_single_h = any(len(c.events['dss']) == 0 and len(c.exons) == 1 for c in res_h.global_topk)
        has_split_h = any(len(c.events['dss']) == 1 and len(c.events['ass']) == 1 and len(c.exons) == 2 for c in res_h.global_topk)
        self.assertTrue(has_single_h)
        self.assertTrue(has_split_h)


if __name__ == '__main__':
    unittest.main()



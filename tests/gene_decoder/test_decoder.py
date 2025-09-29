import unittest
import numpy as np

from gene_decoder.types import PredictedSequence
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
        res = decode_sequence(ps, k_per_start=1, k_global=1, beam_size=8)

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
        res = decode_sequence(ps, k_per_start=1, k_global=1, beam_size=32)

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


if __name__ == '__main__':
    unittest.main()



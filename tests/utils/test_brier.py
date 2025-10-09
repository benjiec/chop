import unittest
import numpy as np

from utils.metrics import event_based_brier_factory
from utils.events import build_event_motifs
from utils.constants import GenePredictionClass as P
from utils.constants import DNAEmbed, StandardDonorDinucleotides


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
        rd = [{'sequence_tokens': tokens, 'targets': targets, 'probabilities': probs}]
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
        rd = [{'sequence_tokens': tokens, 'targets': targets, 'probabilities': probs}]
        out = self.compute_brier(rd)
        # START per-class: mean((0.8-1)^2) = 0.04
        # STOP per-class: mean((0.7-1)^2) = 0.09
        self.assertAlmostEqual(out['brier_by_class'][P.START], 0.04, places=6)
        self.assertAlmostEqual(out['brier_by_class'][P.STOP], 0.09, places=6)
        self.assertAlmostEqual(out['brier'], (0.04 + 0.09) / 2.0, places=6)

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
        rd = [{'sequence_tokens': tokens, 'targets': targets, 'probabilities': probs}]
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
        rd = [{'sequence_tokens': tokens, 'targets': targets, 'probabilities': probs}]
        out = self.compute_brier(rd, event_only=True, valid_masks=valid_masks)
        # Included positions 3 and 5 only; (0.8-1)^2 each = 0.04 -> mean 0.04
        self.assertAlmostEqual(out['brier_by_class'][P.START], 0.04, places=6)
        self.assertAlmostEqual(out['brier'], 0.04, places=6)

        


if __name__ == '__main__':
    unittest.main()

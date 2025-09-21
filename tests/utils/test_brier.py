import unittest
import numpy as np

from utils.metrics import compute_brier_scores


class TestBrier(unittest.TestCase):
    def test_binary_perfect(self):
        # Two tokens, binary classes
        probs = np.array([[0.1, 0.9], [0.8, 0.2]], dtype=np.float32)
        targets = np.array([1, 0], dtype=int)
        rd = [{'targets': targets, 'probabilities': probs}]
        out = compute_brier_scores(rd)
        # Perfect predictions at 0.9 and 0.8 are not perfect; compute explicitly
        # token1: (0.1-0)^2 + (0.9-1)^2 = 0.01 + 0.01 = 0.02
        # token2: (0.8-1)^2 + (0.2-0)^2 = 0.04 + 0.04 = 0.08
        # mean over tokens: (0.02 + 0.08) / 2 = 0.05
        self.assertAlmostEqual(out['brier'], 0.05, places=6)
        # per-class
        # class 0: ((0.1-0)^2 + (0.8-1)^2)/2 = (0.01 + 0.04)/2 = 0.025
        # class 1: ((0.9-1)^2 + (0.2-0)^2)/2 = (0.01 + 0.04)/2 = 0.025
        self.assertAlmostEqual(out['brier_by_class'][0], 0.025, places=6)
        self.assertAlmostEqual(out['brier_by_class'][1], 0.025, places=6)

    def test_multiclass_simple(self):
        probs = np.array([
            [0.7, 0.2, 0.1],
            [0.1, 0.8, 0.1],
            [0.2, 0.2, 0.6],
        ], dtype=np.float32)
        targets = np.array([0, 1, 2], dtype=int)
        rd = [{'targets': targets, 'probabilities': probs}]
        out = compute_brier_scores(rd)
        # Compute overall per token sum of squared errors then mean
        # token1 vs [1,0,0]: (0.7-1)^2+(0.2-0)^2+(0.1-0)^2 = 0.09+0.04+0.01=0.14
        # token2 vs [0,1,0]: (0.1-0)^2+(0.8-1)^2+(0.1-0)^2 = 0.01+0.04+0.01=0.06
        # token3 vs [0,0,1]: (0.2-0)^2+(0.2-0)^2+(0.6-1)^2 = 0.04+0.04+0.16=0.24
        # mean: (0.14+0.06+0.24)/3 = 0.146666...
        self.assertAlmostEqual(out['brier'], (0.14+0.06+0.24)/3.0, places=6)


if __name__ == '__main__':
    unittest.main()

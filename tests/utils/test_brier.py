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

    def test_min_weight_filtering(self):
        # 3 classes; only class 1 has weight > 1.0
        probs = np.array([
            [0.7, 0.2, 0.1],  # y=0
            [0.1, 0.8, 0.1],  # y=1
            [0.2, 0.2, 0.6],  # y=2
        ], dtype=np.float32)
        targets = np.array([0, 1, 2], dtype=int)
        rd = [{'targets': targets, 'probabilities': probs}]
        class_weights = [1.0, 2.0, 0.0]
        out = compute_brier_scores(rd, class_weights=class_weights, min_weight=1.0)
        # Only class 1 should be included in overall and per-class
        # per-token error for class 1: (p1 - y1)^2
        e0 = (0.2 - 0.0)**2
        e1 = (0.8 - 1.0)**2
        e2 = (0.2 - 0.0)**2
        overall = (e0 + e1 + e2) / 3.0
        self.assertAlmostEqual(out['brier'], overall, places=6)
        self.assertIn(1, out['brier_by_class'])
        self.assertNotIn(0, out['brier_by_class'])
        self.assertNotIn(2, out['brier_by_class'])
        self.assertAlmostEqual(out['brier_by_class'][1], (e0 + e1 + e2) / 3.0, places=6)

    def test_event_only_positions(self):
        # 2 classes; include only positions where target or prediction is class 1
        probs = np.array([
            [0.9, 0.1],  # y=0, pred=0
            [0.6, 0.4],  # y=0, pred=0 (near miss)
            [0.1, 0.9],  # y=1, pred=1
            [0.8, 0.2],  # y=0, pred=0 (should be excluded if class 1 only)
        ], dtype=np.float32)
        targets = np.array([0, 0, 1, 0], dtype=int)
        preds = np.array([0, 0, 1, 0], dtype=int)
        rd = [{'targets': targets, 'probabilities': probs, 'predictions': preds}]
        # Only class 1 allowed
        cw = [0.0, 2.0]
        out = compute_brier_scores(rd, class_weights=cw, min_weight=1.0, event_only=True)
        # Included positions: idx 1 (pred not class1, target not class1) -> excluded; idx 2 (target=1) -> included
        # idx 0,3 are TN for class1 and excluded when event_only=True with class1 only
        # So overall equals per-class for class1 at idx 2 only: (0.9 - 1)^2 = 0.01
        self.assertAlmostEqual(out['brier'], 0.01, places=6)
        self.assertIn(1, out['brier_by_class'])
        self.assertAlmostEqual(out['brier_by_class'][1], 0.01, places=6)


if __name__ == '__main__':
    unittest.main()

import unittest
import numpy as np

from utils.windowing import compute_window_slices, window_weights, blend_logits


class TestWindowingUtils(unittest.TestCase):
    def test_compute_window_slices_basic(self):
        # seq shorter than window
        self.assertEqual(compute_window_slices(5, 10, 5), [(0, 5)])
        # exact fit
        self.assertEqual(compute_window_slices(10, 10, 5), [(0, 10)])
        # overlap coverage with tail
        sl = compute_window_slices(25, 10, 5)
        # Expected: [0,10], [5,15], [10,20], [15,25]
        self.assertEqual(sl, [(0, 10), (5, 15), (10, 20), (15, 25)])

    def test_window_weights_shapes_and_modes(self):
        w_cos = window_weights(7, mode='cosine')
        w_tri = window_weights(7, mode='triangular')
        self.assertEqual(w_cos.shape, (7,))
        self.assertEqual(w_tri.shape, (7,))
        # Center-peaked
        self.assertGreaterEqual(w_cos[3], w_cos[0])
        self.assertGreaterEqual(w_cos[3], w_cos[-1])
        self.assertGreaterEqual(w_tri[3], w_tri[0])
        self.assertGreaterEqual(w_tri[3], w_tri[-1])

    def test_blend_logits_identical_windows_equals_identity(self):
        L = 20
        C = 3
        slices = [(0, L)]
        logits = [np.random.randn(L, C).astype(np.float32)]
        blended = blend_logits(L, slices, logits)
        np.testing.assert_allclose(blended, logits[0], rtol=1e-6, atol=1e-6)

    def test_blend_logits_overlap_weighting_behaviour(self):
        # Two windows overlapping; second window logits are larger by +1.0
        L = 12
        C = 2
        base = np.zeros((L, C), dtype=np.float32)
        w1 = base[:8] + 0.0
        w2 = base[4:] + 1.0
        slices = [(0, 8), (4, 12)]
        blended = blend_logits(L, slices, [w1, w2], weight_mode='triangular')
        # Positions 0-3 only from w1 → ~0; 8-11 only from w2 → ~1
        self.assertTrue(np.allclose(blended[0:4], 0.0, atol=1e-5))
        self.assertTrue(np.allclose(blended[8:12], 1.0, atol=1e-5))

    def test_blend_logits_margin_excludes_edges(self):
        # Two windows covering [0,8) and [4,12); set margin=2 and exclude_edges
        # For positions inside the first/last 2 tokens of each window, weight should be 0
        L = 12
        C = 1
        w1 = np.zeros((8, C), dtype=np.float32) + 1.0  # window1 logits = 1.0
        w2 = np.zeros((8, C), dtype=np.float32) + 2.0  # window2 logits = 2.0
        slices = [(0, 8), (4, 12)]
        # With margin=2 and exclude_edges, positions [0,1] of w1 and [6,7] of w1 get zeroed;
        # positions [4,5] of w2 (local indices 0,1) and [10,11] (local 6,7) get zeroed.
        blended = blend_logits(L, slices, [w1, w2], weight_mode='cosine', margin=2, exclude_edges=True)
        # At absolute positions 0..11, only contributions where local index in [2..5] should count per window
        # Check a few boundary points explicitly
        # pos 0,1: only w1 exists but its edge is excluded -> weight sums should fallback via eps, result ~0
        self.assertTrue(np.allclose(blended[0:2], 0.0, atol=1e-6))
        # pos 2: w1 center allowed, equals 1.0
        self.assertTrue(np.allclose(blended[2], 1.0, atol=1e-6))
        # pos 10,11: only w2 window covers but edges excluded -> ~0
        self.assertTrue(np.allclose(blended[10:12], 0.0, atol=1e-6))



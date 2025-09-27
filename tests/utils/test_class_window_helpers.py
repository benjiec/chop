#!/usr/bin/env python3

import unittest
import random
import numpy as np

from utils.genome import build_class_windows
from utils.constants import GenePredictionClass as P


class TestClassWindowHelpers(unittest.TestCase):
    def test_build_class_windows_excludes_edges(self):
        # One contig, single window covers entire 10-length region
        windows = [(0, 0, 10)]
        # Targets: class tokens only on edges (positions 0..1 and 8..9)
        targets = [np.array([P.START, P.START, 0, 0, 0, 0, 0, 0, P.STOP, P.STOP], dtype=np.int64)]
        classes = [P.START, P.STOP]
        class_windows = build_class_windows(windows, targets, classes_to_balance=classes, exclude_margin_bps=2)
        self.assertEqual(class_windows, {}, "Windows with weighted classes only on edges should be excluded")

    def test_build_class_windows_assigns_highest_weight(self):
        # Both classes present; weight should decide
        windows = [(0, 0, 10)]
        targets = [np.array([0, 0, P.START, 0, P.STOP, 0, 0, 0, 0, 0], dtype=np.int64)]
        classes = [P.START, P.STOP]
        # Prefer STOP when STOP has higher weight
        cw = [1.0]*8
        cw[P.START] = 2.0
        cw[P.STOP] = 5.0
        class_windows = build_class_windows(windows, targets, classes_to_balance=classes, exclude_margin_bps=1, class_weights=cw)
        self.assertIn(P.STOP, class_windows)
        self.assertEqual(class_windows[P.STOP], [0])

        # Prefer START when START has higher weight
        cw2 = [1.0]*8
        cw2[P.START] = 7.0
        cw2[P.STOP] = 2.0
        class_windows = build_class_windows(windows, targets, classes_to_balance=classes, exclude_margin_bps=1, class_weights=cw2)
        self.assertIn(P.START, class_windows)
        self.assertEqual(class_windows[P.START], [0])
        self.assertNotIn(P.STOP, class_windows)

    def test_build_class_windows_tie_breaker_smallest_class(self):
        windows = [(0, 0, 10)]
        # Inner region (1..9) has two START at 2,3 and two STOP at 4,5
        targets = [np.array([0, 0, P.START, P.START, P.STOP, P.STOP, 0, 0, 0, 0], dtype=np.int64)]
        classes = [P.START, P.STOP]
        # With equal weights pick smaller class id (START < STOP)
        class_windows = build_class_windows(windows, targets, classes_to_balance=classes, exclude_margin_bps=1, class_weights=[1.0]*8)
        self.assertEqual(class_windows, {P.START: [0]})

    def test_edge_only_higher_weight_ignored_middle_next_highest_selected(self):
        # Higher weight class (STOP) only on edges; lower weight class (START) in center
        windows = [(0, 0, 10)]
        # Positions 0 and 9: STOP (edge). Position 4: START (center).
        targets = [np.array([P.STOP, 0, 0, 0, P.START, 0, 0, 0, 0, P.STOP], dtype=np.int64)]
        classes = [P.START, P.STOP]
        cw = [1.0]*8
        cw[P.START] = 2.0
        cw[P.STOP] = 5.0
        # Exclude edges of width 2 on both sides, so STOP is ignored
        class_windows = build_class_windows(windows, targets, classes_to_balance=classes, exclude_margin_bps=2, class_weights=cw)
        self.assertEqual(class_windows, {P.START: [0]})

    # Balanced selection is now handled in dataset sampling; no separate helper to test


if __name__ == '__main__':
    unittest.main()



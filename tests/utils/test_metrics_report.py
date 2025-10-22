#!/usr/bin/env python3

import unittest

import numpy as np

from utils.metrics import SequenceResult
from utils.constants import GenePredictionClass as P
from utils.events import build_event_motifs
from utils.metrics_report import compute_event_metrics


class TestMetricsReport(unittest.TestCase):
    def test_compute_event_metrics_structure(self):
        # Build a toy result: sequence 'ATGAAATAA' with START at 0..2, STOP at 6..8
        seq_tokens = np.array([0, 1, 2])  # not used directly by compute_event_metrics
        # Minimal SequenceResult; probabilities/targets drive metrics
        L = 9
        probs = np.zeros((L, len(P.idx_to_cls)), dtype=np.float32)
        probs[:, P.INTERGENIC] = 0.9
        probs[0:3, P.START] = 0.95
        probs[6:9, P.STOP] = 0.96
        targets = np.full(L, P.INTERGENIC, dtype=np.int64)
        targets[0:3] = P.START
        targets[6:9] = P.STOP
        preds = np.argmax(probs, axis=1)

        r = SequenceResult(
            sequence_index=0,
            sequence_tokens=np.array([0]*L, dtype=np.int64),
            targets=targets,
            sequence_id='contig1',
            predictions=preds,
            probabilities=probs,
        )

        dss = {'GT'}
        motifs = build_event_motifs(dss)

        out = compute_event_metrics([r], motifs, min_weight=1.0)
        self.assertIn('generic', out)
        self.assertIn('events', out)
        self.assertIn('brier_overall', out)
        self.assertIn('brier_by_class', out)
        self.assertIn('beta_fits', out)
        # sanity: brier overall within [0,1]
        self.assertGreaterEqual(out['brier_overall'], 0.0)
        self.assertLessEqual(out['brier_overall'], 1.0)


if __name__ == '__main__':
    unittest.main()



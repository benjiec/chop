#!/usr/bin/env python3

import unittest
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from synthetic.gene_boundary.predict_and_analyze import (
    analyze_all_predictions,
    calculate_start_stop_metrics,
    validate_predictions,
    compute_triplet_prob_stats,
    run_predictions,
)
from utils.windowing import compute_window_slices


def encode_sequence(seq: str):
    vocab = {'A': 0, 'T': 1, 'G': 2, 'C': 3, 'N': 4}
    return np.array([vocab.get(ch, 4) for ch in seq], dtype=np.int64)


class TestGeneBoundaryPredictAndAnalyze(unittest.TestCase):
    def test_start_and_stop_classification_and_validation(self):
        # DNA with ATG at 4 and TAA at 10
        dna = "NNNNATGNNNTAA"
        tokens = encode_sequence(dna)

        targets = np.zeros(len(dna), dtype=np.int64)
        targets[4:7] = 2
        targets[10:13] = 4

        predictions = np.zeros(len(dna), dtype=np.int64)
        predictions[4:7] = 2
        predictions[10:13] = 4

        probs = np.zeros((len(dna), 6), dtype=np.float32)
        probs[4:7, 2] = 0.9
        probs[10:13, 4] = 0.8

        results_data = [{
            'sequence_index': 0,
            'sequence_tokens': tokens,
            'targets': targets,
            'predictions': predictions,
            'probabilities': probs,
        }]

        preds = analyze_all_predictions(results_data)
        validate_predictions(results_data, preds)

        start_metrics, stop_metrics = calculate_start_stop_metrics(results_data)
        self.assertEqual(start_metrics['tp'], 1)
        self.assertEqual(stop_metrics['tp'], 1)

    def test_triplet_awareness_for_stop(self):
        dna = "NNNNNNTGANN"
        tokens = encode_sequence(dna)

        targets = np.zeros(len(dna), dtype=np.int64)
        targets[6:9] = 4

        predictions = np.zeros(len(dna), dtype=np.int64)
        predictions[7] = 4

        probs = np.zeros((len(dna), 6), dtype=np.float32)
        probs[7, 4] = 0.9

        results_data = [{
            'sequence_index': 0,
            'sequence_tokens': tokens,
            'targets': targets,
            'predictions': predictions,
            'probabilities': probs,
        }]

        start_metrics, stop_metrics = calculate_start_stop_metrics(results_data)
        self.assertEqual(stop_metrics['tp'], 1)
        self.assertEqual(stop_metrics['fn'], 0)

    def test_compute_triplet_prob_stats(self):
        # Build a small probability array and verify per-site, max, avg
        probs = np.zeros((10, 6), dtype=np.float32)
        # Put class START=2 values at positions 3..5
        probs[3, 2] = 0.1
        probs[4, 2] = 0.6
        probs[5, 2] = 0.3

        stats = compute_triplet_prob_stats(probs, 3, 2)
        self.assertAlmostEqual(stats['pos'], 0.1, places=6)
        self.assertAlmostEqual(stats['max'], 0.6, places=6)
        self.assertAlmostEqual(stats['avg'], (0.1 + 0.6 + 0.3) / 3, places=6)

        # For STOP=4, at the end of array so it falls back to pos
        probs2 = np.zeros((5, 6), dtype=np.float32)
        probs2[4, 4] = 0.7
        stats2 = compute_triplet_prob_stats(probs2, 4, 4)
        self.assertAlmostEqual(stats2['pos'], 0.7, places=6)
        self.assertAlmostEqual(stats2['max'], 0.7, places=6)
        self.assertAlmostEqual(stats2['avg'], 0.7, places=6)

    def test_run_predictions_splits_long_sequences_into_windows(self):
        class LongSeqDummyDataset(Dataset):
            def __init__(self, total_len: int, num_classes: int = 6):
                self.total_len = total_len
                self.num_classes = num_classes
            def __len__(self):
                return 1
            def __getitem__(self, _):
                seq = torch.full((self.total_len,), 4, dtype=torch.long)
                tgt = torch.zeros((self.total_len,), dtype=torch.long)
                return seq, tgt

        class SmallMaxLenDummyModel(torch.nn.Module):
            def __init__(self, max_len: int, num_classes: int = 2):
                super().__init__()
                self.model = torch.nn.Module()
                class Emb:
                    def __init__(self, m):
                        self.max_seq_length = m
                self.model.embedding = Emb(max_len)
                self.num_classes = num_classes
                self.max_len = max_len
                self.calls = 0
                self.call_lengths = []
            def forward(self, x, return_attention: bool = False):
                B, L = x.shape
                assert L <= self.max_len
                self.calls += 1
                self.call_lengths.append(L)
                v = float(self.calls)
                logits = torch.zeros((B, L, self.num_classes), dtype=torch.float32)
                logits[..., 1] = v
                if return_attention:
                    return logits, {}
                return logits

        model_max = 64
        seq_len = 200
        loader = DataLoader(LongSeqDummyDataset(seq_len), batch_size=1, shuffle=False)
        model = SmallMaxLenDummyModel(model_max)

        results = run_predictions(model, loader, device='cpu', return_attention=False)
        self.assertEqual(len(results), 1)
        r = results[0]
        self.assertEqual(r['probabilities'].shape[0], seq_len)
        self.assertEqual(r['predictions'].shape[0], seq_len)

        expected_slices = compute_window_slices(seq_len, window=model_max, stride=model_max // 2)
        self.assertEqual(model.calls, len(expected_slices))
        self.assertTrue(all(l <= model_max for l in model.call_lengths))
        if seq_len > model_max:
            self.assertGreater(model.calls, 1)

        # Validate blending numerically
        window_logits = []
        for i, (s, e) in enumerate(expected_slices, start=1):
            wl = np.zeros((e - s, 2), dtype=np.float32)
            wl[:, 1] = float(i)
            window_logits.append(wl)
        from utils.windowing import blend_logits
        blended = blend_logits(seq_len, expected_slices, window_logits, weight_mode='cosine', margin=None)
        v = blended[:, 1]
        expected_p1 = 1.0 / (1.0 + np.exp(-v))
        np.testing.assert_allclose(r['probabilities'][:, 1], expected_p1, rtol=1e-6, atol=1e-6)


if __name__ == '__main__':
    unittest.main()

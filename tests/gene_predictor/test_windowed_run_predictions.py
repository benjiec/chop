import unittest
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from gene_predictor.predict_and_analyze import run_predictions
from utils.windowing import compute_window_slices


class TestWindowedRunPredictions(unittest.TestCase):
    def test_run_predictions_splits_long_sequences_into_windows(self):
        class LongSeqDummyDataset(Dataset):
            def __init__(self, total_len: int, num_classes: int = 6):
                self.total_len = total_len
                self.num_classes = num_classes
            def __len__(self):
                return 1
            def __getitem__(self, _):
                seq = torch.full((self.total_len,), 4, dtype=torch.long)  # all 'N'
                tgt = torch.zeros((self.total_len,), dtype=torch.long)    # all INTERGENIC
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
                # Assert windowing respects max_len
                assert L <= self.max_len, f"Window length {L} exceeds max_len {self.max_len}"
                self.calls += 1
                self.call_lengths.append(L)
                # Produce predictable window-specific logits: class1 logit = call index, class0 logit = 0
                v = float(self.calls)
                logits = torch.zeros((B, L, self.num_classes), dtype=torch.float32)
                logits[..., 1] = v
                if return_attention:
                    return logits, {}
                return logits

        # Sequence length > model max len should trigger windowing path
        model_max = 64
        seq_len = 200
        loader = DataLoader(LongSeqDummyDataset(seq_len), batch_size=1, shuffle=False)
        model = SmallMaxLenDummyModel(model_max)

        results = run_predictions(
            model,
            loader,
            device='cpu',
            return_attention=False,
            blending_window_margin_bp=0,
            random_prefix_ns=False,
        )
        self.assertEqual(len(results), 1)
        r = results[0]
        self.assertEqual(r['probabilities'].shape[0], seq_len)
        self.assertEqual(r['predictions'].shape[0], seq_len)

        # Verify windowing actually occurred: multiple forwards with lengths <= max_len
        # Default stride is one-third overlap windows
        expected_slices = compute_window_slices(seq_len, window=model_max, stride=model_max // 3)
        self.assertEqual(model.calls, len(expected_slices))
        self.assertTrue(all(l <= model_max for l in model.call_lengths))
        if seq_len > model_max:
            self.assertGreater(model.calls, 1)

        # Verify blending numerically against known window logits
        # Build per-window logits matching the model's outputs (class1 = call_index, class0 = 0)
        window_logits = []
        for i, (s, e) in enumerate(expected_slices, start=1):
            wl = np.zeros((e - s, 2), dtype=np.float32)
            wl[:, 1] = float(i)
            window_logits.append(wl)
        # Import blend_logits only for expected comparison
        from utils.windowing import blend_logits
        blended = blend_logits(seq_len, expected_slices, window_logits, weight_mode='cosine', margin=None)
        # Expected probabilities from blended logits (softmax over 2 classes [0, v])
        v = blended[:, 1]
        expected_p1 = 1.0 / (1.0 + np.exp(-v))
        np.testing.assert_allclose(r['probabilities'][:, 1], expected_p1, rtol=1e-6, atol=1e-6)


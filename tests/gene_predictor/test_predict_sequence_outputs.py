import unittest
import numpy as np
import torch

from gene_predictor.predict_and_analyze import predict_sequence_outputs


class TestPredictSequenceOutputs(unittest.TestCase):
    def test_non_blended_short_sequence(self):
        class DummyModel(torch.nn.Module):
            def __init__(self, max_len: int = 128, num_classes: int = 3):
                super().__init__()
                self.model = torch.nn.Module()
                class Emb:
                    def __init__(self, m):
                        self.max_seq_length = m
                self.model.embedding = Emb(max_len)
                self.num_classes = num_classes
            def forward(self, x, return_attention: bool = False):
                B, L = x.shape
                logits = torch.zeros((B, L, self.num_classes), dtype=torch.float32)
                # Make class 1 slightly higher to get deterministic preds
                logits[..., 1] = 1.0
                if return_attention:
                    return logits, {}
                return logits

        model = DummyModel(max_len=64, num_classes=3)
        seq_len = 20
        seq = torch.full((1, seq_len), 4, dtype=torch.long)
        preds, probs, logits_raw, attn = predict_sequence_outputs(model, model.model.embedding.max_seq_length, seq, device='cpu', return_attention=True, temperature=None)
        self.assertEqual(preds.shape[0], seq_len)
        self.assertEqual(probs.shape, (seq_len, 3))
        self.assertEqual(logits_raw.shape, (seq_len, 3))
        self.assertTrue(np.all(preds == 1))
        self.assertIsInstance(attn, dict)

    def test_blended_long_sequence(self):
        class SmallMaxLenModel(torch.nn.Module):
            def __init__(self, max_len: int = 32, num_classes: int = 2):
                super().__init__()
                self.model = torch.nn.Module()
                class Emb:
                    def __init__(self, m):
                        self.max_seq_length = m
                self.model.embedding = Emb(max_len)
                self.num_classes = num_classes
                self.calls = 0
                self.call_lengths = []
            def forward(self, x, return_attention: bool = False):
                B, L = x.shape
                assert L <= self.model.embedding.max_seq_length
                self.calls += 1
                self.call_lengths.append(L)
                logits = torch.zeros((B, L, self.num_classes), dtype=torch.float32)
                logits[..., 1] = float(self.calls)
                if return_attention:
                    return logits, {}
                return logits

        model = SmallMaxLenModel(max_len=32, num_classes=2)
        seq_len = 101
        seq = torch.full((1, seq_len), 4, dtype=torch.long)
        preds, probs, logits_raw, attn = predict_sequence_outputs(model, model.model.embedding.max_seq_length, seq, device='cpu', return_attention=False, temperature=None,
                                                                 blending_window_margin_bp=0)
        self.assertEqual(preds.shape[0], seq_len)
        self.assertEqual(probs.shape, (seq_len, 2))
        self.assertEqual(logits_raw.shape, (seq_len, 2))
        # Sanity: probabilities for class 1 should be > 0.5
        self.assertTrue(np.all(probs[:, 1] > 0.5))

    def test_max_weight_aggregator_and_random_prefix(self):
        class SmallMaxLenModel(torch.nn.Module):
            def __init__(self, max_len: int = 32, num_classes: int = 2):
                super().__init__()
                self.model = torch.nn.Module()
                class Emb:
                    def __init__(self, m):
                        self.max_seq_length = m
                self.model.embedding = Emb(max_len)
                self.num_classes = num_classes
                self.calls = 0
            def forward(self, x, return_attention: bool = False):
                B, L = x.shape
                assert L <= self.model.embedding.max_seq_length
                self.calls += 1
                logits = torch.zeros((B, L, self.num_classes), dtype=torch.float32)
                # Put increasing score on class 1 by position index to make selection stable
                for i in range(L):
                    logits[:, i, 1] = float(i)
                if return_attention:
                    return logits, {}
                return logits

        model = SmallMaxLenModel(max_len=32, num_classes=2)
        seq_len = 80
        seq = torch.full((1, seq_len), 4, dtype=torch.long)
        preds, probs, logits_raw, attn = predict_sequence_outputs(
            model,
            model.model.embedding.max_seq_length,
            seq,
            device='cpu',
            return_attention=False,
            temperature=None,
            blending_window_margin_bp=8,
            aggregator='max_weight',
            random_prefix_ns=True,
            random_prefix_min=5,
            random_prefix_max=5,
        )
        self.assertEqual(preds.shape[0], seq_len)  # prefix trimmed
        self.assertEqual(probs.shape, (seq_len, 2))
        self.assertEqual(logits_raw.shape, (seq_len, 2))
        self.assertEqual(preds.shape[0], seq_len)
        self.assertEqual(probs.shape, (seq_len, 2))
        self.assertEqual(logits_raw.shape, (seq_len, 2))
        # Sanity: probabilities for class 1 should be > 0.5
        self.assertTrue(np.all(probs[:, 1] > 0.5))


if __name__ == '__main__':
    unittest.main()



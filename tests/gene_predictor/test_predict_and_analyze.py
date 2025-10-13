#!/usr/bin/env python3

import unittest
import tempfile
from pathlib import Path
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from gene_predictor.predict_and_analyze import predict_sequence_outputs, generate_test_data, run_predictions
from utils.constants import GenePredictionClass as P
from utils.constants import EventHeadIdx as H
from gene_decoder import PredictedSequence
from utils.windowing import compute_window_slices, blend_logits


class TestPredictSequenceOutputs(unittest.TestCase):
    def test_non_blended_short_sequence(self):
        class DummyModel(torch.nn.Module):
            def __init__(self, max_len: int = 64, num_classes: int = 3):
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
                logits[..., 1] = 1.0
                if return_attention:
                    return logits, {}
                return logits

        model = DummyModel(max_len=64, num_classes=3)
        seq_len = 20
        seq = torch.full((1, seq_len), 4, dtype=torch.long)
        sr, attn = predict_sequence_outputs(model, model.model.embedding.max_seq_length, seq, device='cpu', return_attention=True)
        self.assertEqual(sr.predictions.shape[0], seq_len)
        self.assertEqual(sr.probabilities.shape, (seq_len, 3))
        self.assertTrue(np.all(sr.predictions == 1))
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
        sr, attn = predict_sequence_outputs(model, model.model.embedding.max_seq_length, seq, device='cpu', return_attention=False,
                                                                 blending_window_margin_bp=0)
        self.assertEqual(sr.predictions.shape[0], seq_len)
        self.assertEqual(sr.probabilities.shape, (seq_len, 2))
        self.assertTrue(np.all(sr.probabilities[:, 1] > 0.5))

    def test_event_head_mode_overrides_probabilities(self):
        class DummyEventHeadModel(torch.nn.Module):
            def __init__(self, max_len: int = 64, num_classes: int = 8):
                super().__init__()
                self.model = torch.nn.Module()
                class Emb:
                    def __init__(self, m):
                        self.max_seq_length = m
                self.model.embedding = Emb(max_len)
                self.num_classes = num_classes
                # Mimic dna_learner.model behavior: store latest event logits here
                setattr(self.model, '_latest_computed_event_logits', None)
                # Minimal config with event-head enablement and maps
                self.config = {
                    'model': {'num_event_heads': 4},
                    'custom': {
                        'event_motifs_by_head_idx': {
                            int(H.START): ['ATG'],
                            int(H.STOP): ['TAA'],
                            int(H.DSS): ['GT'],
                            int(H.ASS): ['AG'],
                        },
                        'head_to_class_id': {
                            int(H.START): int(P.START),
                            int(H.STOP): int(P.STOP),
                            int(H.DSS): int(P.DSS),
                            int(H.ASS): int(P.ASS),
                        },
                    },
                }
            def forward(self, x, return_attention: bool = False):
                B, L = x.shape
                logits = torch.zeros((B, L, self.num_classes), dtype=torch.float32)
                # Provide event logits: per head constants
                ev = torch.zeros((B, L, 4), dtype=torch.float32)
                ev[:, :, int(H.START)] = 4.0
                ev[:, :, int(H.STOP)] = -4.0
                ev[:, :, int(H.DSS)] = 2.0
                ev[:, :, int(H.ASS)] = 0.0
                setattr(self.model, '_latest_computed_event_logits', ev)
                if return_attention:
                    return logits, {}
                return logits

        model = DummyEventHeadModel(max_len=64, num_classes=8)
        # Build a sequence containing one instance for each event
        seq_len = 20
        seq = torch.full((1, seq_len), 4, dtype=torch.long)
        # ATG at 5..7, TAA at 10..12, GT at 14..15, AG at 2..3
        seq[0, 5] = 0; seq[0, 6] = 1; seq[0, 7] = 2
        seq[0,10] = 1; seq[0,11] = 0; seq[0,12] = 0
        seq[0,14] = 2; seq[0,15] = 1
        seq[0, 2] = 0; seq[0, 3] = 2

        event_motifs_by_class = {
            int(P.START): {'ATG'},
            int(P.STOP): {'TAA'},
            int(P.DSS): {'GT'},
            int(P.ASS): {'AG'},
        }
        sr, _ = predict_sequence_outputs(
            model,
            model.model.embedding.max_seq_length,
            seq,
            device='cpu',
            return_attention=False,
            blending_window_margin_bp=0,
            event_motifs_by_class=event_motifs_by_class,
        )
        # Verify event-only probabilities and predictions
        pr = sr.probabilities
        # START 5..7 should be >0.5; non-event positions NaN
        self.assertTrue(np.all(np.isnan(pr[:5, int(P.START)])))
        self.assertGreater(pr[6, int(P.START)], 0.5)
        # STOP 10..12 < 0.5
        self.assertLess(pr[11, int(P.STOP)], 0.5)
        # DSS 14..15 > 0.5 and NaN elsewhere
        self.assertGreater(pr[14, int(P.DSS)], 0.5)
        self.assertTrue(np.isnan(pr[0, int(P.DSS)]))
        # Predictions come from event logits (argmax)
        self.assertEqual(int(sr.predictions[6]), int(P.START))

    def test_random_prefix_and_blending(self):
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
        sr, attn = predict_sequence_outputs(
            model,
            model.model.embedding.max_seq_length,
            seq,
            device='cpu',
            return_attention=False,
            blending_window_margin_bp=8,
            random_prefix_ns=True,
            random_prefix_min=5,
            random_prefix_max=5,
        )
        self.assertEqual(sr.predictions.shape[0], seq_len)  # prefix trimmed
        self.assertEqual(sr.probabilities.shape, (seq_len, 2))
        # Sanity: probabilities for class 1 should be >= 0.5 everywhere due to positive logits
        self.assertTrue(np.all(sr.probabilities[:, 1] >= 0.5))


class TestSequenceIdPropagation(unittest.TestCase):
    def test_sequence_id_in_results_and_decoder_pickle(self):
        exon = "ATG" + ("G" * 10) + "TAA"
        fasta = f">accX|random description\nNNNN{exon}NN\n"
        tsv = (
            "sequence_id\tgene_id\tgene_start\tgene_end\texon_start\texon_end\tstrand\n"
            "accX|random\trandom\t5\t20\t5\t20\t+\n"
        )
        with tempfile.TemporaryDirectory() as td:
            fp = Path(td)
            fna = fp / "seq.fna.gz"
            ann = fp / "ann.tsv"
            import gzip
            with gzip.open(fna, "wt") as f:
                f.write(fasta)
            with open(ann, "w") as f:
                f.write(tsv)

            dl, ds = generate_test_data(str(fna), str(ann), num_contigs=0)
            class DummyModel(torch.nn.Module):
                def __init__(self, max_len: int = 64, num_classes: int = 3):
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
                    logits[..., 1] = 1.0
                    if return_attention:
                        return logits, {}
                    return logits

            model = DummyModel(max_len=64, num_classes=3)
            results = run_predictions(model, dl, device='cpu', return_attention=False,
                                      blending_window_margin_bp=200,
                                      random_prefix_ns=False)
            self.assertGreaterEqual(len(results), 1)
            r0 = results[0]
            self.assertEqual(r0.sequence_id, 'accX|random')

            class_order = ['intergenic', 'gene', 'start']
            items = [
                PredictedSequence(
                    sequence_index=r.sequence_index,
                    sequence=''.join(['N'] * len(r.sequence_tokens)),
                    probabilities=r.probabilities,
                    class_order=class_order,
                    sequence_id=r.sequence_id,
                ) for r in results
            ]
            out_pkl = fp / "items.pkl"
            with open(out_pkl, 'wb') as f:
                pickle.dump(items, f)
            with open(out_pkl, 'rb') as f:
                loaded = pickle.load(f)
            self.assertIsInstance(loaded[0], PredictedSequence)
            self.assertEqual(loaded[0].sequence_id, 'accX|random')


class TestWindowedRunPredictions(unittest.TestCase):
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
        self.assertEqual(r.probabilities.shape[0], seq_len)
        self.assertEqual(r.predictions.shape[0], seq_len)

        expected_slices = compute_window_slices(seq_len, window=model_max, stride=model_max // 3)
        self.assertEqual(model.calls, len(expected_slices))
        self.assertTrue(all(l <= model_max for l in model.call_lengths))
        if seq_len > model_max:
            self.assertGreater(model.calls, 1)

        window_logits = []
        for i, (s, e) in enumerate(expected_slices, start=1):
            wl = np.zeros((e - s, 2), dtype=np.float32)
            wl[:, 1] = float(i)
            window_logits.append(wl)
        blended = blend_logits(seq_len, expected_slices, window_logits, weight_mode='cosine', margin=None)
        v = blended[:, 1]
        expected_p1 = 1.0 / (1.0 + np.exp(-v))
        np.testing.assert_allclose(r.probabilities[:, 1], expected_p1, rtol=1e-6, atol=1e-6)


if __name__ == '__main__':
    unittest.main()



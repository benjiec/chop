#!/usr/bin/env python3

import unittest
import torch
from torch.utils.data import TensorDataset, DataLoader

from gene_predictor.metrics_callback import F1Callback


def encode_sequence_tokens(seq: str):
    vocab = {'A': 0, 'T': 1, 'G': 2, 'C': 3, 'N': 4}
    return torch.tensor([vocab.get(ch, 4) for ch in seq], dtype=torch.long)


class DummyModule:
    def __init__(self, logits, class_weights):
        self._param = torch.nn.Parameter(torch.zeros(1))
        self._logits = logits
        self.logged = {}
        self.config = {'loss': {'class_weights': class_weights}}

    def parameters(self):
        return iter([self._param])

    def model(self, sequences):
        return self._logits

    def eval(self):
        return self

    def log(self, name, value, prog_bar=False, on_epoch=False):
        self.logged[name] = float(value)


class TestMetricsCallback(unittest.TestCase):
    def test_macro_f1_perfect_for_start_and_stop(self):
        # DNA with ATG and three STOP codons
        dna = "NNATGNNTAANNNTAGNNNTGANNN"
        tokens = encode_sequence_tokens(dna)
        L = tokens.size(0)

        num_classes = 8
        START = 2
        STOP = 4

        targets = torch.zeros(L, dtype=torch.long)
        targets[2:5] = START
        targets[8:11] = STOP
        targets[14:17] = STOP
        targets[20:23] = STOP

        logits = torch.zeros(1, L, num_classes)
        logits[:, 2:5, START] = 5.0
        logits[:, 8:11, STOP] = 5.0
        logits[:, 14:17, STOP] = 5.0
        logits[:, 20:23, STOP] = 5.0

        ds = TensorDataset(tokens.unsqueeze(0), targets.unsqueeze(0))
        val_loader = DataLoader(ds, batch_size=1)

        # Boost weights for START/STOP to ensure inclusion in generic metrics
        cw = [1.0] * num_classes
        cw[START] = 10.0
        cw[STOP] = 10.0

        cb = F1Callback(val_loader, print_per_class_every=0)
        mod = DummyModule(logits, cw)
        cb.on_validation_epoch_end(trainer=None, pl_module=mod)

        self.assertIn('val_f1', mod.logged)
        self.assertAlmostEqual(mod.logged['val_f1'], 1.0, places=6)
        # Per-class keys logged (logger-only)
        self.assertIn('val_f1_classes/START', mod.logged)
        self.assertIn('val_f1_classes/STOP', mod.logged)

    def test_macro_f1_partial_for_center_only_prediction(self):
        dna = "NNNNNNTGANN"
        tokens = encode_sequence_tokens(dna)
        L = tokens.size(0)

        num_classes = 8
        STOP = 4

        targets = torch.zeros(L, dtype=torch.long)
        targets[6:9] = STOP

        logits = torch.zeros(1, L, num_classes)
        logits[:, 7, STOP] = 5.0  # predict only center of STOP

        ds = TensorDataset(tokens.unsqueeze(0), targets.unsqueeze(0))
        val_loader = DataLoader(ds, batch_size=1)

        cw = [1.0] * num_classes
        cw[STOP] = 10.0

        cb = F1Callback(val_loader, print_per_class_every=0)
        mod = DummyModule(logits, cw)
        cb.on_validation_epoch_end(trainer=None, pl_module=mod)

        self.assertIn('val_f1', mod.logged)
        self.assertGreaterEqual(mod.logged['val_f1'], 0.0)
        self.assertLessEqual(mod.logged['val_f1'], 1.0)
        self.assertIn('val_f1_classes/STOP', mod.logged)


if __name__ == '__main__':
    unittest.main()



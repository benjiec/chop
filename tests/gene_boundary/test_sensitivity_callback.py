#!/usr/bin/env python3

import unittest
import torch
from torch.utils.data import TensorDataset, DataLoader

from synthetic.gene_boundary.sensitivity_callback import BoundarySensitivityCallback


def encode_sequence_tokens(seq: str):
    vocab = {'A': 0, 'T': 1, 'G': 2, 'C': 3, 'N': 4}
    return torch.tensor([vocab.get(ch, 4) for ch in seq], dtype=torch.long)


class DummyModule:
    def __init__(self, logits):
        self._param = torch.nn.Parameter(torch.zeros(1))
        self._logits = logits
        self.logged = {}

    def parameters(self):
        return iter([self._param])

    def model(self, sequences):
        return self._logits

    def eval(self):
        return self

    def log(self, name, value, prog_bar=False, on_epoch=False):
        self.logged[name] = float(value)


class TestBoundarySensitivityCallback(unittest.TestCase):
    def test_start_and_all_stop_codons_sensitivity(self):
        # DNA with ATG and all three STOP codons: TAA, TAG, TGA
        dna = "NNATGNNTAANNNTAGNNNTGANNN"
        tokens = encode_sequence_tokens(dna)
        L = tokens.size(0)

        # Targets with 6 classes: mark START and STOP triplets
        targets = torch.zeros(L, dtype=torch.long)
        # ATG at pos 2..4
        targets[2:5] = 2
        # TAA at pos 8..10
        targets[8:11] = 4
        # TAG at pos 14..16
        targets[14:17] = 4
        # TGA at pos 20..22
        targets[20:23] = 4

        # Logits predicting correct START and STOP across full triplets
        logits = torch.zeros(1, L, 6)
        logits[:, 2:5, 2] = 5.0  # START
        logits[:, 8:11, 4] = 5.0  # STOP TAA
        logits[:, 14:17, 4] = 5.0  # STOP TAG
        logits[:, 20:23, 4] = 5.0  # STOP TGA

        ds = TensorDataset(tokens.unsqueeze(0), targets.unsqueeze(0))
        val_loader = DataLoader(ds, batch_size=1)

        cb = BoundarySensitivityCallback(val_loader)
        mod = DummyModule(logits)
        cb.on_validation_epoch_end(trainer=None, pl_module=mod)

        self.assertIn('val_start_ss', mod.logged)
        self.assertIn('val_stop_ss', mod.logged)
        self.assertAlmostEqual(mod.logged['val_start_ss'], 1.0, places=6)
        self.assertAlmostEqual(mod.logged['val_stop_ss'], 1.0, places=6)

    def test_triplet_awareness_for_stop_codons(self):
        # DNA with STOP TGA at pos 6..8; predict only at center position
        dna = "NNNNNNTGANN"
        tokens = encode_sequence_tokens(dna)
        L = tokens.size(0)

        targets = torch.zeros(L, dtype=torch.long)
        targets[6:9] = 4  # STOP triplet

        logits = torch.zeros(1, L, 6)
        logits[:, 7, 4] = 5.0  # Predict only center position of STOP

        ds = TensorDataset(tokens.unsqueeze(0), targets.unsqueeze(0))
        val_loader = DataLoader(ds, batch_size=1)

        cb = BoundarySensitivityCallback(val_loader)
        mod = DummyModule(logits)
        cb.on_validation_epoch_end(trainer=None, pl_module=mod)

        self.assertGreaterEqual(mod.logged['val_stop_ss'], 1.0)


if __name__ == '__main__':
    unittest.main()



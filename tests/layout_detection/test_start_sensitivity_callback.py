#!/usr/bin/env python3

import unittest
import torch
import pytorch_lightning as pl


class TestStartSensitivityCallback(unittest.TestCase):
    def test_start_sensitivity_callback_correctness(self):
        from layout_detection.start_sensitivity_callback import StartSensitivityCallback

        # Build two sequences length 16
        def enc(s):
            m = {'A':0,'T':1,'G':2,'C':3,'N':4}
            return torch.tensor([m[ch] for ch in s], dtype=torch.long)

        seq1 = enc("NNNNATGNNNNNNNNN")  # ATG at 4
        seq2 = enc("NNNNATGNNNATGNNN")  # ATG at 4 and 10
        B, L = 2, 16
        sequences = torch.stack([seq1, seq2], dim=0)  # (B,L)

        START = 2
        targets = torch.zeros(B, L, dtype=torch.long)
        # Mark true STARTs at all ATG positions
        for b, dna in enumerate(["NNNNATGNNNNNNNNN","NNNNATGNNNATGNNN"]):
            for i in range(L-2):
                if dna[i:i+3] == 'ATG':
                    targets[b, i] = START
                    targets[b, i+1] = START
                    targets[b, i+2] = START

        # Create logits so preds: TP at pos 4 in both; miss pos 10 in seq2
        logits = torch.zeros(B, L, 3)
        # default: class 0
        logits[:, :, 0] = 1.0
        # make START high at pos 4 (and its +1,+2) in both sequences
        for b in [0,1]:
            for i in [4,5,6]:
                logits[b, i, 2] = 5.0
        # leave pos 10 triplet as non-START (FN)

        class DummyModule(pl.LightningModule):
            def __init__(self, logits):
                super().__init__()
                self._p = torch.nn.Parameter(torch.zeros(1))
                self._logits = logits
                self.logged = None
            def model(self, sequences):
                return self._logits.to(sequences.device)
            def log(self, name, value, prog_bar=False, on_epoch=False):
                self.logged = (name, float(value))

        # Custom one-shot loader that yields the prebuilt batch
        class OneShotLoader:
            def __iter__(self_inner):
                yield sequences, targets

        module = DummyModule(logits)
        cb = StartSensitivityCallback(OneShotLoader())
        trainer = pl.Trainer(max_epochs=0, enable_progress_bar=False, logger=False)
        cb.on_validation_epoch_end(trainer, module)
        name, value = module.logged
        self.assertEqual(name, 'val_start_sensitivity_atg')
        # TP=2 (both seqs at pos4), FN=1 (seq2 at pos10) => 2/3
        self.assertAlmostEqual(value, 2/3, places=6)


if __name__ == '__main__':
    unittest.main()



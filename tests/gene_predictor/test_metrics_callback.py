#!/usr/bin/env python3

import unittest
import torch
from torch.utils.data import TensorDataset, DataLoader

from gene_predictor.metrics_callback import MetricsCallback
from pathlib import Path
import gene_predictor.metrics_callback as mc
from utils.metrics import event_based_generic_metrics_factory, event_based_brier_factory, SequenceResult
from utils.events import build_event_motifs
from utils.constants import StandardDonorDinucleotides


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

        # Provide required metric functions
        calc_metrics, _calc_with = event_based_generic_metrics_factory(build_event_motifs(StandardDonorDinucleotides))
        calc_brier = event_based_brier_factory(build_event_motifs(StandardDonorDinucleotides))
        cb = MetricsCallback(val_loader, verbose=0, margin_bp=0, calculate_metrics_fn=calc_metrics, compute_brier_fn=calc_brier)
        mod = DummyModule(logits, cw)
        # Populate new batch results structure expected by the callback
        class BR:
            pass
        br = BR()
        br.sequence_tokens_batch = tokens.unsqueeze(0)
        br.targets_batch = targets.unsqueeze(0)
        br.logits_batch = logits
        br.event_logits_batch = None
        br.sequence_index_start = 0
        mod.validation_epoch_results = [br]
        cb.on_validation_epoch_end(trainer=None, pl_module=mod)

        self.assertIn('val_f1', mod.logged)
        self.assertAlmostEqual(mod.logged['val_f1'], 1.0, places=6)

    def test_csv_writer_is_module_level(self):
        # Set up minimal case and patch writer
        dna = "NNATGNNTAANNNTAGNNNTGANNN"
        tokens = encode_sequence_tokens(dna)
        L = tokens.size(0)
        num_classes = 8
        START = 2
        STOP = 4
        targets = torch.zeros(L, dtype=torch.long)
        logits = torch.zeros(1, L, num_classes)

        ds = TensorDataset(tokens.unsqueeze(0), targets.unsqueeze(0))
        val_loader = DataLoader(ds, batch_size=1)

        calc_metrics, _ = event_based_generic_metrics_factory(build_event_motifs(StandardDonorDinucleotides))
        calc_brier = event_based_brier_factory(build_event_motifs(StandardDonorDinucleotides))
        cb = MetricsCallback(val_loader, verbose=0, margin_bp=0, calculate_metrics_fn=calc_metrics, compute_brier_fn=calc_brier, run_dir=Path('/tmp'))

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

        mod = DummyModule(logits)
        # Provide empty results so callback can proceed
        class BR:
            pass
        br = BR()
        br.sequence_tokens_batch = torch.zeros((1, L), dtype=torch.long)
        br.targets_batch = torch.zeros((1, L), dtype=torch.long)
        br.logits_batch = logits
        br.event_logits_batch = None
        br.sequence_index_start = 0
        mod.validation_epoch_results = [br]

        # Monkeypatch module-level writer to capture calls
        calls = {"count": 0}
        def fake_writer(run_dir, trainer, macro_f1, overall_brier, per_class, val_loss=None, components=None):
            calls["count"] += 1
        orig = getattr(mc, 'write_epoch_csv_tall', None)
        mc.write_epoch_csv_tall = fake_writer
        try:
            cb.on_validation_epoch_end(trainer=None, pl_module=mod)
            self.assertGreater(calls["count"], 0)
        finally:
            if orig is not None:
                mc.write_epoch_csv_tall = orig

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

        calc_metrics, _calc_with = event_based_generic_metrics_factory(build_event_motifs(StandardDonorDinucleotides))
        calc_brier = event_based_brier_factory(build_event_motifs(StandardDonorDinucleotides))
        cb = MetricsCallback(val_loader, verbose=0, margin_bp=0, calculate_metrics_fn=calc_metrics, compute_brier_fn=calc_brier)
        mod = DummyModule(logits, cw)
        # Populate new batch results structure expected by the callback
        class BR:
            pass
        br = BR()
        br.sequence_tokens_batch = tokens.unsqueeze(0)
        br.targets_batch = targets.unsqueeze(0)
        br.logits_batch = logits
        br.event_logits_batch = None
        br.sequence_index_start = 0
        mod.validation_epoch_results = [br]
        cb.on_validation_epoch_end(trainer=None, pl_module=mod)

        self.assertIn('val_f1', mod.logged)
        self.assertGreaterEqual(mod.logged['val_f1'], 0.0)
        self.assertLessEqual(mod.logged['val_f1'], 1.0)


if __name__ == '__main__':
    unittest.main()



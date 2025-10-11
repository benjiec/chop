#!/usr/bin/env python3

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from pathlib import Path
import numpy as np
import argparse

from dna_learner.model import GenePredictorModule as ModelModule
from utils.genome import AnnotatedGenomeDataset
from utils.metrics import SequenceResult
from utils.events import build_event_motifs
from utils.metrics import event_based_generic_metrics_factory, event_based_brier_factory
from gene_predictor.metrics_callback import MetricsCallback


def load_model(model_path):
    model = ModelModule.load_from_checkpoint(model_path, map_location="cpu", custom_loss_fn=None)
    model.eval()
    model = model.to("cpu")
    cfg = getattr(model, 'config', None)
    if cfg and 'model' in cfg:
        m = cfg['model']
        print(f"✓ Model loaded successfully (layers={m.get('n_layers')}, heads={m.get('n_heads')}, kmer={m.get('kmer_size')})")
    else:
        print("✓ Model loaded successfully")
    return model


def run_test(dataset, model, margin_bp = 200, batch_size = 8):

    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    # Prefer dataset-provided split (e.g., contig-aware) if available
    if hasattr(dataset, 'split') and callable(getattr(dataset, 'split')):
        print("Customized dataset splitting for training and validation")
        train_dataset, val_dataset = dataset.split(train_size, val_size)
    else:
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    print(f"training samples: {len(train_dataset)}")
    print(f"validation samples: {len(val_dataset)}")
    
    device = "cpu"
    with torch.no_grad():
        # Build SequenceResult list over validation loader
        seq_results = []
        for sequences, targets in val_loader:
            sequences = sequences.to(device)
            targets = targets.to(device)
            logits = model.model(sequences)
            seq_results.extend(SequenceResult.from_batch(
                sequence_tokens_batch=sequences,
                targets_batch=targets,
                logits_batch=logits,
                sequence_index_start=len(seq_results),
            ))

        # Set up metrics callback (no CSV) and invoke epoch end
        # Default to standard DSS motif set for metrics in this utility
        event_motifs_by_class = build_event_motifs('standard')
        calc_metrics, _ = event_based_generic_metrics_factory(event_motifs_by_class)
        calc_brier = event_based_brier_factory(event_motifs_by_class)
        cb = MetricsCallback(val_loader, print_per_class_every=0, margin_bp=int(margin_bp),
                             calculate_metrics_fn=calc_metrics, compute_brier_fn=calc_brier, run_dir=None)

        class DummyModule:
            def __init__(self, results):
                self._cb_results_data = results
                self.logged = {}
                self._param = torch.nn.Parameter(torch.zeros(1))
            def parameters(self):
                return iter([self._param])
            def eval(self):
                return self
            def log(self, name, value, prog_bar=False, on_epoch=False):
                try:
                    self.logged[name] = float(value)
                except Exception:
                    self.logged[name] = value

        mod = DummyModule(seq_results)
        cb.on_validation_epoch_end(trainer=None, pl_module=mod)
        print("val_f1:", mod.logged.get('val_f1'))
        print("val_brier:", mod.logged.get('val_brier'))



def main():
    parser = argparse.ArgumentParser(description='Gene prediction analysis')
    parser.add_argument('--fna-fn', type=str, required=True, help='File name for genome sequence in FASTA format')
    parser.add_argument('--tsv-fn', type=str, required=True, help='File name for annotations in TSV format')
    parser.add_argument('--run-dir', type=str, required=True,
                       help='Run directory that contains the checkpoints subdirectory.')
    parser.add_argument('--model-path', type=str, required=True,
                       help='Checkpoint path. If relative, it is resolved under <run-dir>/checkpoints/. Absolute paths are accepted.')
    parser.add_argument('--num-windows', type=int, required=True)
    parser.add_argument('--window-stride', type=int, required=True)

    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    raw_model_path = Path(args.model_path)
    ckpt_path = raw_model_path if raw_model_path.is_absolute() else (run_dir / 'checkpoints' / raw_model_path)
    model = load_model(ckpt_path)

    max_seq_len = int(model.model.embedding.max_seq_length)
    print("config", model.config)

    dataset = AnnotatedGenomeDataset(
        args.fna_fn,
        args.tsv_fn,
        window=max_seq_len,
        stride=args.window_stride,
        num_windows=args.num_windows,
    )

    run_test(dataset, model)

if __name__ == "__main__":
    main()

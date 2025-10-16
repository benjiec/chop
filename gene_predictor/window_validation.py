#!/usr/bin/env python3

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from pathlib import Path
import numpy as np
import argparse

from dna_learner.model import GenePredictorModule as ModelModule
from dna_learner.model import BatchResult
from utils.genome import AnnotatedGenomeDataset
from utils.metrics import SequenceResult
from utils.events import build_event_motifs, build_class_logits_from_event_head_logits
from utils.metrics import event_based_generic_metrics_factory, event_based_brier_factory
from gene_predictor.metrics_callback import MetricsCallback
from utils.constants import StandardDonorDinucleotides, DinoDonorDinucleotides
from utils.constants import GenePredictionClass as P


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


def run_test(dataset, model, dss_set, margin_bp = 200, batch_size = 8):

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
        # Discover event-head settings from config (align with training/predict)
        cfg = getattr(model, 'config', {}) or {}
        ccfg = cfg.get('custom', {}) if isinstance(cfg, dict) else {}
        model_cfg = cfg.get('model', {}) if isinstance(cfg, dict) else {}
        num_event_heads = int(model_cfg.get('num_event_heads') or 0)
        event_motifs_by_class = ccfg.get('event_motifs_by_class')
        head_class_ids = ccfg.get('head_class_ids')

        # Fallback for motifs if not present in config
        if not event_motifs_by_class:
            event_motifs_by_class = build_event_motifs(dss_set)

        # Build BatchResult list over validation loader (like training path)
        batch_results = []
        total_so_far = 0
        for sequences, targets in val_loader:
            sequences = sequences.to(device)
            targets = targets.to(device)

            extras = {}
            logits = model.model(
                sequences,
                extras=extras,
                return_event_logits=('event_logits' if num_event_heads > 0 else None),
            )
            ev = extras['event_logits'] if 'event_logits' in extras else None

            batch_results.append(BatchResult(
                sequence_tokens_batch=sequences.detach(),
                targets_batch=targets.detach(),
                logits_batch=logits.detach(),
                event_logits_batch=(ev.detach() if isinstance(ev, torch.Tensor) else None),
                sequence_index_start=int(total_so_far),
            ))
            total_so_far += int(sequences.size(0))

        # Set up metrics callback (no CSV) and invoke epoch end
        calc_metrics, _ = event_based_generic_metrics_factory(event_motifs_by_class)
        calc_brier = event_based_brier_factory(event_motifs_by_class)

        # Build conversion fn (align with train.py) if event-head mode
        logits_conversion_fn = None
        if num_event_heads > 0:
            def _convert(seq_tokens_batch, event_logits_batch):
                if not (isinstance(event_logits_batch, torch.Tensor) and event_logits_batch.dim() == 3 and int(event_logits_batch.size(0)) >= 1):
                    raise RuntimeError('Event-head mode active but event logits are missing for conversion')
                print("converting logits")
                B = int(event_logits_batch.size(0))
                outs = []
                for b in range(B):
                    wl = build_class_logits_from_event_head_logits(
                        seq_window_tokens=seq_tokens_batch[b:b+1, :],
                        event_logits_window=event_logits_batch[b:b+1, :, :],
                        event_motifs_by_class=event_motifs_by_class,
                        head_class_ids=head_class_ids,
                        num_classes=len(P.idx_to_cls),
                    )
                    outs.append(wl)
                return torch.from_numpy(np.stack(outs, axis=0))
            logits_conversion_fn = _convert

        cb = MetricsCallback(val_loader, verbose=1, margin_bp=int(margin_bp),
                             calculate_metrics_fn=calc_metrics, compute_brier_fn=calc_brier, run_dir=None,
                             event_logits_conversion_fn=logits_conversion_fn,
                             event_motifs_by_class=event_motifs_by_class,
                             head_class_ids=head_class_ids)

        class DummyModule:
            def __init__(self, results):
                self.validation_epoch_results = results
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

        mod = DummyModule(batch_results)
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
    parser.add_argument('--dss-motifs', type=str, required=True, choices=['standard', 'dino'], help='Donor splice site motifs to use for event-based analysis: standard or dino')

    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    raw_model_path = Path(args.model_path)
    ckpt_path = raw_model_path if raw_model_path.is_absolute() else (run_dir / 'checkpoints' / raw_model_path)
    model = load_model(ckpt_path)

    max_seq_len = int(model.model.embedding.max_seq_length)
    print("config", model.config)

    if args.dss_motifs == 'dino':
        dss_set = DinoDonorDinucleotides
    else:
        dss_set = StandardDonorDinucleotides

    if args.num_windows:
        dataset = AnnotatedGenomeDataset(
            args.fna_fn,
            args.tsv_fn,
            window=max_seq_len,
            stride=args.window_stride,
            num_windows=args.num_windows,
            class_weights=model.config["loss"]["class_weights"]
        )
    else:
        dataset = AnnotatedGenomeDataset(
            args.fna_fn,
            args.tsv_fn,
            window=max_seq_len,
            stride=args.window_stride,
            class_weights=model.config["loss"]["class_weights"]
        )

    run_test(dataset, model, dss_set)

if __name__ == "__main__":
    main()

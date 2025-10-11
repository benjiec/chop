#!/usr/bin/env python3

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from pathlib import Path
import numpy as np
import argparse

from utils.constants import StandardDonorDinucleotides, DinoDonorDinucleotides
from utils.losses import event_based_ce_loss_factory, event_based_bce_loss_factory
from utils.events import build_event_motifs
from dna_learner.model import GenePredictorModule as ModelModule
from utils.genome import AnnotatedGenomeDataset
from utils.metrics import calculate_generic_metrics
from utils.constants import GenePredictionClass as P


def load_model(model_path):
    model = ModelModule.load_from_checkpoint(model_path, map_location="cpu")
    model.eval()
    model = model.to("cpu")
    cfg = getattr(model, 'config', None)
    if cfg and 'model' in cfg:
        m = cfg['model']
        print(f"✓ Model loaded successfully (layers={m.get('n_layers')}, heads={m.get('n_heads')}, kmer={m.get('kmer_size')})")
    else:
        print("✓ Model loaded successfully")
    return model


def run_test(dataset, model, loss_type, dss_set, margin_bp = 200, batch_size = 8):

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

    config = model.config
    bce_pos_weight_map = config['custom']['loss']['bce_pos_weights']
    bce_neg_weight_map = config['custom']['loss']['bce_neg_weights']
    ce_weight_map = config['custom']['loss']['ce_class_weights']

    event_motifs_by_class = build_event_motifs(dss_set)

    # Build event motifs map and custom loss
    if loss_type == 'event-ce':
        custom_loss = event_based_ce_loss_factory(
            event_motifs_by_class,
            class_weights=ce_weight_map,
            loss_window_margin_bp=margin_bp,
        )
    else:
        custom_loss = event_based_bce_loss_factory(
            event_motifs_by_class,
            pos_weights=bce_pos_weight_map,
            neg_weights=bce_neg_weight_map,
            loss_window_margin_bp=margin_bp,
        )
    
    print(f"training samples: {len(train_dataset)}")
    print(f"validation samples: {len(val_dataset)}")
    
    device = "cpu"
    losses = []
    with torch.no_grad():
        results_data = []
        for sequences, targets in val_loader:
            sequences = sequences.to(device)
            targets = targets.to(device)
            logits = model.model(sequences)
            predictions = logits.argmax(dim=-1)
            probabilities = torch.softmax(logits, dim=-1)
            loss = self.custom_loss_fn(sequences, targets, logits)
            losses.append(loss)

            batch_size = sequences.size(0)
            for b in range(batch_size):
                seq_tokens = sequences[b].detach().cpu().numpy()
                tgt = targets[b].detach().cpu().numpy()
                pred = predictions[b].detach().cpu().numpy()
                probs = probabilities[b].detach().cpu().numpy()
                results_data.append({
                    'sequence_index': len(results_data),
                    'sequence_tokens': seq_tokens,
                    'targets': tgt,
                    'predictions': pred,
                    'probabilities': probs,
                })
            if len(results_data) % 40 == 0:
                print("Processed", len(results_data), "windows")

        valid_masks = None
        if margin_bp > 0:
            valid_masks = []
            for r in results_data:
                L = len(r['sequence_tokens'])
                mask = [True] * L
                if L > 2 * margin_bp:
                    for i in range(0, margin_bp):
                        mask[i] = False
                    for i in range(L - margin_bp, L):
                        mask[i] = False
                valid_masks.append(mask)



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
        class_weights=class_weights
    )

    run_test(dataset, model)

if __name__ == "__main__":
    main()

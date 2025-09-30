#!/usr/bin/env python3

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from pathlib import Path
import numpy as np
import argparse

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


def run_test(dataset, model):

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
        batch_size=8,
        shuffle=False,
        num_workers=0
    )
    
    print(f"training samples: {len(train_dataset)}")
    print(f"validation samples: {len(val_dataset)}")
    
    device = "cpu"
    with torch.no_grad():
        results_data = []
        for sequences, targets in val_loader:
            sequences = sequences.to(device)
            targets = targets.to(device)
            logits = model.model(sequences)
            predictions = logits.argmax(dim=-1)
            probabilities = torch.softmax(logits, dim=-1)

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
        margin_bp = 200
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

        class_weights = model.config.get('loss', {}).get('class_weights')
        metrics_by_class = calculate_generic_metrics(results_data, class_weights=class_weights, min_weight=1.0, valid_masks=valid_masks)

        # Compute macro-average F1 across all classes returned
        f1_values = []
        per_class_f1 = {}
        for cls_idx, m in metrics_by_class.items():
            tp = float(m.get('tp', 0))
            fp = float(m.get('fp', 0))
            fn = float(m.get('fn', 0))
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            denom = precision + sensitivity
            f1 = (2.0 * precision * sensitivity / denom) if denom > 0.0 else 0.0
            f1_values.append(f1)
            cls_name = P.idx_to_cls.get(int(cls_idx), str(int(cls_idx)))
            per_class_f1[cls_name] = f1
            print(cls_name,"tp",tp,"fp",fp,"fn",fn,"sen",sensitivity,"pre",precision,"f1",f1)

        macro_f1 = float(np.median(f1_values)) if f1_values else 0.0
        print("val_f1", macro_f1)


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

    class_weights = model.config.get('loss', {}).get('class_weights')
    max_seq_len = int(model.model.embedding.max_seq_length)
    print("class weights", class_weights, "max seq len", max_seq_len)

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

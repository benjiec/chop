#!/usr/bin/env python3

import sys
import os
from pathlib import Path

from gene_predictor.trainer import train
import argparse
from datetime import datetime
import numpy as np
from typing import Optional, Dict


from utils.constants import GenePredictionClass as P
from gene_predictor.model import GenePredictorModule, create_base_config
from layout_detection.training_dynamics_callback import TrainingDynamicsCallback
from utils.constants import GenePredictionClass as P
from gene_boundary.sensitivity_callback import BoundarySensitivityCallback
from gene_boundary.layouts import generate_dataset


def create_boundary_config(d_model: int = 512, n_layers: int = 4, n_heads: int = 8,
                           learning_rate: float = 5e-5, max_epochs: int = 25, batch_size: int = 4,
                           use_class_weights: bool = True, start_weight: float = 10.0, stop_weight: float = 10.0, utr_weight: float = 3.0,
                           attention_masks: Optional[Dict[int, int]] = None, kmer_size: int = 0,
                           max_seq_length: int = 1000,
                           use_focal: bool = False, focal_gamma: float = 2.0,
                           focal_alpha: Optional[list] = None,
                           cc_enabled: bool = True,
                           start_before: int = 300, start_after: int = 0,
                           stop_before: int = 0, stop_after: int = 300,
                           cc_gap: int = 0) -> dict:

    # Class weights for START/STOP detection
    # START/STOP codons are rare and important, UTR5 regions provide context
    if use_class_weights:
        # Order must match GenePredictionClass indices
        # [INTERGENIC, UTR5, START, GENE, STOP, UTR3]
        class_weights = [1.0, utr_weight, start_weight, 1.0, stop_weight, utr_weight]
    else:
        class_weights = None
    
    cfg = create_base_config(
        max_seq_length=max_seq_length,
        num_classes=6,
        class_names=[P.idx_to_cls[i] for i in sorted(P.idx_to_cls.keys())],
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        learning_rate=learning_rate,
        max_epochs=max_epochs,
        batch_size=batch_size,
        class_weights=class_weights,
        attention_masks=attention_masks,
        kmer_size=kmer_size,
        use_focal=use_focal,
        focal_gamma=focal_gamma,
        focal_alpha=focal_alpha,
    )

    if cc_enabled:
        cfg['model']['class_conditional_readouts'] = {
            'enabled': True,
            'entries': [
                {'class': 'START', 'before': int(start_before), 'after': int(start_after), 'gap': int(cc_gap)},
                {'class': 'STOP', 'before': int(stop_before), 'after': int(stop_after), 'gap': int(cc_gap)},
            ]
        }

    return cfg


def train_boundaries(d_model: int = 512, n_layers: int = 4, n_heads: int = 8,
                    num_contigs: int = 20, layouts_per_contig: int = 1,
                    learning_rate: float = 5e-5, max_epochs: int = 25, batch_size: int = 4,
                    use_class_weights: bool = True, start_weight: float = 10.0, stop_weight: float = 10.0, utr_weight: float = 3.0,
                    attention_masks: Optional[Dict[int, int]] = None, kmer_size: int = 3,
                    max_seq_length: int = 1000,
                    use_focal: bool = False, focal_gamma: float = 2.0,
                    focal_alpha: Optional[list] = None,
                    cc_enabled: bool = True,
                    start_before: int = 300, start_after: int = 0,
                    stop_before: int = 0, stop_after: int = 300,
                    cc_gap: int = 0):

    # Create config
    config = create_boundary_config(
        d_model=d_model, n_layers=n_layers, n_heads=n_heads,
        learning_rate=learning_rate, max_epochs=max_epochs, batch_size=batch_size,
        use_class_weights=use_class_weights, start_weight=start_weight, stop_weight=stop_weight, utr_weight=utr_weight,
        attention_masks=attention_masks, kmer_size=kmer_size, max_seq_length=max_seq_length,
        use_focal=use_focal, focal_gamma=focal_gamma, focal_alpha=focal_alpha,
        cc_enabled=cc_enabled,
        start_before=start_before, start_after=start_after,
        stop_before=stop_before, stop_after=stop_after,
        cc_gap=cc_gap,
    )
    
    dataset = generate_dataset(num_contigs, max_seq_length, layouts_per_contig)

    # Create output directory early for saving sample data
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"gene_boundary/boundary_run_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # StartSensitivityCallback now imported from layout_detection.start_sensitivity_callback

    def mk_training_dynamic_cb(val_loader):
        return [
            TrainingDynamicsCallback(
                val_loader=val_loader,
                output_dir=output_dir / "training_dynamics",
                analysis_frequency=5
            ),
            BoundarySensitivityCallback(val_loader),
        ]
    
    model, val_loader = train(
        dataset,
        config,
        output_dir,
        mk_training_dynamic_cb,
        monitor_metric='val_event_ss',
        monitor_mode='max',
    )

    print(f"results saved to: {output_dir}")
    return output_dir


def main():
    parser = argparse.ArgumentParser(description="Train START/STOP boundary detection")
    parser.add_argument('--d-model', type=int, default=512, help='Model dimension')
    parser.add_argument('--layers', type=int, default=4, help='Number of transformer layers')
    parser.add_argument('--heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--contigs', type=int, default=1000, help='Number of contigs')
    parser.add_argument('--layouts', type=int, default=1, help='Layouts per contig')
    parser.add_argument('--disable-class-weights', action='store_true', help='Disable class weights')
    parser.add_argument('--start-weight', type=float, default=10.0, help='Weight for START class')
    parser.add_argument('--stop-weight', type=float, default=10.0, help='Weight for STOP class')
    parser.add_argument('--utr-weight', type=float, default=3.0, help='Weight for UTR5 class')
    parser.add_argument('--learning-rate', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=25, help='Maximum epochs')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size')
    parser.add_argument('--attention-masks', type=str, help='Head attention masks: symmetric "head:window", asymmetric "head:before:after", or donut "head:before:gap:after" (e.g., "0:4,1:8:6,2:50:0,3:20:10:0")')
    parser.add_argument('--kmer', type=int, default=0, help='K-mer size for convolution (0=disabled, 3=codon sensitive)')
    parser.add_argument('--max-seq-length', type=int, default=1000, help='Maximum sequence length (also used as dataset window size; stride=max_seq_length/2)')
    # Focal loss options
    parser.add_argument('--use-focal', action='store_true', help='Enable focal loss instead of cross-entropy')
    parser.add_argument('--focal-gamma', type=float, default=2.0, help='Focal loss gamma (focusing parameter)')
    parser.add_argument('--focal-alpha', type=str, default=None, help='Comma-separated per-class alpha weights for focal loss (e.g., "1.0,3.0,8.0"). Defaults to class-weights if omitted')
    # Class-conditional readouts
    parser.add_argument('--cc-enabled', action='store_true', default=True, help='Enable class-conditional readouts for START/STOP (enabled by default)')
    parser.add_argument('--start-before', type=int, default=300, help='CC upstream window for START')
    parser.add_argument('--start-after', type=int, default=0, help='CC downstream window for START')
    parser.add_argument('--stop-before', type=int, default=0, help='CC upstream window for STOP')
    parser.add_argument('--stop-after', type=int, default=300, help='CC downstream window for STOP')
    parser.add_argument('--cc-gap', type=int, default=0, help='Relative donut gap for CC masks')
    
    args = parser.parse_args()
    
    # Parse attention masks (support symmetric, asymmetric, and donut)
    attention_masks = None
    if args.attention_masks:
        attention_masks = {}
        for mask_spec in args.attention_masks.split(','):
            parts = mask_spec.split(':')
            head = int(parts[0])
            
            if len(parts) == 2:
                # Symmetric: head:window
                attention_masks[head] = int(parts[1])
            elif len(parts) == 3:
                # Asymmetric: head:before:after
                before, after = int(parts[1]), int(parts[2])
                attention_masks[head] = (before, after)
            elif len(parts) == 4:
                # Donut: head:before:gap:after
                before, gap, after = int(parts[1]), int(parts[2]), int(parts[3])
                attention_masks[head] = (before, gap, after)
            else:
                raise ValueError(f"Invalid attention mask format: {mask_spec}. Use 'head:window', 'head:before:after', or 'head:before:gap:after'")

    # Parse focal alpha list if provided
    focal_alpha = None
    if args.focal_alpha:
        try:
            focal_alpha = [float(x) for x in args.focal_alpha.split(',') if x.strip() != '']
        except Exception as e:
            raise ValueError(f"Invalid --focal-alpha '{args.focal_alpha}': must be comma-separated floats") from e
    
    # Run test
    output_dir = train_boundaries(
        d_model=args.d_model,
        n_layers=args.layers,
        n_heads=args.heads,
        num_contigs=args.contigs,
        layouts_per_contig=args.layouts,
        learning_rate=args.learning_rate,
        max_epochs=args.epochs,
        batch_size=args.batch_size,
        use_class_weights=not args.disable_class_weights,
        start_weight=args.start_weight,
        stop_weight=args.stop_weight,
        utr_weight=args.utr_weight,
        attention_masks=attention_masks,
        kmer_size=args.kmer,
        max_seq_length=args.max_seq_length,
        use_focal=args.use_focal,
        focal_gamma=args.focal_gamma,
        focal_alpha=focal_alpha,
        cc_enabled=args.cc_enabled,
        start_before=args.start_before,
        start_after=args.start_after,
        stop_before=args.stop_before,
        stop_after=args.stop_after,
        cc_gap=args.cc_gap,
    )

if __name__ == "__main__":
    main()

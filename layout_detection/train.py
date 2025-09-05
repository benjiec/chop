#!/usr/bin/env python3

"""
Test driver for UTR-START context learning.

This tests if transformers can learn biological context:
- Real START codons follow 5' UTRs
- Random ATGs in background are just noise (INTERGENIC)

Classes:
- 0: INTERGENIC (background + decoy ATGs)
- 1: UTR5 (5' UTR sequences)
- 2: START (ATGs that follow UTRs)

The key test: Can the model learn that context (UTR5 -> ATG) determines
whether an ATG should be classified as START vs INTERGENIC?

Usage:
    cd /Users/benjie/git/chop && source chop_env/bin/activate
    python layout_detection/train.py
"""

import sys
import os
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from gene_predictor.trainer import train
from layout_detection.layer_analysis import LayerAnalyzer
import argparse
from datetime import datetime
import numpy as np
from typing import Optional, Dict


from utils.dataset import (
    GenomicSyntheticTestingDataset,
    RandomBasesGenerator,
    RandomUTR5Generator,
    AddATGGenerator,
)
from utils.sequences import KOZAK_SEQUENCES, UTR5_REAL_SEQUENCES, IRES_SEQUENCES
from utils.constants import GenePredictionClass as P
from gene_predictor.model import GenePredictorModule, create_base_config
from layout_detection.training_dynamics_callback import TrainingDynamicsCallback
from utils.constants import GenePredictionClass as P
from layout_detection.start_sensitivity_callback import StartSensitivityCallback


def create_utr_start_config(d_model: int = 504, n_layers: int = 3, n_heads: int = 6,
                           learning_rate: float = 5e-5, max_epochs: int = 25, batch_size: int = 4,
                           use_class_weights: bool = False, start_weight: float = 10.0,
                           attention_masks: Optional[Dict[int, int]] = None, kmer_size: int = 3,
                           max_seq_length: int = 1000,
                           use_focal: bool = False, focal_gamma: float = 2.0,
                           focal_alpha: Optional[list] = None) -> dict:

    # Class weights for UTR-START detection
    # START codons are rare and important, UTR5 regions provide context
    if use_class_weights:
        class_weights = [1.0, 3.0, start_weight]  # [INTERGENIC, UTR5, START]
    else:
        class_weights = None
    
    return create_base_config(
        max_seq_length=max_seq_length,
        num_classes=3,
        class_names=['INTERGENIC', 'UTR5', 'START'],
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


def train_utr_start(d_model: int = 504, n_layers: int = 3, n_heads: int = 6,
                    num_contigs: int = 20, layouts_per_contig: int = 1,
                    learning_rate: float = 5e-5, max_epochs: int = 25, batch_size: int = 4,
                    use_class_weights: bool = False, start_weight: float = 10.0,
                    attention_masks: Optional[Dict[int, int]] = None, kmer_size: int = 3,
                    max_seq_length: int = 1000,
                    use_focal: bool = False, focal_gamma: float = 2.0,
                    focal_alpha: Optional[list] = None):

    # Create config
    config = create_utr_start_config(
        d_model=d_model, n_layers=n_layers, n_heads=n_heads,
        learning_rate=learning_rate, max_epochs=max_epochs, batch_size=batch_size,
        use_class_weights=use_class_weights, start_weight=start_weight,
        attention_masks=attention_masks, kmer_size=kmer_size, max_seq_length=max_seq_length,
        use_focal=use_focal, focal_gamma=focal_gamma, focal_alpha=focal_alpha,
    )
    
    # Build layout per contig: [Random with ATG decoys] -> [UTR5 choice (mutated)] -> [ensure ATG] -> [Random with ATG decoys]
    background_len = 450
    utr_choices = KOZAK_SEQUENCES + UTR5_REAL_SEQUENCES + IRES_SEQUENCES
    layouts = [
        RandomBasesGenerator(length=background_len // 2, target=P.INTERGENIC, random_min_length=background_len // 4),
        RandomBasesGenerator(length=background_len // 2, target=P.INTERGENIC, random_min_length=background_len // 4, avoid="ATG"),
        RandomUTR5Generator(choices=utr_choices, target=P.UTR5, mutation_prob=0.1),
        AddATGGenerator(),
        RandomBasesGenerator(length=background_len // 2, target=P.INTERGENIC, random_min_length=background_len // 4, avoid="ATG"),
        RandomBasesGenerator(length=background_len // 2, target=P.INTERGENIC, decoy="ATG", max_decoy=3, random_min_length=background_len // 4),
    ]

    # One sample per contig; enforce contig length <= max_seq_length
    dataset = GenomicSyntheticTestingDataset(
        max_sequence_length=max_seq_length,
        num_contigs=num_contigs,
        layouts_per_contig=layouts_per_contig,
        layouts=layouts,
    )
   
    # Sanity check 
    for contig_idx in range(dataset.num_contigs):
        full_sequence = dataset.contigs[contig_idx]
        full_targets = dataset.contig_targets[contig_idx]
        
        utr5_positions = np.sum(full_targets == 1)
        total_atgs = 0
        real_start_atgs = 0
        for i in range(len(full_sequence) - 2):
            if full_sequence[i:i+3] == 'ATG':
                total_atgs += 1
                if full_targets[i] == 2:  # Check if this ATG is labeled as START
                    real_start_atgs += 1
        
        print(f"contig {contig_idx}: {real_start_atgs} real START ATGs, {utr5_positions} UTR5 positions, {total_atgs} total ATGs, {len(full_sequence)} bps")
        assert real_start_atgs == layouts_per_contig

    # Create output directory early for saving sample data
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"layout_detection/utr_start_test_run_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # StartSensitivityCallback now imported from layout_detection.start_sensitivity_callback

    def mk_training_dynamic_cb(val_loader):
        return [
            TrainingDynamicsCallback(
                val_loader=val_loader,
                output_dir=output_dir / "training_dynamics",
                analysis_frequency=5
            ),
            StartSensitivityCallback(val_loader),
        ]
    
    model, val_loader = train(
        dataset,
        config,
        output_dir,
        mk_training_dynamic_cb,
        monitor_metric='val_start_sensitivity_atg',
        monitor_mode='max',
    )

    print("running layer analysis")
    run_comprehensive_layer_analysis(model, val_loader, output_dir)
    print(f"results saved to: {output_dir}")
    return output_dir


def run_comprehensive_layer_analysis(model, val_loader, output_dir):
    analysis_dir = output_dir / "layer_analysis"
    analysis_dir.mkdir(exist_ok=True)
    
    analyzer = LayerAnalyzer(model)
    analyzer.analyze_all(val_loader, analysis_dir, max_samples=20)


def main():
    parser = argparse.ArgumentParser(description="Test UTR-START context learning")
    parser.add_argument('--d-model', type=int, default=504, help='Model dimension')
    parser.add_argument('--layers', type=int, default=3, help='Number of transformer layers')
    parser.add_argument('--heads', type=int, default=6, help='Number of attention heads')
    parser.add_argument('--contigs', type=int, default=1000, help='Number of contigs')
    parser.add_argument('--layouts', type=int, default=1, help='Layouts per contig')
    parser.add_argument('--class-weights', action='store_true', help='Use class weights')
    parser.add_argument('--start-weight', type=float, default=10.0, help='Weight for START class')
    parser.add_argument('--learning-rate', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=25, help='Maximum epochs')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size')
    parser.add_argument('--attention-masks', type=str, help='Head attention masks: symmetric "head:window", asymmetric "head:before:after", or donut "head:before:gap:after" (e.g., "0:4,1:8:6,2:50:0,3:20:10:0")')
    parser.add_argument('--kmer', type=int, default=3, help='K-mer size for convolution (0=disabled, 3=default)')
    parser.add_argument('--max-seq-length', type=int, default=1000, help='Maximum sequence length (also used as dataset window size; stride=max_seq_length/2)')
    # Focal loss options
    parser.add_argument('--use-focal', action='store_true', help='Enable focal loss instead of cross-entropy')
    parser.add_argument('--focal-gamma', type=float, default=2.0, help='Focal loss gamma (focusing parameter)')
    parser.add_argument('--focal-alpha', type=str, default=None, help='Comma-separated per-class alpha weights for focal loss (e.g., "1.0,3.0,8.0"). Defaults to class-weights if omitted')
    
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
    output_dir = train_utr_start(
        d_model=args.d_model,
        n_layers=args.layers,
        n_heads=args.heads,
        num_contigs=args.contigs,
        layouts_per_contig=args.layouts,
        learning_rate=args.learning_rate,
        max_epochs=args.epochs,
        batch_size=args.batch_size,
        use_class_weights=args.class_weights,
        start_weight=args.start_weight,
        attention_masks=attention_masks,
        kmer_size=args.kmer,
        max_seq_length=args.max_seq_length,
        use_focal=args.use_focal,
        focal_gamma=args.focal_gamma,
        focal_alpha=focal_alpha,
    )

if __name__ == "__main__":
    main()

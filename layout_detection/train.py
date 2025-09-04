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

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
import argparse
from datetime import datetime
import json
import numpy as np
from typing import Optional, Dict

from utils.dataset import UTRStartDataset
from layout_detection.layout_model import LayoutDetectionModule, create_base_config
from layout_detection.training_dynamics_callback import TrainingDynamicsCallback


def create_utr_start_config(d_model: int = 504, n_layers: int = 3, n_heads: int = 6,
                           learning_rate: float = 5e-5, max_epochs: int = 25, batch_size: int = 4,
                           use_class_weights: bool = False, start_weight: float = 10.0,
                           attention_masks: Optional[Dict[int, int]] = None, kmer_size: int = 3,
                           max_seq_length: int = 1000) -> dict:
    """Create configuration for the UTR-START context test."""
    
    # Class weights for UTR-START detection
    # START codons are rare and important, UTR5 regions provide context
    if use_class_weights:
        class_weights = [1.0, 3.0, start_weight]  # [INTERGENIC, UTR5, START]
    else:
        class_weights = None
    
    return create_base_config(
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
        max_seq_length=max_seq_length
    )


def run_utr_start_test(d_model: int = 504, n_layers: int = 3, n_heads: int = 6,
                      num_contigs: int = 20, layouts_per_contig: int = 10,
                      learning_rate: float = 5e-5, max_epochs: int = 25, batch_size: int = 4,
                      use_class_weights: bool = False, start_weight: float = 10.0,
                      attention_masks: Optional[Dict[int, int]] = None, kmer_size: int = 3,
                      max_seq_length: int = 1000):
    """Run the UTR-START context learning test."""
    
    print(f"=" * 70)
    print(f"UTR-START CONTEXT LEARNING TEST")
    print(f"Model: {n_layers} layers, {n_heads} heads, d_model={d_model}")
    print(f"Data: {num_contigs} contigs, {layouts_per_contig} layouts each")
    print(f"K-mer: {kmer_size if kmer_size > 0 else 'disabled'}")
    if attention_masks:
        mask_str = ','.join([f'{h}:{w}' for h, w in attention_masks.items()])
        print(f"Attention masks: {mask_str}")
    if use_class_weights:
        print(f"Class weights: INTERGENIC=1.0, UTR5=3.0, START={start_weight}")
    else:
        print(f"Class weights: DISABLED")
    print(f"=" * 70)
    
    # Create config
    config = create_utr_start_config(
        d_model=d_model, n_layers=n_layers, n_heads=n_heads,
        learning_rate=learning_rate, max_epochs=max_epochs, batch_size=batch_size,
        use_class_weights=use_class_weights, start_weight=start_weight,
        attention_masks=attention_masks, kmer_size=kmer_size, max_seq_length=max_seq_length
    )
    
    # Create dataset
    print("\n1. Creating UTR-START dataset...")
    # Default dataset windowing based on max_seq_length
    window_size = max_seq_length
    window_stride = max(1, max_seq_length // 2)
    if window_size < max_seq_length:
        print(f"Warning: window_size ({window_size}) < max_seq_length ({max_seq_length})")
    dataset = UTRStartDataset(
        num_contigs=num_contigs,
        layouts_per_contig=layouts_per_contig,
        background_length=500,
        window_size=window_size,
        window_stride=window_stride
    )
    
    # VERIFICATION: Check that full contigs have the expected number of real STARTs
    print("\n1a. Verifying dataset generation...")
    for contig_idx in range(min(3, dataset.num_contigs)):  # Check first 3 full contigs
        full_sequence = dataset.contigs[contig_idx]
        full_targets = dataset.contig_targets[contig_idx]
        
        # Count real STARTs (class 2) and UTR5s (class 1)
        real_starts = np.sum(full_targets == 2)
        utr5_positions = np.sum(full_targets == 1)
        
        # Count ATGs in sequence
        total_atgs = 0
        real_start_atgs = 0
        for i in range(len(full_sequence) - 2):
            if full_sequence[i:i+3] == 'ATG':
                total_atgs += 1
                if full_targets[i] == 2:  # Check if this ATG is labeled as START
                    real_start_atgs += 1
        
        print(f"  Full Contig {contig_idx}: {real_start_atgs} real START ATGs, {utr5_positions} UTR5 positions, {total_atgs} total ATGs")
        
        # Assert expected counts
        expected_start_atgs = layouts_per_contig    # Number of ATG codons
        
        assert real_start_atgs == expected_start_atgs, f"Contig {contig_idx}: Expected {expected_start_atgs} real START ATGs, got {real_start_atgs}"
        
    print("  ✓ Dataset generation verified - all contigs have correct START counts")
    print(f"  ✓ Total training windows: {len(dataset)}")
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['training']['batch_size'], 
        shuffle=True,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['training']['batch_size'], 
        shuffle=False,
        num_workers=0
    )
    
    print(f"  Training windows: {len(train_dataset)}")
    print(f"  Validation windows: {len(val_dataset)}")
    
    # Create output directory early for saving sample data
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"layout_detection/utr_start_test_run_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create model
    print("\n2. Creating model...")
    model = LayoutDetectionModule(config)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Full configuration: {config}")
    
    # Create trainer
    print("\n3. Starting training...")
    
    callbacks = [
        pl.callbacks.ModelCheckpoint(
            dirpath=output_dir / "checkpoints",
            filename='utr_start_model_{epoch:02d}_{val_loss:.3f}',
            monitor='val_loss',
            mode='min',
            save_top_k=3,
            save_last=True
        ),
        pl.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=8,
            mode='min'
        ),
        pl.callbacks.LearningRateMonitor(logging_interval='epoch'),
        TrainingDynamicsCallback(
            val_loader=val_loader,
            output_dir=output_dir / "training_dynamics",
            analysis_frequency=5  # Analyze every 5 epochs
        )
    ]
    
    trainer = pl.Trainer(
        max_epochs=config['training']['max_epochs'],
        accelerator='auto',
        devices='auto',
        callbacks=callbacks,
        enable_progress_bar=True,
        log_every_n_steps=10,
        enable_model_summary=True,
        default_root_dir=output_dir
    )
    
    # Train model
    trainer.fit(model, train_loader, val_loader)
    
    # Test model
    print("\n4. Testing model...")
    trainer.test(model, val_loader)
    
    # Layer analysis
    print("\n5. Comprehensive Layer Analysis...")
    run_comprehensive_layer_analysis(model, dataset, output_dir)
    
    print(f"\nResults saved to: {output_dir}")
    return output_dir


def run_comprehensive_layer_analysis(model, dataset, output_dir):
    """Run comprehensive layer analysis."""
    
    # Create analysis output directory
    analysis_dir = output_dir / "layer_analysis"
    analysis_dir.mkdir(exist_ok=True)
    
    # Import and run layer analysis
    from layout_detection.layer_analysis import LayerAnalyzer
    from torch.utils.data import DataLoader, random_split
    
    # Create validation loader
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=0)
    
    # Run analysis
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
    parser.add_argument('--attention-masks', type=str, help='Head attention masks: symmetric "head:window" or asymmetric "head:before:after" (e.g., "0:1,1:200:0")')
    parser.add_argument('--kmer', type=int, default=3, help='K-mer size for convolution (0=disabled, 3=default)')
    parser.add_argument('--max-seq-length', type=int, default=1000, help='Maximum sequence length (also used as dataset window size; stride=max_seq_length/2)')
    
    args = parser.parse_args()
    
    # Parse attention masks (support both symmetric and asymmetric)
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
            else:
                raise ValueError(f"Invalid attention mask format: {mask_spec}. Use 'head:window' or 'head:before:after'")
    
    # Run test
    output_dir = run_utr_start_test(
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
        max_seq_length=args.max_seq_length
    )
    
    print(f"\nUTR-START context test completed! Results in: {output_dir}")


if __name__ == "__main__":
    main()

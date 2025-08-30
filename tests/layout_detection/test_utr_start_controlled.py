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
    python tests/layout_detection/test_utr_start_controlled.py
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
import argparse
from datetime import datetime
import json
import numpy as np
from typing import Optional, Dict

from tests.layout_detection.utr_start_dataset import UTRStartDataset
from tests.layout_detection.layout_model import LayoutDetectionModule, create_base_config


def save_sample_data(train_loader, output_dir, original_dataset, num_samples=3):
    """Save sample sequences and targets for manual verification."""
    
    print(f"  Saving {num_samples} sample contigs to {output_dir}/sample_data.json")
    
    # Convert indices back to nucleotides
    idx_to_nucleotide = {0: 'A', 1: 'T', 2: 'G', 3: 'C', 4: 'N'}
    class_names = {0: 'INTERGENIC', 1: 'UTR5', 2: 'START'}
    
    samples = []
    batch_count = 0
    
    for batch_idx, batch in enumerate(train_loader):
        sequences, targets = batch
        
        print(f"    Debug: Processing batch {batch_idx}")
        
        # Process first few sequences in this batch
        for i in range(min(num_samples - len(samples), sequences.size(0))):
            seq_indices = sequences[i].numpy()
            target_labels = targets[i].numpy()
            
            # Convert to nucleotide string
            sequence_str = ''.join([idx_to_nucleotide[idx] for idx in seq_indices])
            
            # Find all ATG positions and their classifications
            atg_analysis = []
            for pos in range(len(sequence_str) - 2):
                if sequence_str[pos:pos+3] == 'ATG':
                    atg_class = target_labels[pos]  # Check first position of ATG
                    context_before = sequence_str[max(0, pos-20):pos] if pos >= 20 else sequence_str[:pos]
                    context_after = sequence_str[pos+3:pos+23] if pos+23 < len(sequence_str) else sequence_str[pos+3:]
                    
                    atg_analysis.append({
                        'position': pos,
                        'classification': class_names[atg_class],
                        'context_before': context_before,
                        'context_after': context_after,
                        'is_real_start': bool(atg_class == 2)  # Convert to Python bool
                    })
            
            # Find UTR5 regions
            utr5_regions = []
            current_class = target_labels[0]
            current_start = 0
            
            for pos in range(1, len(target_labels)):
                if target_labels[pos] != current_class:
                    if current_class == 1:  # UTR5 region ended
                        utr5_regions.append({
                            'start': current_start,
                            'end': pos - 1,
                            'length': pos - current_start,
                            'sequence': sequence_str[current_start:pos]
                        })
                    current_class = target_labels[pos]
                    current_start = pos
            
            # Handle last region
            if current_class == 1:
                utr5_regions.append({
                    'start': current_start,
                    'end': len(target_labels) - 1,
                    'length': len(target_labels) - current_start,
                    'sequence': sequence_str[current_start:]
                })
            
            sample = {
                'contig_index': len(samples),
                'sequence_length': len(sequence_str),
                'sequence_preview': sequence_str[:100] + '...',
                'total_atgs': len(atg_analysis),
                'real_starts': sum(1 for atg in atg_analysis if atg['is_real_start']),
                'decoy_atgs': sum(1 for atg in atg_analysis if not atg['is_real_start']),
                'utr5_regions': utr5_regions,
                'atg_analysis': atg_analysis[:10]  # First 10 ATGs for inspection
            }
            
            samples.append(sample)
            
            if len(samples) >= num_samples:
                break
        
        batch_count += 1
        if len(samples) >= num_samples or batch_count >= 3:
            break
    
    # Save to file
    sample_file = output_dir / "sample_data.json"
    with open(sample_file, 'w') as f:
        json.dump(samples, f, indent=2)
    
    # Print summary
    print(f"  Saved {len(samples)} contigs:")
    for i, sample in enumerate(samples):
        print(f"    Contig {i}: {sample['total_atgs']} ATGs ({sample['real_starts']} real STARTs, {sample['decoy_atgs']} decoys)")
        print(f"      UTR5 regions: {len(sample['utr5_regions'])}")


def create_utr_start_config(d_model: int = 504, n_layers: int = 3, n_heads: int = 6,
                           learning_rate: float = 5e-5, max_epochs: int = 25, batch_size: int = 4,
                           use_class_weights: bool = False, start_weight: float = 10.0,
                           attention_masks: Optional[Dict[int, int]] = None, kmer_size: int = 3) -> dict:
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
        kmer_size=kmer_size
    )


def run_utr_start_test(d_model: int = 504, n_layers: int = 3, n_heads: int = 6,
                      num_contigs: int = 20, layouts_per_contig: int = 10,
                      learning_rate: float = 5e-5, max_epochs: int = 25, batch_size: int = 4,
                      use_class_weights: bool = False, start_weight: float = 10.0,
                      attention_masks: Optional[Dict[int, int]] = None, kmer_size: int = 3):
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
        attention_masks=attention_masks, kmer_size=kmer_size
    )
    
    # Create dataset
    print("\n1. Creating UTR-START dataset...")
    dataset = UTRStartDataset(
        num_contigs=num_contigs,
        layouts_per_contig=layouts_per_contig,
        background_length=500,
        window_size=2000,
        window_stride=500
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
    output_dir = Path(f"tests/layout_detection/utr_start_test_run_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save sample data for verification BEFORE training
    print("\n2. Saving sample data for verification...")
    save_sample_data(train_loader, output_dir, dataset)
    
    # Create model
    print("\n3. Creating model...")
    model = LayoutDetectionModule(config)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Full configuration: {config}")
    
    # Create trainer
    print("\n4. Starting training...")
    
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
        pl.callbacks.LearningRateMonitor(logging_interval='epoch')
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
    print("\n5. Testing model...")
    trainer.test(model, val_loader)
    
    # Detailed analysis
    print("\n6. Detailed Analysis...")
    analyze_utr_start_predictions(model, val_loader, output_dir)
    
    print(f"\nResults saved to: {output_dir}")
    return output_dir


def analyze_utr_start_predictions(model, data_loader, output_dir):
    """Analyze UTR-START context learning."""
    
    model.eval()
    device = next(model.parameters()).device
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch in data_loader:
            sequences, targets = batch
            sequences = sequences.to(device)
            targets = targets.to(device)
            
            # Get predictions
            logits = model(sequences)
            predictions = torch.argmax(logits, dim=-1)
            
            # Convert to numpy and collect
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    # Flatten arrays
    pred_flat = np.array(all_predictions).flatten()
    target_flat = np.array(all_targets).flatten()
    
    # Calculate detailed metrics for each class
    results = {}
    class_names = ['INTERGENIC', 'UTR5', 'START']
    
    for class_idx, class_name in enumerate(class_names):
        # True positives, false positives, false negatives, true negatives
        tp = ((pred_flat == class_idx) & (target_flat == class_idx)).sum()
        fp = ((pred_flat == class_idx) & (target_flat != class_idx)).sum()
        fn = ((pred_flat != class_idx) & (target_flat == class_idx)).sum()
        tn = ((pred_flat != class_idx) & (target_flat != class_idx)).sum()
        
        # Calculate metrics
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        
        target_count = (target_flat == class_idx).sum()
        pred_count = (pred_flat == class_idx).sum()
        over_prediction_ratio = pred_count / target_count if target_count > 0 else float('inf')
        
        results[class_name] = {
            'tp': int(tp),
            'fp': int(fp), 
            'fn': int(fn),
            'tn': int(tn),
            'sensitivity': sensitivity,
            'specificity': specificity,
            'precision': precision,
            'target_count': int(target_count),
            'pred_count': int(pred_count),
            'over_prediction_ratio': over_prediction_ratio
        }
    
    # Print detailed results
    print(f"UTR-START CONTEXT TEST RESULTS:")
    print(f"=" * 50)
    
    for class_name, metrics in results.items():
        print(f"\n{class_name}:")
        print(f"  TP: {metrics['tp']}, FP: {metrics['fp']}, FN: {metrics['fn']}, TN: {metrics['tn']}")
        print(f"  Sensitivity: {metrics['sensitivity']:.3f}")
        print(f"  Specificity: {metrics['specificity']:.3f}")
        print(f"  Precision: {metrics['precision']:.3f}")
        print(f"  Over-prediction: {metrics['over_prediction_ratio']:.1f}x")
        print(f"  Targets: {metrics['target_count']}, Predictions: {metrics['pred_count']}")
    
    # Save detailed results
    with open(output_dir / "detailed_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save detailed predictions for verification
    save_detailed_predictions(all_predictions, all_targets, output_dir)
    
    # Critical assessment for context learning
    start_precision = results['START']['precision']
    start_sensitivity = results['START']['sensitivity']
    
    print(f"\n" + "=" * 50)
    print("CRITICAL ASSESSMENT - CONTEXT LEARNING:")
    
    if start_precision > 0.8 and start_sensitivity > 0.8:
        print("✅ SUCCESS: High precision + sensitivity suggests real context learning!")
        print("   Model learned to distinguish real STARTs from decoy ATGs")
    elif start_precision > 0.6 and start_sensitivity > 0.6:
        print("🟡 PARTIAL: Moderate performance, some context learning")
    else:
        print("❌ FAILURE: Low performance suggests model cannot learn biological context")
    
    # Check if model is just finding all ATGs vs learning context
    total_starts = results['START']['target_count']
    total_predictions = results['START']['pred_count']
    
    if total_predictions > total_starts * 2:
        print("⚠️  WARNING: Significant over-prediction - may be finding all ATGs, not just contextual ones")
    
    print(f"=" * 50)


def save_detailed_predictions(all_predictions, all_targets, output_dir, max_windows=10):
    """Save detailed predictions for manual verification."""
    
    print(f"  Saving detailed predictions for first {max_windows} windows...")
    
    # Convert indices back to nucleotides
    idx_to_nucleotide = {0: 'A', 1: 'T', 2: 'G', 3: 'C', 4: 'N'}
    class_names = {0: 'INTERGENIC', 1: 'UTR5', 2: 'START'}
    
    detailed_predictions = []
    
    for window_idx in range(min(max_windows, len(all_predictions))):
        pred_array = all_predictions[window_idx]
        target_array = all_targets[window_idx]
        
        # Convert sequences back to strings (this would require storing sequences too)
        # For now, just save the predictions and targets
        
        # Find positions where predictions differ from targets
        mismatches = []
        matches = []
        
        for pos in range(len(pred_array)):
            pred_class = pred_array[pos]
            target_class = target_array[pos]
            
            if pred_class != target_class:
                mismatches.append({
                    'position': pos,
                    'predicted': class_names[pred_class],
                    'actual': class_names[target_class]
                })
            else:
                matches.append({
                    'position': pos,
                    'class': class_names[pred_class]
                })
        
        # Find START predictions specifically
        start_predictions = []
        start_targets = []
        
        for pos in range(len(pred_array)):
            if pred_array[pos] == 2:  # Predicted START
                start_predictions.append({
                    'position': pos,
                    'correct': target_array[pos] == 2
                })
            if target_array[pos] == 2:  # Actual START
                start_targets.append({
                    'position': pos,
                    'predicted': pred_array[pos] == 2
                })
        
        window_analysis = {
            'window_index': window_idx,
            'sequence_length': len(pred_array),
            'total_mismatches': len(mismatches),
            'accuracy': len(matches) / len(pred_array),
            'start_predictions': start_predictions,
            'start_targets': start_targets,
            'mismatches': mismatches[:20],  # First 20 mismatches
            'predictions': [int(x) for x in pred_array],  # Full prediction array
            'targets': [int(x) for x in target_array]      # Full target array
        }
        
        detailed_predictions.append(window_analysis)
    
    # Save to file
    with open(output_dir / "detailed_predictions.json", 'w') as f:
        json.dump(detailed_predictions, f, indent=2)
    
    print(f"    Saved predictions for {len(detailed_predictions)} windows")
    print(f"    File: detailed_predictions.json")


def main():
    parser = argparse.ArgumentParser(description="Test UTR-START context learning")
    parser.add_argument('--d-model', type=int, default=504, help='Model dimension')
    parser.add_argument('--layers', type=int, default=3, help='Number of transformer layers')
    parser.add_argument('--heads', type=int, default=6, help='Number of attention heads')
    parser.add_argument('--contigs', type=int, default=20, help='Number of contigs')
    parser.add_argument('--layouts', type=int, default=10, help='Layouts per contig')
    parser.add_argument('--class-weights', action='store_true', help='Use class weights')
    parser.add_argument('--start-weight', type=float, default=10.0, help='Weight for START class')
    parser.add_argument('--learning-rate', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=25, help='Maximum epochs')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size')
    parser.add_argument('--attention-masks', type=str, help='Head attention masks: symmetric "head:window" or asymmetric "head:before:after" (e.g., "0:1,1:200:0")')
    parser.add_argument('--kmer', type=int, default=3, help='K-mer size for convolution (0=disabled, 3=default)')
    
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
        kmer_size=args.kmer
    )
    
    print(f"\nUTR-START context test completed! Results in: {output_dir}")


if __name__ == "__main__":
    main()

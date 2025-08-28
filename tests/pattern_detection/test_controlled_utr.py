#!/usr/bin/env python3
"""
Test driver for controlled UTR pattern detection.

This is the definitive test: can transformers learn sequence specificity?

Controlled conditions:
- 5% UTR5 (realistic sequences)
- 5% UTR3 (realistic sequences)  
- 90% INTERGENIC (all Cs - trivial background)

Model: 3 layers, 6 heads, ~4M parameters

Usage:
    cd /Users/benjie/git/chop && source chop_env/bin/activate
    python tests/pattern_detection/test_controlled_utr.py
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

from tests.pattern_detection.controlled_utr_dataset import ControlledUTRDataset
from tests.pattern_detection.controlled_utr_model import ControlledUTRModule


def save_sample_data(train_loader, output_dir, original_dataset, num_samples=3):
    """Save sample sequences and targets for manual verification."""
    
    print(f"  Saving {num_samples} sample sequences to {output_dir}/sample_data.json")
    
    # Convert indices back to nucleotides
    idx_to_nucleotide = {0: 'A', 1: 'T', 2: 'G', 3: 'C', 4: 'N'}
    class_names = {0: 'INTERGENIC', 1: 'UTR5', 2: 'UTR3'}
    
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
            
            # Check background composition
            c_count = sequence_str.count('C')
            background_purity = c_count / len(sequence_str)
            
            print(f"    Debug: Sequence {len(samples)} - Background purity: {background_purity:.1%} Cs")
            
            # Find UTR5 and UTR3 regions
            utr5_regions = []
            utr3_regions = []
            
            # Group consecutive positions of same class
            current_class = target_labels[0]
            current_start = 0
            
            for pos in range(1, len(target_labels)):
                if target_labels[pos] != current_class:
                    # Class changed, record the previous region
                    if current_class == 1:  # UTR5
                        utr5_regions.append({
                            'start': current_start,
                            'end': pos - 1,
                            'length': pos - current_start,
                            'sequence': sequence_str[current_start:pos]
                        })
                    elif current_class == 2:  # UTR3
                        utr3_regions.append({
                            'start': current_start,
                            'end': pos - 1,
                            'length': pos - current_start,
                            'sequence': sequence_str[current_start:pos]
                        })
                    
                    current_class = target_labels[pos]
                    current_start = pos
            
            # Handle the last region
            if current_class == 1:  # UTR5
                utr5_regions.append({
                    'start': current_start,
                    'end': len(target_labels) - 1,
                    'length': len(target_labels) - current_start,
                    'sequence': sequence_str[current_start:]
                })
            elif current_class == 2:  # UTR3
                utr3_regions.append({
                    'start': current_start,
                    'end': len(target_labels) - 1,
                    'length': len(target_labels) - current_start,
                    'sequence': sequence_str[current_start:]
                })
            
            sample = {
                'sequence_index': len(samples),
                'sequence_length': len(sequence_str),
                'background_purity': background_purity,
                'sequence_preview': sequence_str[:100] + '...',
                'targets_preview': target_labels[:100].tolist(),
                'utr5_count': len(utr5_regions),
                'utr3_count': len(utr3_regions),
                'utr5_regions': utr5_regions,
                'utr3_regions': utr3_regions
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
    print(f"  Saved {len(samples)} sequences:")
    for i, sample in enumerate(samples):
        print(f"    Sequence {i}: {sample['background_purity']:.1%} Cs, {sample['utr5_count']} UTR5s, {sample['utr3_count']} UTR3s")


def create_test_config(use_class_weights: bool = False) -> dict:
    """Create configuration for the controlled UTR test."""
    
    # Class weights: slight boost to UTR classes if requested
    if use_class_weights:
        # With 90% intergenic, 5% each UTR, give UTRs a modest boost
        class_weights = [1.0, 2.0, 2.0]  # [INTERGENIC, UTR5, UTR3]
    else:
        class_weights = None
    
    return {
        'model': {
            'vocab_size': 5,      # A, C, G, T, N
            'd_model': 504,       # Divisible by 6 heads (504 = 6 * 84)
            'n_layers': 3,        # As requested
            'n_heads': 6,         # As requested
            'dropout': 0.1,
            'max_seq_length': 1000  # As requested (overkill)
        },
        'training': {
            'learning_rate': 5e-5,  # Lower LR for larger model
            'max_epochs': 25,       # More epochs for controlled test
            'batch_size': 8         # Smaller batch for larger model
        },
        'loss': {
            'class_weights': class_weights
        }
    }


def run_controlled_utr_test(use_class_weights: bool = False):
    """Run the controlled UTR pattern detection test."""
    
    print(f"=" * 70)
    print(f"CONTROLLED UTR PATTERN DETECTION TEST")
    print(f"Model: 3 layers, 6 heads, ~4M parameters")
    print(f"Data: 5% UTR5, 5% UTR3, 90% INTERGENIC (all Cs)")
    print(f"Sequences: 400 total (200 UTR5 + 200 UTR3)")
    if use_class_weights:
        print(f"Class weights: INTERGENIC=1.0, UTR5=2.0, UTR3=2.0")
    else:
        print(f"Class weights: DISABLED")
    print(f"=" * 70)
    
    # Create config
    config = create_test_config(use_class_weights=use_class_weights)
    
    # Create dataset
    print("\n1. Creating controlled dataset...")
    dataset = ControlledUTRDataset(
        sequence_length=1000,
        num_sequences=400,  # 200 each UTR type as requested
        utr5_density=0.05,  # 5% UTR5
        utr3_density=0.05   # 5% UTR3
    )
    
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
    
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}")
    
    # Create output directory early for saving sample data
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"tests/pattern_detection/controlled_test_run_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save sample data for verification BEFORE training
    print("\n2. Saving sample data for verification...")
    save_sample_data(train_loader, output_dir, dataset)
    
    # Create model
    print("\n3. Creating model...")
    model = ControlledUTRModule(config)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # Create trainer
    print("\n4. Starting training...")
    
    callbacks = [
        pl.callbacks.ModelCheckpoint(
            dirpath=output_dir / "checkpoints",
            filename='controlled_utr_model_{epoch:02d}_{val_loss:.3f}',
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
    analyze_controlled_predictions(model, val_loader, output_dir)
    
    print(f"\nResults saved to: {output_dir}")
    return output_dir


def analyze_controlled_predictions(model, data_loader, output_dir):
    """Detailed analysis of controlled predictions."""
    
    model.eval()
    device = next(model.parameters()).device
    
    all_predictions = []
    all_targets = []
    all_sequences = []
    
    # Convert indices back to nucleotides
    idx_to_nucleotide = {0: 'A', 1: 'T', 2: 'G', 3: 'C', 4: 'N'}
    
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
            
            # Convert sequences back to strings
            for seq in sequences.cpu().numpy():
                seq_str = ''.join([idx_to_nucleotide[idx] for idx in seq])
                all_sequences.append(seq_str)
    
    # Analyze overall metrics
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    
    # Flatten for analysis
    pred_flat = all_predictions.flatten()
    target_flat = all_targets.flatten()
    
    # Calculate detailed metrics for each class
    results = {}
    class_names = ['INTERGENIC', 'UTR5', 'UTR3']
    
    for class_idx, class_name in enumerate(class_names):
        # True positives, false positives, false negatives
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
            'sensitivity': sensitivity,
            'specificity': specificity,
            'precision': precision,
            'target_count': int(target_count),
            'pred_count': int(pred_count),
            'over_prediction_ratio': over_prediction_ratio
        }
    
    # Print detailed results
    print(f"CONTROLLED TEST RESULTS:")
    print(f"=" * 50)
    
    for class_name, metrics in results.items():
        print(f"\n{class_name}:")
        print(f"  Sensitivity: {metrics['sensitivity']:.3f}")
        print(f"  Specificity: {metrics['specificity']:.3f}")
        print(f"  Precision: {metrics['precision']:.3f}")
        print(f"  Over-prediction: {metrics['over_prediction_ratio']:.1f}x")
        print(f"  Targets: {metrics['target_count']}, Predictions: {metrics['pred_count']}")
    
    # Save detailed results
    with open(output_dir / "detailed_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    # Critical assessment
    print(f"\n" + "=" * 50)
    print("CRITICAL ASSESSMENT:")
    
    # Check if model learned sequence specificity
    utr5_precision = results['UTR5']['precision']
    utr3_precision = results['UTR3']['precision']
    
    if utr5_precision > 0.8 and utr3_precision > 0.8:
        print("✅ SUCCESS: High precision suggests real pattern learning!")
    elif utr5_precision > 0.5 and utr3_precision > 0.5:
        print("🟡 PARTIAL: Moderate precision, some pattern learning")
    else:
        print("❌ FAILURE: Low precision suggests over-prediction, not pattern learning")
    
    # Check over-prediction
    utr5_over_pred = results['UTR5']['over_prediction_ratio']
    utr3_over_pred = results['UTR3']['over_prediction_ratio']
    
    if utr5_over_pred > 2.0 or utr3_over_pred > 2.0:
        print("⚠️  WARNING: Significant over-prediction detected")
    
    print(f"=" * 50)


def main():
    parser = argparse.ArgumentParser(description="Test controlled UTR pattern detection")
    parser.add_argument('--class-weights', action='store_true', help='Use class weights')
    
    args = parser.parse_args()
    
    # Run test
    output_dir = run_controlled_utr_test(use_class_weights=args.class_weights)
    
    print(f"\nControlled test completed! Results in: {output_dir}")


if __name__ == "__main__":
    main()

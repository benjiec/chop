#!/usr/bin/env python3
"""
Test driver for UTR pattern detection.

This tests if the transformer model can learn complex, statistical patterns:
- UTR5 elements (Kozak, IRES, etc.)
- UTR3 elements (Poly-A signals, AREs, GREs)
- INTERGENIC (random DNA)

This is a more realistic test than simple ATG detection, using the actual
UTR sequences from the gene prediction codebase.

Usage:
    cd /Users/benjie/git/chop && source chop_env/bin/activate
    python tests/pattern_detection/test_utr_pattern_detection.py
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

from tests.pattern_detection.utr_pattern_dataset import UTRPatternDataset
from tests.pattern_detection.utr_pattern_model import UTRPatternModule


def save_sample_data(train_loader, output_dir, original_dataset, num_samples=3):
    """Save sample sequences and targets for manual verification."""
    
    print(f"  Saving {num_samples} sample sequences to {output_dir}/sample_data.json")
    
    # Convert indices back to nucleotides (MUST match the encoding in UTRPatternDataset)
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
            
            # DEBUG: Compare with original dataset
            print(f"    Debug: Checking sequence {len(samples)} from batch")
            print(f"      From batch: {sequence_str[:50]}...")
            
            # Check a few sequences from original dataset to see if any match
            for orig_idx in range(min(5, len(original_dataset.sequences))):
                orig_seq = original_dataset.sequences[orig_idx]
                if orig_seq == sequence_str:
                    print(f"      MATCH: This is original dataset sequence {orig_idx}")
                    orig_targets = original_dataset.targets[orig_idx]
                    print(f"      Original targets match: {np.array_equal(target_labels, orig_targets)}")
                    break
            else:
                print(f"      WARNING: No match found in first 5 original sequences")
            
            # Find UTR5 and UTR3 positions and their labels
            utr5_positions = []
            utr3_positions = []
            
            # Group consecutive positions of same class
            current_class = target_labels[0]
            current_start = 0
            
            for pos in range(1, len(target_labels)):
                if target_labels[pos] != current_class:
                    # Class changed, record the previous region
                    if current_class == 1:  # UTR5
                        utr5_positions.append({
                            'start': current_start,
                            'end': pos - 1,
                            'length': pos - current_start,
                            'sequence': sequence_str[current_start:pos],
                            'class': 'UTR5'
                        })
                    elif current_class == 2:  # UTR3
                        utr3_positions.append({
                            'start': current_start,
                            'end': pos - 1,
                            'length': pos - current_start,
                            'sequence': sequence_str[current_start:pos],
                            'class': 'UTR3'
                        })
                    
                    current_class = target_labels[pos]
                    current_start = pos
            
            # Handle the last region
            if current_class == 1:  # UTR5
                utr5_positions.append({
                    'start': current_start,
                    'end': len(target_labels) - 1,
                    'length': len(target_labels) - current_start,
                    'sequence': sequence_str[current_start:],
                    'class': 'UTR5'
                })
            elif current_class == 2:  # UTR3
                utr3_positions.append({
                    'start': current_start,
                    'end': len(target_labels) - 1,
                    'length': len(target_labels) - current_start,
                    'sequence': sequence_str[current_start:],
                    'class': 'UTR3'
                })
            
            # Find some intergenic samples for comparison
            intergenic_samples = []
            for pos in range(0, len(sequence_str), 50):  # Sample every 50bp
                if pos < len(sequence_str) - 10 and target_labels[pos] == 0:  # INTERGENIC
                    intergenic_samples.append({
                        'position': pos,
                        'sequence': sequence_str[pos:pos+10],
                        'class': 'INTERGENIC'
                    })
            
            sample = {
                'sequence_index': len(samples),
                'sequence_length': len(sequence_str),
                'sequence': sequence_str,
                'targets': [int(x) for x in target_labels],
                'utr5_count': len(utr5_positions),
                'utr3_count': len(utr3_positions),
                'utr5_positions': utr5_positions[:5],  # First 5 UTR5 elements
                'utr3_positions': utr3_positions[:5],  # First 5 UTR3 elements
                'intergenic_samples': intergenic_samples[:5]  # First 5 intergenic samples
            }
            
            samples.append(sample)
            
            if len(samples) >= num_samples:
                break
        
        batch_count += 1
        if len(samples) >= num_samples or batch_count >= 5:  # Don't process too many batches
            break
    
    # Save to file
    sample_file = output_dir / "sample_data.json"
    with open(sample_file, 'w') as f:
        json.dump(samples, f, indent=2)
    
    # Print summary
    print(f"  Saved {len(samples)} sequences:")
    for i, sample in enumerate(samples):
        print(f"    Sequence {i}: {sample['utr5_count']} UTR5 elements, {sample['utr3_count']} UTR3 elements")


def create_test_config(n_heads: int = 4, d_model: int = 256, use_class_weights: bool = False) -> dict:
    """Create configuration for the UTR pattern test."""
    
    # Class weights: balance the classes if requested
    if use_class_weights:
        # Assume roughly 70% intergenic, 15% UTR5, 15% UTR3
        class_weights = [1.0, 4.0, 4.0]  # [INTERGENIC, UTR5, UTR3]
    else:
        class_weights = None
    
    return {
        'model': {
            'vocab_size': 5,      # A, C, G, T, N
            'd_model': d_model,
            'n_layers': 4,
            'n_heads': n_heads,
            'dropout': 0.1,
            'max_seq_length': 1000
        },
        'training': {
            'learning_rate': 1e-4,
            'max_epochs': 20,
            'batch_size': 16
        },
        'loss': {
            'class_weights': class_weights
        }
    }


def run_utr_pattern_test(n_heads: int = 4, d_model: int = 256, num_sequences: int = 1000, use_class_weights: bool = False):
    """Run the UTR pattern detection test."""
    
    print(f"=" * 60)
    print(f"UTR PATTERN DETECTION TEST")
    print(f"Model: {n_heads} heads, d_model={d_model}")
    print(f"Data: {num_sequences} sequences")
    if use_class_weights:
        print(f"Class weights: INTERGENIC=1.0, UTR5=4.0, UTR3=4.0")
    else:
        print(f"Class weights: DISABLED")
    print(f"=" * 60)
    
    # Create config
    config = create_test_config(n_heads=n_heads, d_model=d_model, use_class_weights=use_class_weights)
    
    # Create dataset
    print("\n1. Creating synthetic dataset...")
    dataset = UTRPatternDataset(
        sequence_length=1000,
        num_sequences=num_sequences,
        utr_density=0.3  # 30% of positions should be UTR elements
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
    output_dir = Path(f"tests/pattern_detection/utr_test_run_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save sample data for verification BEFORE training
    print("\n2. Saving sample data for verification...")
    save_sample_data(train_loader, output_dir, dataset)
    
    # Create model
    print("\n3. Creating model...")
    model = UTRPatternModule(config)
    
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Create trainer
    print("\n4. Starting training...")
    
    callbacks = [
        pl.callbacks.ModelCheckpoint(
            dirpath=output_dir / "checkpoints",
            filename='utr_pattern_model_{epoch:02d}_{val_loss:.3f}',
            monitor='val_loss',
            mode='min',
            save_top_k=3,
            save_last=True
        ),
        pl.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
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
    
    # Analyze results
    print("\n6. Analysis...")
    analyze_model_predictions(model, val_loader)
    
    print(f"\nResults saved to: {output_dir}")
    return output_dir


def analyze_model_predictions(model, data_loader):
    """Analyze model predictions to see if it learned UTR patterns."""
    
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
    
    # Overall accuracy
    accuracy = (pred_flat == target_flat).mean()
    
    # Per-class accuracy
    intergenic_mask = (target_flat == 0)
    utr5_mask = (target_flat == 1)
    utr3_mask = (target_flat == 2)
    
    intergenic_accuracy = (pred_flat[intergenic_mask] == target_flat[intergenic_mask]).mean() if intergenic_mask.any() else 0.0
    utr5_accuracy = (pred_flat[utr5_mask] == target_flat[utr5_mask]).mean() if utr5_mask.any() else 0.0
    utr3_accuracy = (pred_flat[utr3_mask] == target_flat[utr3_mask]).mean() if utr3_mask.any() else 0.0
    
    print(f"Pattern recognition analysis:")
    print(f"  Overall accuracy: {accuracy:.4f}")
    print(f"  INTERGENIC accuracy: {intergenic_accuracy:.4f}")
    print(f"  UTR5 accuracy: {utr5_accuracy:.4f}")
    print(f"  UTR3 accuracy: {utr3_accuracy:.4f}")
    
    # Class distribution in predictions vs targets
    print(f"\nClass distribution:")
    for class_idx, class_name in enumerate(['INTERGENIC', 'UTR5', 'UTR3']):
        target_count = (target_flat == class_idx).sum()
        pred_count = (pred_flat == class_idx).sum()
        print(f"  {class_name}: targets={target_count} ({target_count/len(target_flat):.3f}), predictions={pred_count} ({pred_count/len(pred_flat):.3f})")


def main():
    parser = argparse.ArgumentParser(description="Test UTR pattern detection")
    parser.add_argument('--heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--model-size', type=int, default=256, help='Model dimension')
    parser.add_argument('--sequences', type=int, default=100, help='Number of training sequences')
    parser.add_argument('--class-weights', action='store_true', help='Use class weights to balance learning')
    
    args = parser.parse_args()
    
    # Run test
    output_dir = run_utr_pattern_test(
        n_heads=args.heads,
        d_model=args.model_size, 
        num_sequences=args.sequences,
        use_class_weights=args.class_weights
    )
    
    print(f"\nTest completed! Results in: {output_dir}")


if __name__ == "__main__":
    main()

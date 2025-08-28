#!/usr/bin/env python3
"""
Test driver for simple ATG detection.

This tests if the transformer model can learn the most basic pattern:
ATG = START, everything else = INTERGENIC.

Usage:
    cd /Users/benjie/git/chop && source chop_env/bin/activate
    python tests/pattern_detection/test_simple_atg_detection.py
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

from tests.pattern_detection.simple_atg_dataset import SimpleATGDataset
from tests.pattern_detection.simple_atg_model import SimpleATGModule
import json
import numpy as np


def save_sample_data(train_loader, output_dir, original_dataset, num_samples=3):
    """Save sample sequences and targets for manual verification."""
    
    print(f"  Saving {num_samples} sample sequences to {output_dir}/sample_data.json")
    
    # Convert indices back to nucleotides (MUST match the encoding in SimpleATGDataset)
    idx_to_nucleotide = {0: 'A', 1: 'T', 2: 'G', 3: 'C', 4: 'N'}
    
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
            
            # Find ATG positions and their labels
            atg_positions = []
            for pos in range(len(sequence_str) - 2):
                if sequence_str[pos:pos+3] == 'ATG':
                    # Debug: check what we're actually reading
                    actual_labels = [int(target_labels[pos]), int(target_labels[pos+1]), int(target_labels[pos+2])]
                    atg_positions.append({
                        'position': pos,
                        'triplet': sequence_str[pos:pos+3],  # Add the actual triplet for verification
                        'labels': actual_labels,
                        'expected': [1, 1, 1],  # All ATG positions should be labeled as START (class 1)
                        'correct': actual_labels == [1, 1, 1]  # Add correctness check
                    })
            
            # Find some non-ATG positions for comparison
            non_atg_positions = []
            for pos in range(0, len(sequence_str), 50):  # Sample every 50bp
                if pos < len(sequence_str) - 2 and sequence_str[pos:pos+3] != 'ATG':
                    non_atg_positions.append({
                        'position': pos,
                        'triplet': sequence_str[pos:pos+3],
                        'labels': [int(target_labels[pos]), int(target_labels[pos+1]), int(target_labels[pos+2])],
                        'expected': [0, 0, 0]  # Non-ATG should be INTERGENIC (class 0)
                    })
            
            sample = {
                'sequence_index': len(samples),
                'sequence_length': len(sequence_str),
                'sequence': sequence_str,
                'targets': [int(x) for x in target_labels],
                'atg_count': len(atg_positions),
                'atg_positions': atg_positions[:10],  # First 10 ATGs
                'non_atg_samples': non_atg_positions[:10]  # First 10 non-ATG samples
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
        print(f"    Sequence {i}: {sample['atg_count']} ATG codons found")
        
        # Check if ATGs are properly labeled
        correct_atgs = sum(1 for atg in sample['atg_positions'] if atg.get('correct', False))
        total_atgs = len(sample['atg_positions'])
        
        if total_atgs > 0:
            atg_correctness = correct_atgs / total_atgs
            print(f"      ATG labeling accuracy: {atg_correctness:.2%} ({correct_atgs}/{total_atgs} correct)")
            
            # Show first few incorrect ones if any
            incorrect = [atg for atg in sample['atg_positions'] if not atg.get('correct', False)]
            if incorrect:
                print(f"      First incorrect ATG: pos {incorrect[0]['position']}, triplet '{incorrect[0]['triplet']}', labels {incorrect[0]['labels']}")
        else:
            print(f"      No ATGs found")


def create_test_config(n_heads: int = 4, d_model: int = 256, use_class_weights: bool = False, start_weight: float = 10.0) -> dict:
    """Create configuration for the simple ATG test."""
    
    # Class weights: heavily favor START to force learning
    if use_class_weights:
        class_weights = [1.0, start_weight]  # [INTERGENIC, START] - heavily weight START
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


def run_simple_atg_test(n_heads: int = 4, d_model: int = 256, num_sequences: int = 1000, use_class_weights: bool = False, start_weight: float = 10.0):
    """Run the simple ATG detection test."""
    
    print(f"=" * 60)
    print(f"SIMPLE ATG DETECTION TEST")
    print(f"Model: {n_heads} heads, d_model={d_model}")
    print(f"Data: {num_sequences} sequences")
    if use_class_weights:
        print(f"Class weights: INTERGENIC=1.0, START={start_weight}")
    else:
        print(f"Class weights: DISABLED")
    print(f"=" * 60)
    
    # Create config
    config = create_test_config(n_heads=n_heads, d_model=d_model, use_class_weights=use_class_weights, start_weight=start_weight)
    
    # Create dataset
    print("\n1. Creating synthetic dataset...")
    dataset = SimpleATGDataset(
        sequence_length=1000,
        num_sequences=num_sequences,
        atg_density=0.25  # 25% of positions should be ATG starts - balanced test
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
    output_dir = Path(f"tests/pattern_detection/test_run_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save sample data for verification BEFORE training
    print("\n2. Saving sample data for verification...")
    save_sample_data(train_loader, output_dir, dataset)
    
    # Create model
    print("\n3. Creating model...")
    model = SimpleATGModule(config)
    
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Create trainer
    print("\n4. Starting training...")
    
    callbacks = [
        pl.callbacks.ModelCheckpoint(
            dirpath=output_dir / "checkpoints",
            filename='simple_atg_model_{epoch:02d}_{val_loss:.3f}',
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


def analyze_model_predictions(model, val_loader):
    """Analyze what the model learned."""
    model.eval()
    
    # Get a batch of validation data
    batch = next(iter(val_loader))
    sequences, targets = batch
    
    with torch.no_grad():
        logits = model(sequences)
        probabilities = torch.softmax(logits, dim=-1)
        predictions = torch.argmax(logits, dim=-1)
    
    # Flatten everything for analysis
    sequences_flat = sequences.view(-1).cpu().numpy()
    targets_flat = targets.view(-1).cpu().numpy()
    predictions_flat = predictions.view(-1).cpu().numpy()
    start_probs = probabilities[:, :, 1].view(-1).cpu().numpy()  # START class probabilities
    
    # Find all ATG positions
    dna_vocab = ['A', 'T', 'G', 'C', 'N']
    sequence_str = ''.join([dna_vocab[i] for i in sequences_flat])
    
    atg_positions = []
    for i in range(len(sequence_str) - 2):
        if sequence_str[i:i+3] == 'ATG':
            atg_positions.append(i)
    
    # Calculate ATG vs non-ATG START probabilities
    atg_start_probs = [start_probs[pos] for pos in atg_positions if pos < len(start_probs)]
    non_atg_positions = [i for i in range(len(start_probs)) if i not in atg_positions]
    non_atg_start_probs = [start_probs[pos] for pos in non_atg_positions[:1000]]  # Sample 1000
    
    if atg_start_probs and non_atg_start_probs:
        atg_mean = sum(atg_start_probs) / len(atg_start_probs)
        non_atg_mean = sum(non_atg_start_probs) / len(non_atg_start_probs)
        
        print(f"START probability analysis:")
        print(f"  ATG positions: {atg_mean:.4f} (n={len(atg_start_probs)})")
        print(f"  Non-ATG positions: {non_atg_mean:.4f} (n={len(non_atg_start_probs)})")
        print(f"  Ratio (ATG/non-ATG): {atg_mean/non_atg_mean if non_atg_mean > 0 else 'inf':.2f}")
        
        # Check if model learned the pattern
        if atg_mean > non_atg_mean * 2:
            print(f"  ✅ SUCCESS: Model learned ATG → START pattern!")
        else:
            print(f"  ❌ FAILURE: Model did not learn ATG → START pattern")
    
    # Overall accuracy
    accuracy = (predictions_flat == targets_flat).mean()
    start_accuracy = (predictions_flat[targets_flat == 1] == 1).mean() if (targets_flat == 1).sum() > 0 else 0
    intergenic_accuracy = (predictions_flat[targets_flat == 0] == 0).mean() if (targets_flat == 0).sum() > 0 else 0
    
    print(f"\nOverall metrics:")
    print(f"  Total accuracy: {accuracy:.4f}")
    print(f"  START accuracy: {start_accuracy:.4f}")
    print(f"  INTERGENIC accuracy: {intergenic_accuracy:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Test simple ATG detection")
    parser.add_argument('--heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--model-size', type=int, default=256, help='Model dimension')
    parser.add_argument('--sequences', type=int, default=1000, help='Number of training sequences')
    parser.add_argument('--class-weights', action='store_true', help='Use class weights to balance learning')
    parser.add_argument('--start-weight', type=float, default=10.0, help='Weight for START class (default: 10.0)')
    
    args = parser.parse_args()
    
    # Run test
    output_dir = run_simple_atg_test(
        n_heads=args.heads,
        d_model=args.model_size, 
        num_sequences=args.sequences,
        use_class_weights=args.class_weights,
        start_weight=args.start_weight
    )
    
    print(f"\nTest completed! Results in: {output_dir}")


if __name__ == "__main__":
    main()

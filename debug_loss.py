#!/usr/bin/env python3
"""Debug the loss function to see what's causing the issue."""

import torch
from models.gene_predictor import create_model, BiologicalLoss

def debug_loss():
    config = {
        'vocab_size': 5,
        'd_model': 128,
        'n_layers': 2,
        'n_heads': 4,
        'max_seq_length': 1024,
        'dropout': 0.1
    }
    
    model = create_model(config)
    loss_fn = BiologicalLoss()
    model.eval()
    
    # Create sample input
    batch_size = 2
    seq_length = 1024
    sample_input = torch.randint(0, 5, (batch_size, seq_length))
    
    with torch.no_grad():
        outputs = model(sample_input)
    
    print("Model outputs:")
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: shape={value.shape}")
    
    # Test with empty targets (what happens in training)
    empty_targets = {}
    
    try:
        loss = loss_fn(outputs, empty_targets)
        print(f"\nLoss with empty targets: {loss}")
    except Exception as e:
        print(f"\nError with empty targets: {e}")
        print(f"Error type: {type(e)}")
        
        # Debug the dummy targets creation
        print("\nDebugging dummy targets creation...")
        dummy_targets = {
            'gene_boundaries': torch.zeros_like(outputs['gene_boundaries'][:, :, 0]).long(),
            'exon_intron': torch.zeros_like(outputs['exon_intron'][:, :, 0]).long(),
            'splice_sites': torch.zeros_like(outputs['splice_sites'][:, :, 0]).long(),
            'coding_potential': torch.zeros_like(outputs['coding_potential'][:, :, 0]).float()
        }
        
        print("Dummy targets:")
        for key, value in dummy_targets.items():
            print(f"  {key}: shape={value.shape}, dtype={value.dtype}, min={value.min()}, max={value.max()}")
        
        print("\nPredictions for loss calculation:")
        print(f"  gene_boundaries: shape={outputs['gene_boundaries'].shape}")
        print(f"  coding_potential: shape={outputs['coding_potential'].shape}, min={outputs['coding_potential'].min()}, max={outputs['coding_potential'].max()}")
        
        # Test individual loss components
        try:
            coding_pred = torch.clamp(outputs['coding_potential'], 1e-7, 1-1e-7)
            print(f"  clamped coding_potential: min={coding_pred.min()}, max={coding_pred.max()}")
            
            bce_loss = torch.nn.BCELoss()
            coding_loss = bce_loss(coding_pred.view(-1), dummy_targets['coding_potential'].view(-1).float())
            print(f"  coding_loss: {coding_loss}")
        except Exception as e2:
            print(f"  Error in coding loss: {e2}")

if __name__ == "__main__":
    debug_loss()

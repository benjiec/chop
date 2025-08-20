#!/usr/bin/env python3
"""Debug the model output to see what values are being produced."""

import torch
from models.gene_predictor import create_model

def debug_model():
    config = {
        'vocab_size': 5,
        'd_model': 128,
        'n_layers': 2,
        'n_heads': 4,
        'max_seq_length': 1024,
        'dropout': 0.1
    }
    
    model = create_model(config)
    model.eval()
    
    # Create sample input
    batch_size = 2
    seq_length = 1024
    sample_input = torch.randint(0, 5, (batch_size, seq_length))
    
    with torch.no_grad():
        outputs = model(sample_input)
    
    print("Model outputs shapes and value ranges:")
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            print(f"{key}: shape={value.shape}, min={value.min():.4f}, max={value.max():.4f}, mean={value.mean():.4f}")
            
            if key == 'coding_potential':
                print(f"  Coding potential values: {value.flatten()[:10]}")
                print(f"  Are all values in [0,1]? {torch.all((value >= 0) & (value <= 1))}")

if __name__ == "__main__":
    debug_model()

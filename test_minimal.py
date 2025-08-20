#!/usr/bin/env python3
"""
Minimal test to verify the training pipeline works without annotations.
"""

import torch
import yaml
from utils.dna_processor import DNADataset, DNATokenizer
from models.gene_predictor import GenePredictor, create_model
from torch.utils.data import DataLoader


def test_minimal_training():
    """Test basic training setup with minimal synthetic data."""
    
    # Create some synthetic DNA sequences
    sequences = [
        "ATGAAACGCATTAGCAACCCCGATCGATCGAACGCTACGATCGATAA" * 20,  # Make it longer
        "CGATCGATCGAACGCTACGATCGATAATGAAACGCATTAGCAACCCC" * 20,
        "GGCCTTAAGCGATCGATCGAACGCTACGATCGATAATGAAACGCAT" * 20
    ]
    
    # Create dataset without annotations
    dataset = DNADataset(sequences=sequences, annotations=[], max_length=1024)
    
    print(f"Dataset created with {len(dataset)} sequences")
    
    # Test a single item
    item = dataset[0]
    print("Dataset item keys:", list(item.keys()))
    print("Input IDs shape:", item['input_ids'].shape)
    print("Attention mask shape:", item['attention_mask'].shape)
    print("Targets keys:", list(item['targets'].keys()) if item['targets'] else "No targets")
    
    # Create data loader
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)
    
    # Test batch loading
    batch = next(iter(dataloader))
    print("\nBatch keys:", list(batch.keys()))
    print("Batch input_ids shape:", batch['input_ids'].shape)
    print("Batch targets keys:", list(batch['targets'].keys()) if batch['targets'] else "No batch targets")
    
    # Create model
    config = {
        'vocab_size': 5,
        'd_model': 128,
        'n_layers': 2,
        'n_heads': 4,
        'max_seq_length': 1024,
        'dropout': 0.1
    }
    
    model = create_model(config)
    print(f"\nModel created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Test forward pass
    model.eval()
    with torch.no_grad():
        outputs = model(batch['input_ids'])
    
    print("Model outputs:")
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")
    
    print("\nMinimal test completed successfully!")


if __name__ == "__main__":
    test_minimal_training()

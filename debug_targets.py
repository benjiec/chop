#!/usr/bin/env python3
"""Debug target tensor shapes to understand the mismatch."""

import torch
import yaml
from utils.dna_processor import DNADataset, load_fasta_sequences, load_gff_annotations
from torch.utils.data import DataLoader


def debug_targets():
    """Debug the target tensor shapes."""
    
    # Load config
    with open('configs/conservative.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Load data
    sequences = load_fasta_sequences(config['data']['sequences_path'])
    annotations = load_gff_annotations(config['data']['annotations_path'])
    
    print(f"Loaded {len(sequences)} sequences")
    print(f"Loaded {len(annotations)} annotations")
    
    # Create dataset
    dataset = DNADataset(
        sequences=sequences[:10],  # Just first 10 for debugging
        annotations=annotations[:10],
        max_length=config['model']['max_seq_length']
    )
    
    print(f"Dataset created with {len(dataset)} items")
    
    # Test single item
    item = dataset[0]
    print("\nSingle item:")
    print(f"  input_ids: {item['input_ids'].shape}")
    print(f"  targets keys: {list(item['targets'].keys()) if item['targets'] else 'No targets'}")
    
    if item['targets']:
        for key, value in item['targets'].items():
            if isinstance(value, torch.Tensor):
                print(f"  targets['{key}']: {value.shape}, dtype={value.dtype}")
    
    # Test dataloader
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    batch = next(iter(dataloader))
    
    print(f"\nBatch (batch_size=1):")
    print(f"  input_ids: {batch['input_ids'].shape}")
    print(f"  targets keys: {list(batch['targets'].keys()) if batch['targets'] else 'No batch targets'}")
    
    if batch['targets']:
        for key, value in batch['targets'].items():
            if isinstance(value, torch.Tensor):
                print(f"  batch targets['{key}']: {value.shape}, dtype={value.dtype}")
                print(f"    min={value.min()}, max={value.max()}")


if __name__ == "__main__":
    debug_targets()

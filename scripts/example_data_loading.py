#!/usr/bin/env python3
"""
Example script demonstrating the improved data loading system.

Shows how to:
1. Convert GFF to TSV format
2. Load data with sliding windows
3. Use caching and validation
4. Apply data augmentation
"""

import sys
from pathlib import Path
import torch

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from scripts.gff_to_tsv import GFFToTSVConverter
from utils.dna_processor import (
    DNADataset, load_fasta_sequences_with_ids, load_tsv_annotations, 
    map_sequences_to_annotations, DataAugmentation, validate_sequence
)
from utils.constants import DEFAULT_WINDOW_SIZE, DEFAULT_STRIDE


def demonstrate_gff_to_tsv_conversion():
    """Demonstrate GFF to TSV conversion."""
    print("=== GFF to TSV Conversion ===")
    
    # Paths
    gff_path = "data/GCA_001939145.1_filtered_tcov0.99_evalue1e-20.gff"
    tsv_path = "data/GCA_001939145.1_annotations.tsv"
    
    if not Path(gff_path).exists():
        print(f"GFF file not found: {gff_path}")
        print("Use: python scripts/parse_symbiodinium_gff.py (for Symbiodinium data)")
        print("Or:  python scripts/gff_to_tsv.py input.gff output.tsv (for any GFF)")
        return
    
    # Convert
    converter = GFFToTSVConverter(validate_structure=True, verbose=True)
    try:
        converter.convert(gff_path, tsv_path)
        print(f"✓ Conversion completed: {tsv_path}")
    except Exception as e:
        print(f"✗ Conversion failed: {e}")


def demonstrate_sliding_window_loading():
    """Demonstrate sliding window data loading."""
    print("\n=== Sliding Window Data Loading ===")
    
    # Load small test dataset
    sequences_path = "data/GCA_001939145.1.fna"
    tsv_path = "data/GCA_001939145.1_filtered_tcov0.99_evalue1e-20_annotations.tsv"
    
    if not Path(sequences_path).exists():
        print(f"Sequences file not found: {sequences_path}")
        return
    
    # Load sequences with IDs for proper mapping
    print("Loading sequences with IDs...")
    sequences_with_ids = load_fasta_sequences_with_ids(sequences_path, validate=True)
    print(f"Loaded {len(sequences_with_ids)} sequences with IDs")
    
    # Load annotations if available
    annotations = []
    if Path(tsv_path).exists():
        print("Loading annotations from TSV...")
        annotations = load_tsv_annotations(tsv_path)
        print(f"Loaded {len(annotations)} gene annotations")
        
        # Map sequences to annotations
        print("Mapping sequences to annotations...")
        sequences, mapped_annotations = map_sequences_to_annotations(sequences_with_ids, annotations)
    else:
        sequences = [seq for _, seq in sequences_with_ids]
        mapped_annotations = []
    
    # Create dataset with sliding windows
    print("Creating dataset with sliding windows...")
    dataset = DNADataset(
        sequences=sequences[:5],  # Use first 5 sequences for demo
        annotations=mapped_annotations[:5] if mapped_annotations else [],
        max_length=DEFAULT_WINDOW_SIZE,
        use_sliding_windows=True,
        window_size=DEFAULT_WINDOW_SIZE,
        stride=DEFAULT_STRIDE,
        min_gene_coverage=0.5,
        enable_augmentation=False  # Disable for deterministic demo
    )
    
    print(f"Created {len(dataset)} windows from {len(sequences[:5])} sequences")
    
    # Show sample
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"\nSample window:")
        print(f"  - Sequence length: {sample['sequence_length'].item()}")
        print(f"  - Has targets: {bool(sample['targets'])}")
        if 'gene_ids' in sample['targets']:
            unique_genes = torch.unique(sample['targets']['gene_ids'])
            unique_genes = unique_genes[unique_genes != -1]  # Remove UNKNOWN_GENE_ID
            print(f"  - Unique genes in window: {len(unique_genes)}")


def demonstrate_data_augmentation():
    """Demonstrate data augmentation features."""
    print("\n=== Data Augmentation ===")
    
    # Create sample sequence and annotation
    test_sequence = "ATGCGTACGTACGTACGTAG" * 50  # 1000bp sequence
    test_annotation = {
        'genes': [{
            'start': 100,
            'end': 800,
            'strand': '+',
            'gene_id': 'test_gene',
            'exons': [
                {'start': 100, 'end': 300},
                {'start': 500, 'end': 800}
            ],
            'introns': [
                {'start': 300, 'end': 500, 'donor_pos': 300, 'acceptor_pos': 499}
            ]
        }]
    }
    
    # Create augmentation
    augmentor = DataAugmentation(
        reverse_complement_prob=1.0,  # Always apply for demo
        masking_prob=0.5,
        max_mask_length=20
    )
    
    print(f"Original sequence length: {len(test_sequence)}")
    print(f"Original gene: {test_annotation['genes'][0]['start']}-{test_annotation['genes'][0]['end']} ({test_annotation['genes'][0]['strand']})")
    
    # Apply augmentation
    aug_sequence, aug_annotation = augmentor.augment_sequence(test_sequence, test_annotation)
    
    print(f"Augmented sequence length: {len(aug_sequence)}")
    if aug_annotation and 'genes' in aug_annotation:
        aug_gene = aug_annotation['genes'][0]
        print(f"Augmented gene: {aug_gene['start']}-{aug_gene['end']} ({aug_gene['strand']})")


def demonstrate_sequence_validation():
    """Demonstrate sequence validation."""
    print("\n=== Sequence Validation ===")
    
    test_cases = [
        ("ATCGATCGATCG", "Valid DNA sequence"),
        ("ATCGATCGATCGNNNNNATCG", "Sequence with some N bases"),
        ("ATCG" * 100, "Long valid sequence"),
        ("ATCG", "Too short sequence"),
        ("ATCGATCGXYZ", "Invalid bases"),
        ("N" * 1000, "Too many N bases")
    ]
    
    for sequence, description in test_cases:
        is_valid, message = validate_sequence(sequence)
        status = "✓" if is_valid else "✗"
        print(f"{status} {description}: {message}")


def main():
    """Run all demonstrations."""
    print("Data Loading System Demonstration")
    print("=" * 50)
    
    demonstrate_gff_to_tsv_conversion()
    demonstrate_sliding_window_loading()
    demonstrate_data_augmentation()
    demonstrate_sequence_validation()
    
    print("\n" + "=" * 50)
    print("Demonstration completed!")


if __name__ == '__main__':
    main()

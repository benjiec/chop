#!/usr/bin/env python3
"""
CHOP Gene Prediction - Usage Examples

This file demonstrates how to use the CHOP gene prediction tool for:
1. Training a model
2. Making predictions
3. Processing results
"""

import os
import sys
import yaml
import torch
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from models.gene_predictor import GenePredictor, create_model
from utils.dna_processor import DNATokenizer, BiologicalFeatureExtractor, DNADataset
from inference.predict import GenePredictorInference


def example_1_basic_tokenization():
    """Example 1: Basic DNA sequence tokenization and feature extraction"""
    print("=== Example 1: Basic DNA Processing ===")
    
    # Sample DNA sequence
    dna_sequence = "ATGAAACGCATTAGCAACCCCGATCGATCGAACGCTACGATCGATAA"
    
    # Initialize tokenizer
    tokenizer = DNATokenizer()
    
    # Tokenize sequence
    tokens = tokenizer.tokenize(dna_sequence)
    print(f"Original sequence: {dna_sequence}")
    print(f"Tokenized: {tokens}")
    print(f"Detokenized: {tokenizer.detokenize(tokens)}")
    
    # Find biological features
    start_codons = tokenizer.find_start_codons(dna_sequence)
    stop_codons = tokenizer.find_stop_codons(dna_sequence)
    donor_sites, acceptor_sites = tokenizer.find_splice_sites(dna_sequence)
    
    print(f"Start codons at positions: {start_codons}")
    print(f"Stop codons at positions: {stop_codons}")
    print(f"Donor splice sites at: {donor_sites}")
    print(f"Acceptor splice sites at: {acceptor_sites}")
    
    # Extract comprehensive features
    feature_extractor = BiologicalFeatureExtractor()
    features = feature_extractor.extract_features(dna_sequence)
    
    print(f"GC content: {features['gc_content']:.2f}%")
    print(f"Sequence length: {features['length']}")
    print(f"3-mer frequencies (first 5): {list(features['kmer_frequencies'].items())[:5]}")
    print()


def example_2_model_creation_and_forward_pass():
    """Example 2: Create model and run a forward pass"""
    print("=== Example 2: Model Creation and Forward Pass ===")
    
    # Load default configuration
    with open('configs/default.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create model
    model = create_model(config['model'])
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Sample input
    batch_size = 2
    seq_length = 512
    sample_input = torch.randint(0, 5, (batch_size, seq_length))  # Random DNA tokens
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        outputs = model(sample_input)
    
    print("Model outputs:")
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")
    print()


def example_3_training_setup():
    """Example 3: How to set up training (without actually training)"""
    print("=== Example 3: Training Setup ===")
    
    # This shows how you would set up training
    print("To train the model, you would:")
    print("1. Prepare your data:")
    print("   - DNA sequences in FASTA format")
    print("   - Gene annotations in GFF format")
    print()
    print("2. Run training command:")
    print("   python training/train.py --config configs/default.yaml")
    print()
    print("3. Monitor training with:")
    print("   - TensorBoard logs in 'logs/' directory")
    print("   - Model checkpoints in 'models/checkpoints/'")
    print()
    
    # Example of creating a simple dataset
    sample_sequences = [
        "ATGAAACGCATTAGCAACCCCGATCGATCGAACGCTACGATCGATAA",
        "CGATCGATCGAACGCTACGATCGATAATGAAACGCATTAGCAACCCC",
        "GGCCTTAAGCGATCGATCGAACGCTACGATCGATAATGAAACGCAT"
    ]
    
    dataset = DNADataset(sequences=sample_sequences, max_length=1024)
    print(f"Created dataset with {len(dataset)} sequences")
    
    # Show what a single data item looks like
    item = dataset[0]
    print("Sample dataset item keys:", list(item.keys()))
    print("Input shape:", item['input_ids'].shape)
    print("Attention mask shape:", item['attention_mask'].shape)
    print()


def example_4_prediction_workflow():
    """Example 4: How to make predictions (simulated since we don't have a trained model)"""
    print("=== Example 4: Prediction Workflow ===")
    
    print("To make predictions with a trained model:")
    print()
    print("1. Single sequence prediction:")
    print("   python inference/predict.py \\")
    print("     --model models/best.pt \\")
    print("     --config configs/default.yaml \\")
    print("     --input 'ATGAAACGCATTAGCAACCCCGATCGATCGAACGCTACGATCGATAA' \\")
    print("     --output results/ \\")
    print("     --visualize")
    print()
    print("2. FASTA file prediction:")
    print("   python inference/predict.py \\")
    print("     --model models/best.pt \\")
    print("     --config configs/default.yaml \\")
    print("     --input sequences.fasta \\")
    print("     --output results/ \\")
    print("     --visualize")
    print()
    
    # Simulate what the prediction results would look like
    sample_results = {
        'sequence_length': 47,
        'genes': [
            {'start': 0, 'end': 30, 'length': 30, 'sequence': 'ATGAAACGCATTAGCAACCCCGATCGATCG'}
        ],
        'exons': [
            {'start': 0, 'end': 15, 'length': 16},
            {'start': 20, 'end': 30, 'length': 11}
        ],
        'introns': [
            {'start': 15, 'end': 20, 'length': 6}
        ],
        'splice_sites': {
            'donor_sites': [16, 28],
            'acceptor_sites': [19]
        },
        'coding_regions': [
            {'start': 0, 'end': 30, 'length': 31}
        ]
    }
    
    print("Example prediction results structure:")
    print(yaml.dump(sample_results, default_flow_style=False))


def example_5_custom_splice_sites():
    """Example 5: Working with splice sites and customization"""
    print("=== Example 5: Splice Sites and Customization ===")
    
    # Show current splice site definitions
    tokenizer = DNATokenizer()
    print("Current splice site motifs:")
    print(f"  Donor sites (5'): {tokenizer.donor_motifs}")
    print(f"  Acceptor sites (3'): {tokenizer.acceptor_motifs}")
    print()
    
    # Example of how to customize splice sites
    print("To customize splice sites, you can modify the DNATokenizer:")
    print("""
# Custom tokenizer with additional splice sites
class CustomDNATokenizer(DNATokenizer):
    def __init__(self):
        super().__init__()
        # Add more splice site motifs
        self.donor_motifs = ['GT', 'GC', 'AT']  # Added AT
        self.acceptor_motifs = ['AG', 'AC', 'TG']  # Added TG
    """)
    
    # Test splice site detection
    test_sequence = "CGATGTCGAACGTACGAGCTAACGATCG"
    donor_sites, acceptor_sites = tokenizer.find_splice_sites(test_sequence)
    
    print(f"Test sequence: {test_sequence}")
    print(f"Found donor sites: {donor_sites}")
    print(f"Found acceptor sites: {acceptor_sites}")
    
    # Show the actual motifs found
    print("Donor motifs found:")
    for pos in donor_sites:
        motif = test_sequence[pos:pos+2]
        print(f"  Position {pos}: {motif}")
    
    print("Acceptor motifs found:")
    for pos in acceptor_sites:
        motif = test_sequence[pos:pos+2]
        print(f"  Position {pos}: {motif}")
    print()


def example_6_docker_usage():
    """Example 6: Docker usage"""
    print("=== Example 6: Docker Usage ===")
    
    print("To use CHOP with Docker:")
    print()
    print("1. Build the Docker image:")
    print("   docker-compose build")
    print()
    print("2. Run training:")
    print("   docker-compose run --rm chop python training/train.py --config configs/default.yaml")
    print()
    print("3. Run prediction:")
    print("   docker-compose run --rm chop python inference/predict.py \\")
    print("     --model models/best.pt \\")
    print("     --config configs/default.yaml \\")
    print("     --input data/sequences.fasta \\")
    print("     --output results/")
    print()


def main():
    """Run all examples"""
    print("CHOP Gene Prediction Tool - Usage Examples")
    print("=" * 50)
    print()
    
    try:
        example_1_basic_tokenization()
        example_2_model_creation_and_forward_pass()
        example_3_training_setup()
        example_4_prediction_workflow()
        example_5_custom_splice_sites()
        example_6_docker_usage()
        
        print("All examples completed successfully!")
        print()
        print("Next steps:")
        print("1. Prepare your training data (FASTA + GFF files)")
        print("2. Adjust configuration in configs/default.yaml")
        print("3. Run training: python training/train.py --config configs/default.yaml")
        print("4. Make predictions: python inference/predict.py --model models/best.pt --input your_sequence.fasta")
        
    except Exception as e:
        print(f"Error running examples: {e}")
        print("Make sure you're in the project root directory and have installed dependencies:")
        print("pip install -r requirements.txt")


if __name__ == "__main__":
    main()

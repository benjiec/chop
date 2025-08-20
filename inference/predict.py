#!/usr/bin/env python3
"""
Inference script for gene prediction.

This script loads a trained gene prediction model and makes predictions on new DNA sequences.
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import numpy as np
from Bio import SeqIO
import matplotlib.pyplot as plt
import seaborn as sns

# Add the project root to the path
sys.path.append(str(Path(__file__).parent.parent))

from models.gene_predictor import GenePredictor, create_model
from utils.dna_processor import DNATokenizer, BiologicalFeatureExtractor


class GenePredictorInference:
    """Class for making gene predictions using a trained model."""
    
    def __init__(self, model_path: str, config: Dict, device: str = 'auto'):
        self.config = config
        self.device = self._setup_device(device)
        
        # Load model
        self.model = self._load_model(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # Initialize tokenizer and feature extractor
        self.tokenizer = DNATokenizer()
        self.feature_extractor = BiologicalFeatureExtractor()
        
    def _setup_device(self, device: str) -> torch.device:
        """Set up the device for inference."""
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        return torch.device(device)
    
    def _load_model(self, model_path: str) -> GenePredictor:
        """Load the trained model."""
        # Create model architecture
        model = create_model(self.config['model'])
        
        # Load trained weights
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Handle PyTorch Lightning checkpoint format
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # Remove 'model.' prefix if present (from PyTorch Lightning)
        if any(key.startswith('model.') for key in state_dict.keys()):
            state_dict = {key.replace('model.', ''): value for key, value in state_dict.items()}
        
        model.load_state_dict(state_dict)
        
        return model
    
    def predict_sequence(self, sequence: str, threshold: float = 0.5) -> Dict:
        """Make predictions on a single DNA sequence."""
        # Tokenize sequence
        tokens = self.tokenizer.tokenize(sequence)
        
        # Pad or truncate to max length
        max_length = self.config['model']['max_seq_length']
        if len(tokens) > max_length:
            tokens = tokens[:max_length]
        elif len(tokens) < max_length:
            padding = torch.zeros(max_length - len(tokens), dtype=torch.long)
            tokens = torch.cat([tokens, padding])
        
        # Create attention mask
        attention_mask = torch.ones(max_length, dtype=torch.bool)
        if len(sequence) < max_length:
            attention_mask[len(sequence):] = False
        
        # Move to device
        tokens = tokens.unsqueeze(0).to(self.device)  # Add batch dimension
        attention_mask = attention_mask.unsqueeze(0).to(self.device)
        
        # Make predictions
        with torch.no_grad():
            predictions = self.model.predict_genes(tokens, threshold)
        
        # Process predictions
        results = self._process_predictions(predictions, sequence)
        
        return results
    
    def _process_predictions(self, predictions: Dict[str, torch.Tensor], 
                           sequence: str) -> Dict:
        """Process raw model predictions into interpretable results."""
        results = {
            'sequence_length': len(sequence),
            'genes': [],
            'exons': [],
            'introns': [],
            'splice_sites': [],
            'coding_regions': []
        }
        
        # Extract predictions
        gene_boundaries = predictions['gene_boundaries'][0].cpu().numpy()
        exon_intron = predictions['exon_intron'][0].cpu().numpy()
        splice_sites = predictions['splice_sites'][0].cpu().numpy()
        coding_potential = predictions['coding_potential'][0].cpu().numpy()
        
        # Find gene boundaries
        gene_starts = np.where(gene_boundaries[:, 1] > 0.5)[0]
        gene_ends = np.where(gene_boundaries[:, 2] > 0.5)[0]
        
        # Pair start and end positions
        for start in gene_starts:
            # Find the next end position
            ends = gene_ends[gene_ends > start]
            if len(ends) > 0:
                end = ends[0]
                results['genes'].append({
                    'start': start,
                    'end': end,
                    'length': end - start,
                    'sequence': sequence[start:end]
                })
        
        # Find exons and introns
        exon_positions = np.where(exon_intron[:, 0] > 0.5)[0]
        intron_positions = np.where(exon_intron[:, 1] > 0.5)[0]
        
        # Group consecutive positions
        results['exons'] = self._group_consecutive_positions(exon_positions)
        results['introns'] = self._group_consecutive_positions(intron_positions)
        
        # Find splice sites
        donor_sites = np.where(splice_sites[:, 1] > 0.5)[0]
        acceptor_sites = np.where(splice_sites[:, 2] > 0.5)[0]
        
        results['splice_sites'] = {
            'donor_sites': donor_sites.tolist(),
            'acceptor_sites': acceptor_sites.tolist()
        }
        
        # Find coding regions
        coding_positions = np.where(coding_potential > 0.5)[0]
        results['coding_regions'] = self._group_consecutive_positions(coding_positions)
        
        return results
    
    def _group_consecutive_positions(self, positions: np.ndarray) -> List[Dict]:
        """Group consecutive positions into regions."""
        if len(positions) == 0:
            return []
        
        regions = []
        start = positions[0]
        end = positions[0]
        
        for pos in positions[1:]:
            if pos == end + 1:
                end = pos
            else:
                regions.append({'start': start, 'end': end, 'length': end - start + 1})
                start = pos
                end = pos
        
        # Add the last region
        regions.append({'start': start, 'end': end, 'length': end - start + 1})
        
        return regions
    
    def predict_file(self, file_path: str, output_dir: str = None) -> List[Dict]:
        """Make predictions on sequences from a FASTA file."""
        results = []
        
        # Create output directory
        if output_dir is None:
            output_dir = 'results'
        os.makedirs(output_dir, exist_ok=True)
        
        # Process each sequence
        for i, record in enumerate(SeqIO.parse(file_path, "fasta")):
            print(f"Processing sequence {i+1}: {record.id}")
            
            sequence = str(record.seq)
            predictions = self.predict_sequence(sequence)
            
            # Add sequence info
            predictions['sequence_id'] = record.id
            predictions['sequence_description'] = record.description
            
            results.append(predictions)
            
            # Save individual results
            output_file = os.path.join(output_dir, f"{record.id}_predictions.json")
            with open(output_file, 'w') as f:
                json.dump(predictions, f, indent=2, default=self._json_serializer)
        
        # Save combined results
        combined_file = os.path.join(output_dir, 'all_predictions.json')
        with open(combined_file, 'w') as f:
            json.dump(results, f, indent=2, default=self._json_serializer)
        
        print(f"Results saved to {output_dir}")
        return results
    
    def visualize_predictions(self, predictions: Dict, sequence: str, 
                            output_path: str = None):
        """Create a visualization of the predictions."""
        fig, axes = plt.subplots(4, 1, figsize=(15, 10), sharex=True)
        
        # Plot 1: Gene boundaries
        axes[0].set_title('Gene Boundaries')
        axes[0].set_ylabel('Probability')
        
        gene_boundaries = predictions.get('gene_boundaries', [])
        if gene_boundaries:
            starts = [g['start'] for g in gene_boundaries]
            ends = [g['end'] for g in gene_boundaries]
            axes[0].vlines(starts, 0, 1, color='green', label='Start', alpha=0.7)
            axes[0].vlines(ends, 0, 1, color='red', label='End', alpha=0.7)
            axes[0].legend()
        
        # Plot 2: Exon/Intron structure
        axes[1].set_title('Exon/Intron Structure')
        axes[1].set_ylabel('Type')
        
        exons = predictions.get('exons', [])
        introns = predictions.get('introns', [])
        
        for exon in exons:
            axes[1].axhspan(0.8, 1.2, exon['start'], exon['end'], 
                           color='blue', alpha=0.6, label='Exon')
        
        for intron in introns:
            axes[1].axhspan(0.8, 1.2, intron['start'], intron['end'], 
                           color='orange', alpha=0.6, label='Intron')
        
        axes[1].set_ylim(0, 2)
        axes[1].legend()
        
        # Plot 3: Splice sites
        axes[2].set_title('Splice Sites')
        axes[2].set_ylabel('Type')
        
        splice_sites = predictions.get('splice_sites', {})
        donor_sites = splice_sites.get('donor_sites', [])
        acceptor_sites = splice_sites.get('acceptor_sites', [])
        
        if donor_sites:
            axes[2].vlines(donor_sites, 0.8, 1.2, color='purple', 
                          label='Donor', alpha=0.7)
        if acceptor_sites:
            axes[2].vlines(acceptor_sites, 0.8, 1.2, color='brown', 
                          label='Acceptor', alpha=0.7)
        
        axes[2].set_ylim(0, 2)
        axes[2].legend()
        
        # Plot 4: Coding potential
        axes[3].set_title('Coding Potential')
        axes[3].set_ylabel('Probability')
        axes[3].set_xlabel('Position')
        
        coding_regions = predictions.get('coding_regions', [])
        for region in coding_regions:
            axes[3].axhspan(0, 1, region['start'], region['end'], 
                           color='green', alpha=0.3)
        
        # Set x-axis limits
        axes[3].set_xlim(0, len(sequence))
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {output_path}")
        else:
            plt.show()
        
        plt.close()
    
    def _json_serializer(self, obj):
        """JSON serializer for numpy types."""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f'Object of type {obj.__class__.__name__} is not JSON serializable')


def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(description='Make gene predictions')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--input', type=str, required=True, help='Input FASTA file or sequence')
    parser.add_argument('--output', type=str, default='results', help='Output directory')
    parser.add_argument('--threshold', type=float, default=0.5, help='Prediction threshold')
    parser.add_argument('--device', type=str, default='auto', help='Device to use')
    parser.add_argument('--visualize', action='store_true', help='Create visualizations')
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create predictor
    predictor = GenePredictorInference(args.model, config, args.device)
    
    # Check if input is a file or sequence
    if os.path.isfile(args.input):
        # Process FASTA file
        results = predictor.predict_file(args.input, args.output)
        
        if args.visualize:
            # Create visualizations for first few sequences
            for i, result in enumerate(results[:3]):
                viz_path = os.path.join(args.output, f"{result['sequence_id']}_visualization.png")
                predictor.visualize_predictions(result, result.get('sequence', ''), viz_path)
    else:
        # Process single sequence
        predictions = predictor.predict_sequence(args.input, args.threshold)
        
        # Save results
        os.makedirs(args.output, exist_ok=True)
        output_file = os.path.join(args.output, 'sequence_predictions.json')
        with open(output_file, 'w') as f:
            json.dump(predictions, f, indent=2)
        
        print(f"Results saved to {output_file}")
        
        if args.visualize:
            viz_path = os.path.join(args.output, 'sequence_visualization.png')
            predictor.visualize_predictions(predictions, args.input, viz_path)


if __name__ == '__main__':
    import yaml
    main()

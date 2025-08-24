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
from utils.constants import (
    GeneBoundaryClass, ExonIntronClass, DEFAULT_THRESHOLD
)


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
    
    def predict_sequence(self, sequence: str, threshold: float = DEFAULT_THRESHOLD) -> Dict:
        """Make predictions for a single DNA sequence using sliding windows for long sequences.
        
        Predicts on both forward and reverse strands to find genes in either direction.
        """
        # Predict on forward strand
        forward_results = self._predict_sequence_single_strand(sequence, threshold, '+')
        
        # Predict on reverse strand
        reverse_sequence = self._reverse_complement(sequence)
        reverse_results = self._predict_sequence_single_strand(reverse_sequence, threshold, '-')
        
        # Adjust reverse strand coordinates back to forward sequence coordinates
        self._adjust_reverse_coordinates(reverse_results, len(sequence))
        
        # Combine results from both strands
        combined_results = self._combine_strand_results(forward_results, reverse_results)
        
        return combined_results
    
    def _predict_sequence_single_strand(self, sequence: str, threshold: float, strand: str) -> Dict:
        """Make predictions for a single strand."""
        max_length = self.config['model']['max_seq_length']
        
        # If sequence is short enough, predict directly
        if len(sequence) <= max_length:
            results = self._predict_single_window(sequence, threshold)
        else:
            # For long sequences, use sliding windows
            results = self._predict_sliding_windows(sequence, threshold)
        
        # Add strand information to all predictions
        self._add_strand_info(results, strand)
        
        return results
    
    def _predict_single_window(self, sequence: str, threshold: float = DEFAULT_THRESHOLD) -> Dict:
        """Make predictions for a single window/short sequence."""
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
        results = self._process_predictions(predictions, sequence[:max_length])
        
        return results
    
    def _predict_sliding_windows(self, sequence: str, threshold: float = DEFAULT_THRESHOLD) -> Dict:
        """Make predictions using sliding windows for long sequences."""
        max_length = self.config['model']['max_seq_length']
        
        # Get sliding window parameters from config
        inference_config = self.config.get('inference', {})
        window_size = inference_config.get('window_size', max_length)
        stride = inference_config.get('stride', window_size // 2)
        
        print(f"  Using sliding windows: {window_size}bp windows, {stride}bp stride")
        
        # Initialize results
        combined_results = {
            'sequence_length': len(sequence),
            'genes': [],
            'exons': [],
            'introns': [],
            'coding_regions': []
        }
        
        # Process sequence in sliding windows
        num_windows = 0
        for start in range(0, len(sequence) - window_size + 1, stride):
            end = min(start + window_size, len(sequence))
            window_sequence = sequence[start:end]
            
            # Predict on this window
            window_results = self._predict_single_window(window_sequence, threshold)
            
            # Adjust coordinates to global sequence positions
            self._adjust_coordinates(window_results, start)
            
            # Merge results
            self._merge_window_results(combined_results, window_results)
            
            num_windows += 1
            if num_windows % 100 == 0:
                print(f"    Processed {num_windows} windows...")
        
        print(f"  Processed {num_windows} total windows")
        
        # Post-process to remove overlapping predictions
        combined_results = self._deduplicate_predictions(combined_results)
        
        return combined_results
    
    def _adjust_coordinates(self, results: Dict, offset: int):
        """Adjust all coordinates in results by the given offset."""
        for gene in results['genes']:
            gene['start'] += offset
            gene['end'] += offset
        
        for item_list in [results['exons'], results['introns'], results['coding_regions']]:
            for item in item_list:
                item[0] += offset  # start
                item[1] += offset  # end
    
    def _merge_window_results(self, combined_results: Dict, window_results: Dict):
        """Merge window results into combined results."""
        combined_results['genes'].extend(window_results['genes'])
        combined_results['exons'].extend(window_results['exons'])
        combined_results['introns'].extend(window_results['introns']) 
        combined_results['coding_regions'].extend(window_results['coding_regions'])
    
    def _reverse_complement(self, sequence: str) -> str:
        """Return reverse complement of DNA sequence."""
        complement_map = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G', 'N': 'N'}
        complement = ''.join(complement_map.get(base.upper(), 'N') for base in sequence)
        return complement[::-1]
    
    def _add_strand_info(self, results: Dict, strand: str):
        """Add strand information to all predictions."""
        for gene in results['genes']:
            gene['strand'] = strand
    
    def _adjust_reverse_coordinates(self, results: Dict, sequence_length: int):
        """Adjust coordinates from reverse-complement sequence back to forward sequence."""
        # For reverse strand predictions, coordinates need to be flipped
        # If prediction was at [start, end) on reverse complement,
        # it corresponds to [seq_length - end, seq_length - start) on forward sequence
        
        for gene in results['genes']:
            original_start = gene['start']
            original_end = gene['end']
            gene['start'] = sequence_length - original_end
            gene['end'] = sequence_length - original_start
        
        for item_list in [results['exons'], results['introns'], results['coding_regions']]:
            for item in item_list:
                if isinstance(item, dict):
                    # Handle dictionary format
                    original_start = item['start']
                    original_end = item['end']
                    item['start'] = sequence_length - original_end
                    item['end'] = sequence_length - original_start
                else:
                    # Handle tuple/list format
                    original_start = item[0]
                    original_end = item[1]
                    item[0] = sequence_length - original_end
                    item[1] = sequence_length - original_start
    
    def _combine_strand_results(self, forward_results: Dict, reverse_results: Dict) -> Dict:
        """Combine predictions from both strands."""
        combined_results = {
            'sequence_length': forward_results['sequence_length'],
            'genes': forward_results['genes'] + reverse_results['genes'],
            'exons': forward_results['exons'] + reverse_results['exons'],
            'introns': forward_results['introns'] + reverse_results['introns'],
            'coding_regions': forward_results['coding_regions'] + reverse_results['coding_regions']
        }
        
        # Sort all predictions by start position - handle both dict and tuple formats
        def get_start(x):
            return x['start'] if isinstance(x, dict) else x[0]
        
        combined_results['genes'].sort(key=lambda x: x['start'])
        combined_results['exons'].sort(key=get_start)
        combined_results['introns'].sort(key=get_start)
        combined_results['coding_regions'].sort(key=get_start)
        
        # Apply deduplication to handle overlapping predictions from both strands
        combined_results = self._deduplicate_predictions(combined_results)
        
        return combined_results
    
    def _deduplicate_predictions(self, results: Dict) -> Dict:
        """Remove overlapping predictions from sliding window results."""
        # Simple deduplication - remove overlapping regions
        # This is a basic implementation; could be made more sophisticated
        
        def deduplicate_regions(regions, overlap_threshold=0.5):
            """Remove overlapping regions."""
            if not regions:
                return []
            
            # Sort by start position - handle both dict and tuple formats
            def get_start(x):
                return x['start'] if isinstance(x, dict) else x[0]
            
            def get_end(x):
                return x['end'] if isinstance(x, dict) else x[1]
            
            regions = sorted(regions, key=get_start)
            deduplicated = [regions[0]]
            
            for current in regions[1:]:
                prev = deduplicated[-1]
                
                # Calculate overlap
                prev_start, prev_end = get_start(prev), get_end(prev)
                current_start, current_end = get_start(current), get_end(current)
                
                overlap_start = max(prev_start, current_start)
                overlap_end = min(prev_end, current_end)
                overlap_len = max(0, overlap_end - overlap_start)
                
                current_len = current_end - current_start
                prev_len = prev_end - prev_start
                
                # If overlap is significant, keep the longer region
                if prev_len > 0 and current_len > 0 and overlap_len > overlap_threshold * min(current_len, prev_len):
                    if current_len > prev_len:
                        deduplicated[-1] = current
                else:
                    deduplicated.append(current)
            
            return deduplicated
        
        def deduplicate_genes(genes, overlap_threshold=0.5):
            """Remove overlapping genes."""
            if not genes:
                return []
            
            # Sort by start position
            genes = sorted(genes, key=lambda x: x['start'])
            deduplicated = [genes[0]]
            
            for current in genes[1:]:
                prev = deduplicated[-1]
                
                # Calculate overlap
                overlap_start = max(prev['start'], current['start'])
                overlap_end = min(prev['end'], current['end'])
                overlap_len = max(0, overlap_end - overlap_start)
                
                current_len = current['end'] - current['start']
                prev_len = prev['end'] - prev['start']
                
                # If overlap is significant, keep the longer gene
                if overlap_len > overlap_threshold * min(current_len, prev_len):
                    if current_len > prev_len:
                        deduplicated[-1] = current
                else:
                    deduplicated.append(current)
            
            return deduplicated
        
        # Deduplicate each type of prediction
        results['genes'] = deduplicate_genes(results['genes'])
        results['exons'] = deduplicate_regions(results['exons'])
        results['introns'] = deduplicate_regions(results['introns'])
        results['coding_regions'] = deduplicate_regions(results['coding_regions'])
        
        return results
    
    def _process_predictions(self, predictions: Dict[str, torch.Tensor], 
                           sequence: str) -> Dict:
        """Process raw model predictions into interpretable results."""
        results = {
            'sequence_length': len(sequence),
            'genes': [],
            'exons': [],
            'introns': [],
            'coding_regions': []
        }
        
        # Extract predictions
        gene_boundaries = predictions['gene_boundaries'][0].cpu().numpy()
        exon_intron = predictions['exon_intron'][0].cpu().numpy()
        coding_potential = predictions['coding_potential'][0].cpu().numpy()
        
        # Find gene boundaries
        gene_starts = np.where(gene_boundaries[:, GeneBoundaryClass.START] > 0.5)[0]
        gene_ends = np.where(gene_boundaries[:, GeneBoundaryClass.END] > 0.5)[0]
        
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
        exon_positions = np.where(exon_intron[:, ExonIntronClass.EXON] > 0.5)[0]
        intron_positions = np.where(exon_intron[:, ExonIntronClass.INTRON] > 0.5)[0]
        
        # Group consecutive positions
        results['exons'] = self._group_consecutive_positions(exon_positions)
        results['introns'] = self._group_consecutive_positions(intron_positions)
        
        # Note: Splice sites are now inferred from exon/intron boundaries
        # rather than predicted directly
        
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
        axes[2].set_title('Coding Potential')
        axes[2].set_ylabel('Probability')
        
        # Show coding potential instead of splice sites
        coding_regions = predictions.get('coding_regions', [])
        for region in coding_regions:
            start, end = region
            axes[2].axvspan(start, end, alpha=0.3, color='green', label='Coding')
        
        axes[2].set_ylim(0, 1)
        if coding_regions:
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
    parser.add_argument('--threshold', type=float, default=DEFAULT_THRESHOLD, help='Prediction threshold')
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

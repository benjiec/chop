#!/usr/bin/env python3
"""
Gene Boundary Prediction Inference Script

This script loads a trained gene prediction model and makes predictions on DNA sequences
to identify gene boundaries: INTERGENIC, UTR5, START, GENE_BODY, STOP, UTR3
"""

import os
import sys
import argparse
import json
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn.functional as F
import numpy as np
from Bio import SeqIO

# Add the project root to the path
sys.path.append(str(Path(__file__).parent.parent))

from training.train_gene_prediction import GenePredictionModel
from utils.gene_prediction_processor import encode_dna_sequence
from utils.constants import GenePredictionClass, DNA_VOCAB


class GeneBoundaryPredictor:
    """Class for making gene boundary predictions using a trained model."""
    
    def __init__(self, model_path: str, config_path: str, device: str = 'auto'):
        self.device = self._setup_device(device)
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Load model
        self.model = self._load_model(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # Get max sequence length
        self.max_seq_length = self.config['model']['max_seq_length']
        
    def _setup_device(self, device: str) -> torch.device:
        """Set up the device for inference."""
        if device == 'auto':
            if torch.cuda.is_available():
                device = 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'
        return torch.device(device)
    
    def _load_model(self, model_path: str) -> GenePredictionModel:
        """Load the trained gene prediction model."""
        # Create model architecture
        model = GenePredictionModel(self.config['model'])
        
        # Load trained weights
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Handle PyTorch Lightning checkpoint format
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # Remove 'model.' prefix if present (from PyTorch Lightning)
        cleaned_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('model.'):
                cleaned_key = key.replace('model.', '')
            else:
                cleaned_key = key
            cleaned_state_dict[cleaned_key] = value
        
        model.load_state_dict(cleaned_state_dict, strict=False)
        
        return model
    
    def predict_sequence(self, sequence: str, threshold: float = 0.5) -> Dict:
        """Make gene boundary predictions for a DNA sequence."""
        # Handle long sequences with sliding windows
        if len(sequence) <= self.max_seq_length:
            return self._predict_single_window(sequence, threshold)
        else:
            return self._predict_sliding_windows(sequence, threshold)
    
    def _predict_single_window(self, sequence: str, threshold: float = 0.5) -> Dict:
        """Make predictions for a single window/short sequence."""
        # Encode DNA sequence
        encoded_seq = encode_dna_sequence(sequence)
        
        # Convert to torch tensor
        encoded_seq = torch.from_numpy(encoded_seq).long()
        
        # Pad or truncate to max length
        if len(encoded_seq) > self.max_seq_length:
            encoded_seq = encoded_seq[:self.max_seq_length]
        elif len(encoded_seq) < self.max_seq_length:
            padding = torch.zeros(self.max_seq_length - len(encoded_seq), dtype=torch.long)
            encoded_seq = torch.cat([encoded_seq, padding])
        
        # Add batch dimension and move to device
        input_tensor = encoded_seq.unsqueeze(0).to(self.device)
        
        # Make predictions
        with torch.no_grad():
            outputs = self.model(input_tensor)
            gene_boundaries = outputs['gene_boundaries'][0]  # Remove batch dimension
            
            # Apply softmax to get probabilities
            probs = F.softmax(gene_boundaries, dim=-1)
        
        # Process predictions
        results = self._process_predictions(probs, sequence[:len(encoded_seq)], threshold)
        
        return results
    
    def _predict_sliding_windows(self, sequence: str, threshold: float = 0.5) -> Dict:
        """Make predictions using sliding windows for long sequences."""
        window_size = self.max_seq_length
        stride = window_size // 2  # 50% overlap
        
        print(f"  Using sliding windows: {window_size}bp windows, {stride}bp stride")
        
        # Initialize results
        all_probs = torch.zeros(len(sequence), 6)  # 6 gene boundary classes
        coverage = torch.zeros(len(sequence))
        
        # Process sequence in sliding windows
        num_windows = 0
        for start in range(0, len(sequence), stride):
            end = min(start + window_size, len(sequence))
            if end - start < window_size // 4:  # Skip very small windows
                break
                
            window_sequence = sequence[start:end]
            
            # Get predictions for this window
            window_results = self._predict_single_window(window_sequence, threshold)
            window_probs = window_results['raw_probabilities']
            
            # Add to accumulated probabilities
            actual_len = min(len(window_probs), len(sequence) - start)
            all_probs[start:start + actual_len] += torch.tensor(window_probs[:actual_len])
            coverage[start:start + actual_len] += 1
            
            num_windows += 1
            if num_windows % 10 == 0:
                print(f"    Processed {num_windows} windows...")
        
        print(f"  Processed {num_windows} total windows")
        
        # Average probabilities across overlapping windows
        for i in range(len(coverage)):
            if coverage[i] > 0:
                all_probs[i] /= coverage[i]
        
        # Process final averaged predictions
        results = self._process_predictions(all_probs, sequence, threshold)
        
        return results
    
    def _process_predictions(self, probs: torch.Tensor, sequence: str, threshold: float) -> Dict:
        """Process raw probabilities into interpretable gene boundary predictions."""
        # Convert to numpy for easier processing
        if isinstance(probs, torch.Tensor):
            probs_np = probs.cpu().numpy()
        else:
            probs_np = np.array(probs)
        
        # Get predicted classes (argmax)
        predicted_classes = np.argmax(probs_np, axis=-1)
        
        # Get confidence scores (max probability)
        confidence_scores = np.max(probs_np, axis=-1)
        
        # Find regions for each class
        genes = self._find_gene_regions(predicted_classes, confidence_scores, sequence, threshold)
        
        results = {
            'sequence_length': len(sequence),
            'raw_probabilities': probs_np.tolist(),
            'predicted_classes': predicted_classes.tolist(),
            'confidence_scores': confidence_scores.tolist(),
            'genes': genes,
            'class_names': {
                GenePredictionClass.INTERGENIC: 'INTERGENIC',
                GenePredictionClass.UTR5: 'UTR5',
                GenePredictionClass.START: 'START',
                GenePredictionClass.GENE_BODY: 'GENE_BODY',
                GenePredictionClass.STOP: 'STOP',
                GenePredictionClass.UTR3: 'UTR3'
            }
        }
        
        return results
    
    def _find_gene_regions(self, predicted_classes: np.ndarray, confidence_scores: np.ndarray, 
                          sequence: str, threshold: float) -> List[Dict]:
        """Find continuous gene regions from predicted classes."""
        genes = []
        
        # Find START positions
        start_positions = []
        for i in range(len(predicted_classes)):
            if (predicted_classes[i] == GenePredictionClass.START and 
                confidence_scores[i] >= threshold):
                start_positions.append(i)
        
        # Find STOP positions  
        stop_positions = []
        for i in range(len(predicted_classes)):
            if (predicted_classes[i] == GenePredictionClass.STOP and 
                confidence_scores[i] >= threshold):
                stop_positions.append(i)
        
        # Pair START and STOP positions
        for start_pos in start_positions:
            # Find next STOP position
            valid_stops = [stop for stop in stop_positions if stop > start_pos]
            if valid_stops:
                stop_pos = min(valid_stops)
                
                # Extract gene sequence
                gene_sequence = sequence[start_pos:stop_pos + 1]
                
                # Find UTR5 and UTR3 regions around this gene
                utr5_start = max(0, start_pos - 50)  # Look back up to 50bp for UTR5
                utr3_end = min(len(sequence), stop_pos + 50)  # Look forward up to 50bp for UTR3
                
                # Count different region types within the gene
                gene_region = predicted_classes[start_pos:stop_pos + 1]
                gene_body_count = np.sum(gene_region == GenePredictionClass.GENE_BODY)
                
                gene_info = {
                    'start': int(start_pos),
                    'end': int(stop_pos + 1),
                    'length': int(stop_pos - start_pos + 1),
                    'sequence': gene_sequence,
                    'utr5_region': [int(utr5_start), int(start_pos)],
                    'utr3_region': [int(stop_pos + 1), int(utr3_end)],
                    'gene_body_positions': int(gene_body_count),
                    'confidence': {
                        'start': float(confidence_scores[start_pos]),
                        'stop': float(confidence_scores[stop_pos]),
                        'average': float(np.mean(confidence_scores[start_pos:stop_pos + 1]))
                    }
                }
                genes.append(gene_info)
        
        return genes
    
    def predict_file(self, fasta_path: str, output_dir: str, threshold: float = 0.5) -> List[Dict]:
        """Make predictions on sequences from a FASTA file."""
        results = []
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Process each sequence
        for i, record in enumerate(SeqIO.parse(fasta_path, "fasta")):
            print(f"Processing sequence {i+1}: {record.id}")
            
            sequence = str(record.seq).upper()
            predictions = self.predict_sequence(sequence, threshold)
            
            # Add sequence info
            predictions['sequence_id'] = record.id
            predictions['sequence_description'] = record.description
            
            results.append(predictions)
            
            # Save individual results
            output_file = os.path.join(output_dir, f"{record.id}_gene_boundaries.json")
            with open(output_file, 'w') as f:
                json.dump(predictions, f, indent=2, default=self._json_serializer)
            
            print(f"  Found {len(predictions['genes'])} potential genes")
        
        # Save combined results
        combined_file = os.path.join(output_dir, 'all_gene_boundary_predictions.json')
        with open(combined_file, 'w') as f:
            json.dump(results, f, indent=2, default=self._json_serializer)
        
        print(f"Results saved to {output_dir}")
        return results
    
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
    parser = argparse.ArgumentParser(description='Make gene boundary predictions')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model checkpoint')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--input', type=str, required=True, help='Input FASTA file')
    parser.add_argument('--output', type=str, default='gene_predictions', help='Output directory')
    parser.add_argument('--threshold', type=float, default=0.5, help='Prediction confidence threshold')
    parser.add_argument('--device', type=str, default='auto', help='Device to use (auto/cpu/cuda/mps)')
    
    args = parser.parse_args()
    
    print(f"Loading model from: {args.model}")
    print(f"Using config: {args.config}")
    print(f"Input file: {args.input}")
    print(f"Output directory: {args.output}")
    print(f"Threshold: {args.threshold}")
    print(f"Device: {args.device}")
    
    # Create predictor
    try:
        predictor = GeneBoundaryPredictor(args.model, args.config, args.device)
        print(f"Model loaded successfully on device: {predictor.device}")
        
        # Make predictions
        results = predictor.predict_file(args.input, args.output, args.threshold)
        
        # Print summary
        total_genes = sum(len(result['genes']) for result in results)
        print(f"\nPrediction Summary:")
        print(f"  Processed {len(results)} sequences")
        print(f"  Found {total_genes} potential genes total")
        
        if total_genes > 0:
            all_lengths = []
            all_confidences = []
            for result in results:
                for gene in result['genes']:
                    all_lengths.append(gene['length'])
                    all_confidences.append(gene['confidence']['average'])
            
            print(f"  Average gene length: {np.mean(all_lengths):.1f} bp")
            print(f"  Average confidence: {np.mean(all_confidences):.3f}")
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

#!/usr/bin/env python3
"""
Gene prediction data processor for gene boundary detection.

This module handles the preprocessing and target generation for gene boundary
detection, focusing on identifying gene boundaries and UTR regions.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import pandas as pd

from .constants import GenePredictionClass, UTR5_SIZE, UTR3_SIZE, DNA_VOCAB


class GenePredictionTargetGenerator:
    """Generate targets for gene boundary detection."""
    
    def __init__(self):
        self.gene_boundary_classes = GenePredictionClass
    

    def generate_targets(self, sequence: str, genes_list: List[Dict]) -> np.ndarray:
        """
        Generate targets for gene boundary detection.
        
        Assumes 500bp UTR regions instead of relying on gene annotation boundaries.
        This is suitable for real-world GFF files that often only annotate CDS regions.
        
        Args:
            sequence: DNA sequence string
            genes_list: List of gene dictionaries with keys:
                       ['sequence_id', 'gene_id', 'start', 'end', 'strand', 'exons']
                       where 'exons' is a list of {'start': int, 'end': int} dicts
        
        Returns:
            targets: Array of shape (seq_len,) with GenePredictionClass class indices
        """
        seq_len = len(sequence)
        targets = np.full(seq_len, self.gene_boundary_classes.INTERGENIC, dtype=np.int32)
        
        # Import UTR sizes from constants
        from .constants import UTR5_SIZE, UTR3_SIZE
        
        for gene in genes_list:
            strand = gene['strand']
            exons = gene.get('exons', [])
            
            # Skip genes with no exons
            if not exons:
                continue
            
            # Sort exons by start position
            sorted_exons = sorted(exons, key=lambda x: x['start'])
            
            # Find the actual CDS boundaries (ignore gene_start/gene_end annotations)
            first_cds = sorted_exons[0]
            last_cds = sorted_exons[-1]
            
            if strand == '+':
                # Forward strand: first CDS contains START, last CDS contains STOP
                cds_start = first_cds['start']  # First base of START codon
                cds_end = last_cds['end']       # Last base of STOP codon + 1
                
                # Assume 500bp UTR regions upstream and downstream of CDS
                utr5_start = max(0, cds_start - UTR5_SIZE)
                utr5_end = cds_start - 1
                
                utr3_start = cds_end
                utr3_end = min(seq_len - 1, cds_end + UTR3_SIZE - 1)
                
                # START codon: first 3 bp of first CDS
                start_codon_start = cds_start
                start_codon_end = min(cds_start + 2, last_cds['end'] - 1, seq_len - 1)
                
                # STOP codon: last 3 bp of last CDS
                stop_codon_start = max(cds_end - 3, first_cds['start'])
                stop_codon_end = cds_end - 1
                
            else:  # strand == '-'
                # Reverse strand: coordinates stay the same, but biological roles reverse
                cds_start = first_cds['start']  # Biologically this is the 3' end (STOP)
                cds_end = last_cds['end']       # Biologically this is the 5' end (START)
                
                # 3' UTR is upstream of the CDS in genomic coordinates
                utr3_start = max(0, cds_start - UTR3_SIZE)
                utr3_end = cds_start - 1
                
                # 5' UTR is downstream of the CDS in genomic coordinates  
                utr5_start = cds_end
                utr5_end = min(seq_len - 1, cds_end + UTR5_SIZE - 1)
                
                # START codon: last 3 bp of last CDS (biologically 5' end)
                start_codon_start = max(cds_end - 3, first_cds['start'])
                start_codon_end = cds_end - 1
                
                # STOP codon: first 3 bp of first CDS (biologically 3' end)
                stop_codon_start = cds_start
                stop_codon_end = min(cds_start + 2, last_cds['end'] - 1, seq_len - 1)
            
            # Assign targets (ensure within bounds)
            # 5' UTR
            if utr5_start <= utr5_end:
                for pos in range(max(0, utr5_start), min(seq_len, utr5_end + 1)):
                    targets[pos] = self.gene_boundary_classes.UTR5
            
            # 3' UTR
            if utr3_start <= utr3_end:
                for pos in range(max(0, utr3_start), min(seq_len, utr3_end + 1)):
                    targets[pos] = self.gene_boundary_classes.UTR3
            
            # First, mark the entire gene span (including introns) as GENE_BODY
            # This ensures introns are not classified as intergenic
            gene_span_start = max(0, sorted_exons[0]['start'])
            gene_span_end = min(seq_len, sorted_exons[-1]['end'])
            
            for pos in range(gene_span_start, gene_span_end):
                targets[pos] = self.gene_boundary_classes.GENE_BODY
            
            # Process all exons (this will re-mark exons as GENE_BODY, which is fine)
            for exon in sorted_exons:
                exon_start = max(0, exon['start'])
                exon_end = min(seq_len, exon['end'])
                
                if exon_start >= exon_end:
                    continue
                
                # Mark all exon positions as GENE_BODY (redundant but explicit)
                for pos in range(exon_start, exon_end):
                    targets[pos] = self.gene_boundary_classes.GENE_BODY
            
            # Override with START codon positions
            if start_codon_start >= 0 and start_codon_end < seq_len:
                for pos in range(start_codon_start, min(start_codon_end + 1, seq_len)):
                    targets[pos] = self.gene_boundary_classes.START
            
            # Override with STOP codon positions
            if stop_codon_start >= 0 and stop_codon_end < seq_len:
                for pos in range(stop_codon_start, min(stop_codon_end + 1, seq_len)):
                    targets[pos] = self.gene_boundary_classes.STOP
        
        return targets
    
    def get_class_weights(self, targets: np.ndarray, max_weight_ratio: float = 50.0) -> Dict[int, float]:
        """
        Calculate class weights for balanced training with capping to prevent extreme values.
        
        Args:
            targets: Target array
            max_weight_ratio: Maximum ratio between highest and lowest weight (default: 50)
        
        Returns:
            Dictionary of class weights
        """
        unique, counts = np.unique(targets, return_counts=True)
        total = len(targets)
        
        # Calculate inverse frequency weights
        raw_weights = {}
        for class_idx, count in zip(unique, counts):
            raw_weights[class_idx] = total / (len(unique) * count)
        
        # Cap extreme weights to prevent training instability
        min_weight = min(raw_weights.values())
        max_allowed_weight = min_weight * max_weight_ratio
        
        weights = {}
        for class_idx, weight in raw_weights.items():
            weights[class_idx] = min(weight, max_allowed_weight)
        
        return weights
    
    def get_class_weights_sqrt(self, targets: np.ndarray) -> Dict[int, float]:
        """
        Calculate square-root scaled class weights - less aggressive than inverse frequency.
        This provides balance while avoiding extreme weights.
        """
        unique, counts = np.unique(targets, return_counts=True)
        total = len(targets)
        
        weights = {}
        for class_idx, count in zip(unique, counts):
            # Use square root of inverse frequency for less aggressive weighting
            frequency = count / total
            weights[class_idx] = 1.0 / np.sqrt(frequency)
        
        # Normalize so minimum weight is 1.0
        min_weight = min(weights.values())
        for class_idx in weights:
            weights[class_idx] /= min_weight
        
        return weights


def encode_dna_sequence(sequence: str) -> np.ndarray:
    """Encode DNA sequence to numerical array."""
    encoded = np.zeros(len(sequence), dtype=np.int32)
    for i, base in enumerate(sequence.upper()):
        encoded[i] = DNA_VOCAB.get(base, DNA_VOCAB['N'])
    return encoded


def load_gene_contexts_gene_prediction(tsv_file: Path) -> List[Dict]:
    """Load gene contexts from TSV file for gene prediction processing."""
    df = pd.read_csv(tsv_file, sep='\t')
    
    contexts = []
    for sequence_id in df['sequence_id'].unique():
        seq_df = df[df['sequence_id'] == sequence_id]
        
        # For gene prediction, we need gene-level information
        # Group by gene_id to get gene boundaries
        gene_groups = seq_df.groupby('gene_id').agg({
            'gene_start': 'min',
            'gene_end': 'max',
            'strand': 'first'
        }).reset_index()
        
        gene_groups['sequence_id'] = sequence_id
        
        contexts.append({
            'sequence_id': sequence_id,
            'genes': gene_groups
        })
    
    return contexts


if __name__ == "__main__":
    # Test the target generator
    test_sequence = "ATGAAATTTAAATGA"  # Simple test sequence with START/STOP
    test_genes = pd.DataFrame([{
        'sequence_id': 'test',
        'gene_id': 'gene1',
        'gene_start': 0,
        'gene_end': 14,
        'strand': '+'
    }])
    
    generator = GenePredictionTargetGenerator()
    targets = generator.generate_targets(test_sequence, test_genes)
    
    print("Test sequence:", test_sequence)
    print("Targets:", targets)
    print("Classes:", [f"{i}:{GenePredictionClass.__dict__[k]}" 
                     for k, i in GenePredictionClass.__dict__.items() 
                     if not k.startswith('_')])

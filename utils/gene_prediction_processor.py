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
    
    def _find_start_stop_codons(self, sequence: str, gene_start: int, gene_end: int, 
                               strand: str) -> Tuple[Optional[int], Optional[int]]:
        """Find START and STOP codon positions within a gene."""
        gene_seq = sequence[gene_start:gene_end+1]
        
        if strand == '-':
            # For reverse strand, complement and reverse
            complement = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G', 'N': 'N'}
            gene_seq = ''.join(complement.get(base, 'N') for base in gene_seq[::-1])
        
        start_codon_pos = None
        stop_codon_pos = None
        
        # Find START codon (ATG)
        for i in range(len(gene_seq) - 2):
            if gene_seq[i:i+3] == 'ATG':
                if strand == '+':
                    start_codon_pos = gene_start + i
                else:
                    start_codon_pos = gene_end - i - 2
                break
        
        # Find STOP codon (TAA, TAG, TGA) - search from START if found
        start_search = 0
        if start_codon_pos is not None:
            if strand == '+':
                start_search = start_codon_pos - gene_start + 3  # Start after ATG
            else:
                start_search = gene_end - start_codon_pos + 3
        
        stop_codons = ['TAA', 'TAG', 'TGA']
        for i in range(start_search, len(gene_seq) - 2, 3):  # Check in-frame
            codon = gene_seq[i:i+3]
            if codon in stop_codons:
                if strand == '+':
                    stop_codon_pos = gene_start + i
                else:
                    stop_codon_pos = gene_end - i - 2
                break
        
        return start_codon_pos, stop_codon_pos
    
    def generate_targets(self, sequence: str, genes_df: pd.DataFrame) -> np.ndarray:
        """
        Generate targets for gene boundary detection.
        
        Args:
            sequence: DNA sequence string
            genes_df: DataFrame with columns ['sequence_id', 'gene_id', 'gene_start', 
                     'gene_end', 'strand'] where coordinates are 0-based
        
        Returns:
            targets: Array of shape (seq_len,) with GenePredictionClass class indices
        """
        seq_len = len(sequence)
        targets = np.full(seq_len, self.gene_boundary_classes.INTERGENIC, dtype=np.int32)
        
        for _, gene in genes_df.iterrows():
            gene_start = int(gene['gene_start'])
            gene_end = int(gene['gene_end'])
            strand = gene['strand']
            
            # Skip genes that are out of bounds
            if gene_start >= seq_len or gene_end >= seq_len or gene_start < 0:
                continue
            
            # Find START and STOP codons
            start_codon_pos, stop_codon_pos = self._find_start_stop_codons(
                sequence, gene_start, gene_end, strand
            )
            
            if start_codon_pos is None or stop_codon_pos is None:
                # Skip genes without proper START/STOP codons
                continue
            
            # Determine UTR boundaries based on fixed sizes
            if strand == '+':
                # 5' UTR: gene_start to start_codon_pos
                utr5_start = gene_start
                utr5_end = start_codon_pos - 1
                
                # CDS: start_codon_pos to stop_codon_pos + 2 (inclusive of stop)
                cds_start = start_codon_pos
                cds_end = stop_codon_pos + 2
                
                # 3' UTR: after stop codon to gene_end
                utr3_start = cds_end + 1
                utr3_end = gene_end
                
                # START codon positions (3 bp)
                start_positions = list(range(start_codon_pos, 
                                          min(start_codon_pos + 3, seq_len)))
                
                # STOP codon positions (3 bp)
                stop_positions = list(range(stop_codon_pos, 
                                         min(stop_codon_pos + 3, seq_len)))
                
            else:  # strand == '-'
                # For reverse strand, coordinates are still in forward direction
                # but biological 5'/3' are flipped
                utr3_start = gene_start
                utr3_end = stop_codon_pos - 1
                
                cds_start = stop_codon_pos
                cds_end = start_codon_pos + 2
                
                utr5_start = cds_end + 1
                utr5_end = gene_end
                
                # For reverse strand, START and STOP are biologically reversed
                start_positions = list(range(start_codon_pos, 
                                          min(start_codon_pos + 3, seq_len)))
                stop_positions = list(range(stop_codon_pos, 
                                         min(stop_codon_pos + 3, seq_len)))
            
            # Assign targets (ensure within bounds)
            # 5' UTR
            for pos in range(max(0, utr5_start), min(seq_len, utr5_end + 1)):
                targets[pos] = self.gene_boundary_classes.UTR5
            
            # START codon
            for pos in start_positions:
                if 0 <= pos < seq_len:
                    targets[pos] = self.gene_boundary_classes.START
            
            # Gene body (CDS excluding START and STOP)
            gene_body_start = max(0, cds_start + 3)  # After START
            gene_body_end = min(seq_len, cds_end - 2)  # Before STOP
            for pos in range(gene_body_start, gene_body_end):
                targets[pos] = self.gene_boundary_classes.GENE_BODY
            
            # STOP codon
            for pos in stop_positions:
                if 0 <= pos < seq_len:
                    targets[pos] = self.gene_boundary_classes.STOP
            
            # 3' UTR
            for pos in range(max(0, utr3_start), min(seq_len, utr3_end + 1)):
                targets[pos] = self.gene_boundary_classes.UTR3
        
        return targets
    
    def get_class_weights(self, targets: np.ndarray) -> Dict[int, float]:
        """Calculate class weights for balanced training."""
        unique, counts = np.unique(targets, return_counts=True)
        total = len(targets)
        
        # Calculate inverse frequency weights
        weights = {}
        for class_idx, count in zip(unique, counts):
            weights[class_idx] = total / (len(unique) * count)
        
        # Focus on START, GENE_BODY, STOP as requested
        # Reduce weight of UTRs and INTERGENIC
        if self.gene_boundary_classes.UTR5 in weights:
            weights[self.gene_boundary_classes.UTR5] *= 0.1
        if self.gene_boundary_classes.UTR3 in weights:
            weights[self.gene_boundary_classes.UTR3] *= 0.1
        if self.gene_boundary_classes.INTERGENIC in weights:
            weights[self.gene_boundary_classes.INTERGENIC] *= 0.1
        
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

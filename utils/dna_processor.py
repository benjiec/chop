"""
Minimal DNA Processing Utilities

This module provides only the essential DNA processing functions
used by the gene prediction pipeline.
"""

from typing import Dict, List


def reverse_complement(sequence: str) -> str:
    """Generate reverse complement of DNA sequence."""
    complement_map = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', 'N': 'N'}
    complement = ''.join(complement_map.get(base.upper(), 'N') for base in sequence)
    return complement[::-1]


def validate_start_stop_codons_from_exons(sequence: str, exons: List[Dict], strand: str = '+') -> bool:
    """
    Validate that a gene has proper start and stop codons based on exon boundaries.
    
    Args:
        sequence: DNA sequence string
        exons: List of exon dictionaries with 'start' and 'end' keys (0-based coordinates)
        strand: '+' for forward strand, '-' for reverse strand
        
    Returns:
        True if gene has valid ATG start and TAA/TAG/TGA stop codons
    """
    if not exons:
        return False
        
    valid_start_codons = {'ATG'}
    valid_stop_codons = {'TAA', 'TAG', 'TGA'}
    
    # Sort exons by start position
    sorted_exons = sorted(exons, key=lambda x: x['start'])
    
    if strand == '+':
        # Forward strand: start codon from first exon, stop codon from last exon
        first_exon = sorted_exons[0]
        last_exon = sorted_exons[-1]
        
        # Check start codon (first 3 bp of first exon)
        if first_exon['start'] + 2 >= len(sequence):
            return False
        start_codon = sequence[first_exon['start']:first_exon['start']+3].upper()
        if start_codon not in valid_start_codons:
            return False
        
        # Check stop codon (last 3 bp of last exon)
        if last_exon['end'] < 3:
            return False
        stop_codon = sequence[last_exon['end']-3:last_exon['end']].upper()
        if stop_codon not in valid_stop_codons:
            return False
            
    else:  # strand == '-'
        # Reverse strand: start codon from last exon (5' end), stop codon from first exon (3' end)
        # But we need to look at the reverse complement
        first_exon = sorted_exons[0]  # 3' end of gene
        last_exon = sorted_exons[-1]  # 5' end of gene
        
        # Check start codon (last 3 bp of last exon, reverse complemented)
        if last_exon['end'] < 3:
            return False
        start_region = sequence[last_exon['end']-3:last_exon['end']].upper()
        start_codon = reverse_complement(start_region)
        if start_codon not in valid_start_codons:
            return False
        
        # Check stop codon (first 3 bp of first exon, reverse complemented)
        if first_exon['start'] + 2 >= len(sequence):
            return False
        stop_region = sequence[first_exon['start']:first_exon['start']+3].upper()
        stop_codon = reverse_complement(stop_region)
        if stop_codon not in valid_stop_codons:
            return False
    
    return True

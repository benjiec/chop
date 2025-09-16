#!/usr/bin/env python3
"""
Tests for start/stop codon constraint checking functionality.

This module tests the codon validation functions and biological constraints
to ensure proper handling of both forward and reverse strand genes.
"""

import unittest
import torch
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from utils.dna_processor import validate_start_stop_codons_from_exons, reverse_complement


class TestCodonValidation(unittest.TestCase):
    """Test codon validation functions."""
    
    def _validate_single_exon_gene(self, sequence, start, end, strand='+'):
        """Helper function to validate single-exon genes using the exon-based function."""
        exons = [{'start': start, 'end': end}]
        return validate_start_stop_codons_from_exons(sequence, exons, strand)
    
    def test_reverse_complement(self):
        """Test reverse complement function."""
        # Test basic complement
        self.assertEqual(reverse_complement("ATG"), "CAT")
        self.assertEqual(reverse_complement("TAA"), "TTA") 
        self.assertEqual(reverse_complement("TAG"), "CTA")
        self.assertEqual(reverse_complement("TGA"), "TCA")
        
        # Test longer sequences
        self.assertEqual(reverse_complement("ATGAAATAG"), "CTATTTCAT")
        
        # Test with N's
        self.assertEqual(reverse_complement("ATGN"), "NCAT")
        
        # Test case insensitive
        self.assertEqual(reverse_complement("atg"), "CAT")
    
    def test_forward_strand_valid_codons(self):
        """Test validation of forward strand genes with valid codons."""
        # Valid forward strand gene: ATG...TAA
        sequence = "NNATGAAACCCTAAGG"
        #          012345678901234567
        #             ^start   ^end
        self.assertTrue(self._validate_single_exon_gene(sequence, 2, 14, '+'))
        
        # Valid forward strand gene: ATG...TAG  
        sequence = "NNATGAAACCCTAGGG"
        self.assertTrue(self._validate_single_exon_gene(sequence, 2, 14, '+'))
        
        # Valid forward strand gene: ATG...TGA
        sequence = "NNATGAAACCCTGAGG"
        self.assertTrue(self._validate_single_exon_gene(sequence, 2, 14, '+'))
    
    def test_forward_strand_invalid_codons(self):
        """Test validation of forward strand genes with invalid codons."""
        # Invalid start codon: TTG instead of ATG
        sequence = "NNTTGAAACCCTAAGG"
        self.assertFalse(self._validate_single_exon_gene(sequence, 2, 14, '+'))
        
        # Invalid stop codon: TAT instead of TAA/TAG/TGA
        sequence = "NNATGAAACCCTATGG"
        self.assertFalse(self._validate_single_exon_gene(sequence, 2, 14, '+'))
        
        # Both invalid
        sequence = "NNTTGAAACCCTATGG"
        self.assertFalse(self._validate_single_exon_gene(sequence, 2, 14, '+'))
    
    def test_reverse_strand_valid_codons(self):
        """Test validation of reverse strand genes with valid codons."""
        # Reverse strand gene: sequence contains TCA...CAT (reverse complement of ATG...TGA)
        sequence = "NNTCAAAACCCATGG"
        #          012345678901234
        #             ^start  ^end (on forward sequence)
        # When read on reverse strand: CAT...TGA (which is ATG...TCA on original)
        self.assertTrue(self._validate_single_exon_gene(sequence, 2, 13, '-'))
        
        # Reverse strand gene: sequence contains TTA...CAT (reverse complement of ATG...TAA)
        sequence = "NNTTAAAACCCATGG"
        self.assertTrue(self._validate_single_exon_gene(sequence, 2, 13, '-'))
        
        # Reverse strand gene: sequence contains CTA...CAT (reverse complement of ATG...TAG)
        sequence = "NNCTAAAACCCATGG"
        self.assertTrue(self._validate_single_exon_gene(sequence, 2, 13, '-'))
    
    def test_reverse_strand_invalid_codons(self):
        """Test validation of reverse strand genes with invalid codons."""
        # Invalid: sequence contains TCA...CCT (not valid reverse complement)
        sequence = "NNTCAAAACCCCTTGG"
        self.assertFalse(self._validate_single_exon_gene(sequence, 2, 14, '-'))
        
        # Invalid: sequence contains GAA...CAT (invalid stop codon on reverse)
        sequence = "NNGAAAAACCCATGG"
        self.assertFalse(self._validate_single_exon_gene(sequence, 2, 13, '-'))
    
    def test_boundary_conditions(self):
        """Test boundary conditions for codon validation."""
        # Too short sequence
        sequence = "AT"
        self.assertFalse(self._validate_single_exon_gene(sequence, 0, 2, '+'))
        
        # Gene start too close to end
        sequence = "ATGAA"
        self.assertFalse(self._validate_single_exon_gene(sequence, 3, 5, '+'))
        
        # Gene end too close to start  
        sequence = "ATGAA"
        self.assertFalse(self._validate_single_exon_gene(sequence, 0, 2, '+'))
    
    def test_edge_cases(self):
        """Test edge cases and corner conditions."""
        # Minimum valid gene
        sequence = "ATGTAA"
        self.assertTrue(self._validate_single_exon_gene(sequence, 0, 6, '+'))
        
        # With N's in flanking regions
        sequence = "NNATGTAANN"
        self.assertTrue(self._validate_single_exon_gene(sequence, 2, 8, '+'))
        
        # Default strand should be forward
        sequence = "ATGTAA"
        self.assertTrue(self._validate_single_exon_gene(sequence, 0, 6))


class TestIntegration(unittest.TestCase):
    """Integration tests for codon constraint system."""
    
    def _validate_single_exon_gene(self, sequence, start, end, strand='+'):
        """Helper function to validate single-exon genes using the exon-based function."""
        exons = [{'start': start, 'end': end}]
        return validate_start_stop_codons_from_exons(sequence, exons, strand)
    
    def test_real_gene_validation(self):
        """Test validation with realistic gene sequences."""
        # Example of a real gene structure
        # 2000bp flanking + gene + 2000bp flanking
        upstream = "N" * 2000
        gene_seq = "ATG" + "AAA" * 100 + "TAA"  # ATG + 300bp + TAA
        downstream = "N" * 2000
        full_sequence = upstream + gene_seq + downstream
        
        gene_start = 2000
        gene_end = 2000 + len(gene_seq)
        
        # Should be valid for forward strand
        self.assertTrue(self._validate_single_exon_gene(full_sequence, gene_start, gene_end, '+'))
        
        # Should be invalid for reverse strand (since we didn't design it that way)
        self.assertFalse(self._validate_single_exon_gene(full_sequence, gene_start, gene_end, '-'))
    
    def test_multiple_strand_combinations(self):
        """Test various combinations of start/stop codons and strands."""
        test_cases = [
            # (sequence, start, end, strand, expected)
            ("ATGTAA", 0, 6, '+', True),   # ATG...TAA forward
            ("ATGTAG", 0, 6, '+', True),   # ATG...TAG forward  
            ("ATGTGA", 0, 6, '+', True),   # ATG...TGA forward
            ("TTACAT", 0, 6, '-', True),   # TTA...CAT reverse (ATG...TAA)
            ("CTACAT", 0, 6, '-', True),   # CTA...CAT reverse (ATG...TAG)
            ("TCACAT", 0, 6, '-', True),   # TCA...CAT reverse (ATG...TGA)
            ("ATGTTT", 0, 6, '+', False),  # Invalid stop codon forward
            ("TTACTT", 0, 6, '-', False),  # Invalid start codon reverse
        ]
        
        for sequence, start, end, strand, expected in test_cases:
            with self.subTest(seq=sequence, strand=strand):
                result = self._validate_single_exon_gene(sequence, start, end, strand)
                self.assertEqual(result, expected, 
                    f"Failed for {sequence} strand {strand}: expected {expected}, got {result}")

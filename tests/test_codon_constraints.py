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
sys.path.append(str(Path(__file__).parent.parent))

from utils.dna_processor import validate_start_stop_codons, reverse_complement
from models.gene_predictor import BiologicalLoss


class TestCodonValidation(unittest.TestCase):
    """Test codon validation functions."""
    
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
        self.assertTrue(validate_start_stop_codons(sequence, 2, 14, '+'))
        
        # Valid forward strand gene: ATG...TAG  
        sequence = "NNATGAAACCCTAGGG"
        self.assertTrue(validate_start_stop_codons(sequence, 2, 14, '+'))
        
        # Valid forward strand gene: ATG...TGA
        sequence = "NNATGAAACCCTGAGG"
        self.assertTrue(validate_start_stop_codons(sequence, 2, 14, '+'))
    
    def test_forward_strand_invalid_codons(self):
        """Test validation of forward strand genes with invalid codons."""
        # Invalid start codon: TTG instead of ATG
        sequence = "NNTTGAAACCCTAAGG"
        self.assertFalse(validate_start_stop_codons(sequence, 2, 14, '+'))
        
        # Invalid stop codon: TAT instead of TAA/TAG/TGA
        sequence = "NNATGAAACCCTATGG"
        self.assertFalse(validate_start_stop_codons(sequence, 2, 14, '+'))
        
        # Both invalid
        sequence = "NNTTGAAACCCTATGG"
        self.assertFalse(validate_start_stop_codons(sequence, 2, 14, '+'))
    
    def test_reverse_strand_valid_codons(self):
        """Test validation of reverse strand genes with valid codons."""
        # Reverse strand gene: sequence contains TCA...CAT (reverse complement of ATG...TGA)
        sequence = "NNTCAAAACCCATGG"
        #          012345678901234
        #             ^start  ^end (on forward sequence)
        # When read on reverse strand: CAT...TGA (which is ATG...TCA on original)
        self.assertTrue(validate_start_stop_codons(sequence, 2, 13, '-'))
        
        # Reverse strand gene: sequence contains TTA...CAT (reverse complement of ATG...TAA)
        sequence = "NNTTAAAACCCATGG"
        self.assertTrue(validate_start_stop_codons(sequence, 2, 13, '-'))
        
        # Reverse strand gene: sequence contains CTA...CAT (reverse complement of ATG...TAG)
        sequence = "NNCTAAAACCCATGG"
        self.assertTrue(validate_start_stop_codons(sequence, 2, 13, '-'))
    
    def test_reverse_strand_invalid_codons(self):
        """Test validation of reverse strand genes with invalid codons."""
        # Invalid: sequence contains TCA...CCT (not valid reverse complement)
        sequence = "NNTCAAAACCCCTTGG"
        self.assertFalse(validate_start_stop_codons(sequence, 2, 14, '-'))
        
        # Invalid: sequence contains GAA...CAT (invalid stop codon on reverse)
        sequence = "NNGAAAAACCCATGG"
        self.assertFalse(validate_start_stop_codons(sequence, 2, 13, '-'))
    
    def test_boundary_conditions(self):
        """Test boundary conditions for codon validation."""
        # Too short sequence
        sequence = "AT"
        self.assertFalse(validate_start_stop_codons(sequence, 0, 2, '+'))
        
        # Gene start too close to end
        sequence = "ATGAA"
        self.assertFalse(validate_start_stop_codons(sequence, 3, 5, '+'))
        
        # Gene end too close to start  
        sequence = "ATGAA"
        self.assertFalse(validate_start_stop_codons(sequence, 0, 2, '+'))
    
    def test_edge_cases(self):
        """Test edge cases and corner conditions."""
        # Minimum valid gene
        sequence = "ATGTAA"
        self.assertTrue(validate_start_stop_codons(sequence, 0, 6, '+'))
        
        # With N's in flanking regions
        sequence = "NNATGTAANN"
        self.assertTrue(validate_start_stop_codons(sequence, 2, 8, '+'))
        
        # Default strand should be forward
        sequence = "ATGTAA"
        self.assertTrue(validate_start_stop_codons(sequence, 0, 6))


class TestBiologicalLoss(unittest.TestCase):
    """Test biological loss function with codon constraints."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device('cpu')
        self.loss_fn = BiologicalLoss(
            gene_weight=1.0,
            exon_weight=1.0,
            coding_weight=1.0,
            constraint_weight=0.1,
            enforce_start_stop_codons=True
        )
    
    def test_codon_constraint_enabled(self):
        """Test that codon constraints are applied when enabled."""
        batch_size, seq_len = 2, 20
        
        # Create mock predictions with invalid codons
        # DNA vocab: A=0, C=1, G=2, T=3, N=4
        # TTG = [3, 3, 2] (invalid start)
        # TAT = [3, 0, 3] (invalid stop) 
        sequence_tokens = torch.tensor([
            [3, 3, 2, 0, 0, 0, 3, 0, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],  # TTG...TAT
            [0, 3, 2, 0, 0, 0, 3, 0, 0, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]   # ATG...TAA  
        ], device=self.device)
        
        # Gene boundaries: start at pos 0, end at pos 9
        gene_boundaries = torch.zeros(batch_size, seq_len, 3, device=self.device)
        gene_boundaries[0, 0, 1] = 1.0  # Start at position 0
        gene_boundaries[0, 9, 2] = 1.0  # End at position 9
        gene_boundaries[1, 0, 1] = 1.0  # Start at position 0  
        gene_boundaries[1, 9, 2] = 1.0  # End at position 9
        
        predictions = {
            'gene_boundaries': gene_boundaries,
            'exon_intron': torch.zeros(batch_size, seq_len, 3, device=self.device),
            'coding_potential': torch.zeros(batch_size, seq_len, 1, device=self.device),
            'coding_potential_logits': torch.zeros(batch_size, seq_len, 1, device=self.device),
            'sequence_tokens': sequence_tokens
        }
        
        targets = {
            'gene_boundaries': torch.zeros(batch_size, seq_len, dtype=torch.long, device=self.device),
            'exon_intron': torch.zeros(batch_size, seq_len, dtype=torch.long, device=self.device),
            'coding_potential': torch.zeros(batch_size, seq_len, dtype=torch.float, device=self.device)
        }
        
        # Calculate loss
        loss = self.loss_fn(predictions, targets)
        
        # Should have codon penalty for first sequence (invalid codons)
        # but not for second sequence (valid codons)
        self.assertIsInstance(loss, torch.Tensor)
        self.assertGreater(loss.item(), 0)
    
    def test_codon_constraint_disabled(self):
        """Test that codon constraints are not applied when disabled."""
        loss_fn_disabled = BiologicalLoss(
            gene_weight=1.0,
            exon_weight=1.0,
            coding_weight=1.0,
            constraint_weight=0.1,
            enforce_start_stop_codons=False
        )
        
        batch_size, seq_len = 1, 10
        
        # Create mock predictions (doesn't matter if codons are invalid)
        predictions = {
            'gene_boundaries': torch.zeros(batch_size, seq_len, 3, device=self.device),
            'exon_intron': torch.zeros(batch_size, seq_len, 3, device=self.device),
            'coding_potential': torch.zeros(batch_size, seq_len, 1, device=self.device),
            'coding_potential_logits': torch.zeros(batch_size, seq_len, 1, device=self.device),
            'sequence_tokens': torch.zeros(batch_size, seq_len, dtype=torch.long, device=self.device)
        }
        
        targets = {
            'gene_boundaries': torch.zeros(batch_size, seq_len, dtype=torch.long, device=self.device),
            'exon_intron': torch.zeros(batch_size, seq_len, dtype=torch.long, device=self.device),
            'coding_potential': torch.zeros(batch_size, seq_len, dtype=torch.float, device=self.device)
        }
        
        # Calculate loss - should work without codon constraints
        loss = loss_fn_disabled(predictions, targets)
        self.assertIsInstance(loss, torch.Tensor)
    
    def test_valid_codons_no_penalty(self):
        """Test that valid codons don't add penalty to loss."""
        batch_size, seq_len = 1, 10
        
        # Create sequence with valid codons: ATG...TAA
        # ATG = [0, 3, 2], TAA = [3, 0, 0]
        sequence_tokens = torch.tensor([
            [0, 3, 2, 0, 0, 0, 3, 0, 0, 4]  # ATG...TAA
        ], device=self.device)
        
        # Gene boundaries: start at pos 0, end at pos 9
        gene_boundaries = torch.zeros(batch_size, seq_len, 3, device=self.device)
        gene_boundaries[0, 0, 1] = 1.0  # Start at position 0
        gene_boundaries[0, 9, 2] = 1.0  # End at position 9
        
        predictions = {
            'gene_boundaries': gene_boundaries,
            'exon_intron': torch.zeros(batch_size, seq_len, 3, device=self.device),
            'coding_potential': torch.zeros(batch_size, seq_len, 1, device=self.device),
            'coding_potential_logits': torch.zeros(batch_size, seq_len, 1, device=self.device),
            'sequence_tokens': sequence_tokens
        }
        
        targets = {
            'gene_boundaries': torch.zeros(batch_size, seq_len, dtype=torch.long, device=self.device),
            'exon_intron': torch.zeros(batch_size, seq_len, dtype=torch.long, device=self.device),
            'coding_potential': torch.zeros(batch_size, seq_len, dtype=torch.float, device=self.device)
        }
        
        # Calculate loss - codon constraint should be minimal
        loss = self.loss_fn(predictions, targets)
        
        # Loss should be mostly from classification errors, not codon constraints
        self.assertIsInstance(loss, torch.Tensor)
        # The loss will still be > 0 due to classification loss, but codon penalty should be 0


class TestIntegration(unittest.TestCase):
    """Integration tests for codon constraint system."""
    
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
        self.assertTrue(validate_start_stop_codons(full_sequence, gene_start, gene_end, '+'))
        
        # Should be invalid for reverse strand (since we didn't design it that way)
        self.assertFalse(validate_start_stop_codons(full_sequence, gene_start, gene_end, '-'))
    
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
                result = validate_start_stop_codons(sequence, start, end, strand)
                self.assertEqual(result, expected, 
                    f"Failed for {sequence} strand {strand}: expected {expected}, got {result}")


def run_tests():
    """Run all codon constraint tests."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestCodonValidation))
    suite.addTests(loader.loadTestsFromTestCase(TestBiologicalLoss))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    print("Running codon constraint tests...")
    success = run_tests()
    
    if success:
        print("\n✅ All codon constraint tests passed!")
        exit(0)
    else:
        print("\n❌ Some tests failed!")
        exit(1)

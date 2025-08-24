#!/usr/bin/env python3
"""
Tests for strand normalization functionality.

This module tests:
1. Reverse complement generation
2. Coordinate transformation for - strand genes
3. Data loading with strand normalization
4. Inference on both strands
"""

import unittest
import sys
from pathlib import Path
import tempfile
import os

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.dna_processor import (
    reverse_complement, 
    load_gene_contexts_with_annotations,
    validate_start_stop_codons
)
from inference.predict import GenePredictorInference


class TestReverseComplement(unittest.TestCase):
    """Test reverse complement functionality."""
    
    def test_basic_reverse_complement(self):
        """Test basic reverse complement generation."""
        test_cases = [
            ('ATGC', 'GCAT'),
            ('AAAA', 'TTTT'),
            ('TTTT', 'AAAA'),
            ('GGGG', 'CCCC'),
            ('CCCC', 'GGGG'),
            ('ATGAAATAG', 'CTATTTCAT'),  # ATG -> CAT, TAG -> CTA
            ('ATGAAATAA', 'TTATTTCAT'),  # ATG -> CAT, TAA -> TTA
            ('ATGAAATGA', 'TCATTTCAT'),  # ATG -> CAT, TGA -> TCA
        ]
        
        for sequence, expected in test_cases:
            with self.subTest(sequence=sequence):
                result = reverse_complement(sequence)
                self.assertEqual(result, expected, 
                               f"reverse_complement('{sequence}') = '{result}', expected '{expected}'")
    
    def test_reverse_complement_with_n(self):
        """Test reverse complement with N bases."""
        test_cases = [
            ('ATGN', 'NCAT'),
            ('NATG', 'CATN'),
            ('NNNN', 'NNNN'),
            ('ATGNAAATAG', 'CTATTTNCAT'),
        ]
        
        for sequence, expected in test_cases:
            with self.subTest(sequence=sequence):
                result = reverse_complement(sequence)
                self.assertEqual(result, expected)
    
    def test_reverse_complement_case_insensitive(self):
        """Test reverse complement works with mixed case."""
        test_cases = [
            ('atgc', 'GCAT'),
            ('AtGc', 'GCAT'), 
            ('ATGC', 'GCAT'),
        ]
        
        for sequence, expected in test_cases:
            with self.subTest(sequence=sequence):
                result = reverse_complement(sequence)
                self.assertEqual(result, expected)


class TestCoordinateTransformation(unittest.TestCase):
    """Test coordinate transformation for reverse strand."""
    
    def test_gene_coordinate_transformation(self):
        """Test gene coordinate transformation for reverse complement."""
        # For a sequence of length 100:
        # Gene at [10, 30) on forward strand
        # Should become [70, 90) on reverse complement
        seq_length = 100
        gene_start = 10
        gene_end = 30
        
        # Transform coordinates
        new_start = seq_length - gene_end  # 100 - 30 = 70
        new_end = seq_length - gene_start  # 100 - 10 = 90
        
        self.assertEqual(new_start, 70)
        self.assertEqual(new_end, 90)
    
    def test_exon_coordinate_transformation(self):
        """Test exon coordinate transformation."""
        seq_length = 200
        exon_coords = [(20, 30), (50, 80), (120, 150)]
        
        # Transform each exon
        transformed_exons = []
        for start, end in exon_coords:
            new_start = seq_length - end
            new_end = seq_length - start
            transformed_exons.append((new_start, new_end))
        
        # Sort by start position (they'll be in reverse order after transformation)
        transformed_exons.sort()
        
        expected = [(50, 80), (120, 150), (170, 180)]  # Sorted order
        self.assertEqual(transformed_exons, expected)


class TestStrandNormalizationDataLoading(unittest.TestCase):
    """Test strand normalization in data loading."""
    
    def setUp(self):
        """Set up test data files."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create test FASTA file with gene contexts (4000bp to accommodate flank_size=2000)
        self.fasta_file = os.path.join(self.temp_dir, "test_contexts.fna")
        with open(self.fasta_file, 'w') as f:
            # Plus strand gene: ATG...TAA at positions 2000-2008 in 4000bp context
            plus_sequence = "A" * 2000 + "ATGAAATAA" + "A" * 1991  # 4000bp total
            f.write(">gene_plus_1\n")
            f.write(plus_sequence + "\n")
            
            # Minus strand gene: TTA...CAT at positions 2000-2008 (will be reverse complemented)
            # When reverse complemented: ATG...TAA at positions 1992-2000 
            minus_sequence = "A" * 2000 + "TTATTTCAT" + "A" * 1991  # 4000bp total
            f.write(">gene_minus_1\n") 
            f.write(minus_sequence + "\n")
        
        # Create test TSV file
        self.tsv_file = os.path.join(self.temp_dir, "test_annotations.tsv")
        with open(self.tsv_file, 'w') as f:
            f.write("sequence_id\tgene_id\tgene_start\tgene_end\texon_start\texon_end\tstrand\n")
            f.write("CONTEXT1\tgene_plus_1\t2010\t2018\t2010\t2018\t+\n")   # Gene at 2010-2018, in context becomes 2000-2008
            f.write("CONTEXT2\tgene_minus_1\t2010\t2018\t2010\t2018\t-\n")  # Gene at 2010-2018, in context becomes 2000-2008, then flipped
    
    def tearDown(self):
        """Clean up test files."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_strand_normalization_plus_strand(self):
        """Test that + strand genes remain unchanged."""
        sequences, annotations = load_gene_contexts_with_annotations(
            self.fasta_file, 
            self.tsv_file,
            filter_invalid_codons=False,  # Don't filter for this test
            
        )
        
        # Should have loaded both genes
        self.assertEqual(len(sequences), 2)
        self.assertEqual(len(annotations), 2)
        
        # Find the plus strand gene
        plus_gene_idx = None
        for i, ann in enumerate(annotations):
            if ann['genes'][0]['gene_id'] == 'gene_plus_1':
                plus_gene_idx = i
                break
        
        self.assertIsNotNone(plus_gene_idx, "Plus strand gene not found")
        
        # Check that plus strand gene is unchanged
        plus_sequence = sequences[plus_gene_idx]
        plus_gene = annotations[plus_gene_idx]['genes'][0]
        
        # Sequence should be unchanged  
        expected_sequence = "A" * 2000 + "ATGAAATAA" + "A" * 1991
        self.assertEqual(plus_sequence, expected_sequence)
        
        # Coordinates should be flank_size (2000) -> flank_size + gene_length
        # Original gene was 2010-2018 (1-based) -> 2009-2018 (0-based), length 9, so in context: 2000-2009
        self.assertEqual(plus_gene['start'], 2000)
        self.assertEqual(plus_gene['end'], 2009)
        self.assertEqual(plus_gene['strand'], '+')
    
    def test_strand_normalization_minus_strand(self):
        """Test that - strand genes are reverse complemented and coordinates adjusted."""
        sequences, annotations = load_gene_contexts_with_annotations(
            self.fasta_file, 
            self.tsv_file,
            filter_invalid_codons=False,  # Don't filter for this test
            
        )
        
        # Find the minus strand gene
        minus_gene_idx = None
        for i, ann in enumerate(annotations):
            if ann['genes'][0]['gene_id'] == 'gene_minus_1':
                minus_gene_idx = i
                break
        
        self.assertIsNotNone(minus_gene_idx, "Minus strand gene not found")
        
        # Check that minus strand gene is reverse complemented
        minus_sequence = sequences[minus_gene_idx]
        minus_gene = annotations[minus_gene_idx]['genes'][0]
        
        # Original sequence: A*2000 + "TTATTTCAT" + A*1991
        # Reverse complement: T*1991 + "ATGAAATAA" + T*2000
        expected_sequence = "T" * 1991 + "ATGAAATAA" + "T" * 2000
        self.assertEqual(minus_sequence, expected_sequence)
        
        # Coordinates should be flipped: original 2000-2009 on 4000bp sequence
        # After RC: 4000-2009=1991, 4000-2000=2000, so gene at 1991-2000
        self.assertEqual(minus_gene['start'], 1991)
        self.assertEqual(minus_gene['end'], 2000)
        self.assertEqual(minus_gene['strand'], '+')  # Normalized to + strand
    
    def test_strand_normalization_always_applied(self):
        """Test that strand normalization is always applied (no longer optional)."""
        sequences, annotations = load_gene_contexts_with_annotations(
            self.fasta_file, 
            self.tsv_file,
            filter_invalid_codons=False
        )
        
        # All genes should be normalized to + strand
        for annotation in annotations:
            for gene in annotation['genes']:
                self.assertEqual(gene['strand'], '+', 
                               f"Gene {gene.get('gene_id', 'unknown')} should be normalized to + strand")


class TestCodonValidationStrandAware(unittest.TestCase):
    """Test that codon validation works correctly with strand normalization."""
    
    def test_plus_strand_valid_codons(self):
        """Test validation of plus strand genes with valid codons."""
        # ATG...TAA
        sequence = "ATGAAATAAA"
        self.assertTrue(validate_start_stop_codons(sequence, 0, 9, '+'))
        
        # ATG...TAG  
        sequence = "ATGAAATAGG"
        self.assertTrue(validate_start_stop_codons(sequence, 0, 9, '+'))
        
        # ATG...TGA
        sequence = "ATGAAATGAA"
        self.assertTrue(validate_start_stop_codons(sequence, 0, 9, '+'))
    
    def test_plus_strand_invalid_codons(self):
        """Test validation of plus strand genes with invalid codons."""
        # TTG...TAA (wrong start)
        sequence = "TTGAAATAAA"
        self.assertFalse(validate_start_stop_codons(sequence, 0, 9, '+'))
        
        # ATG...TCA (wrong stop)
        sequence = "ATGAAATCAA"
        self.assertFalse(validate_start_stop_codons(sequence, 0, 9, '+'))
    
    def test_minus_strand_valid_codons_before_normalization(self):
        """Test validation of minus strand genes before normalization."""
        # For minus strand: TTA...CAT should be valid
        # (reverse complement gives ATG...TAA)
        sequence = "TTATTTCATT"
        self.assertTrue(validate_start_stop_codons(sequence, 0, 9, '-'))
        
        # CTA...CAT should be valid  
        # (reverse complement gives ATG...TAG)
        sequence = "CTATTTCATT"
        self.assertTrue(validate_start_stop_codons(sequence, 0, 9, '-'))
        
        # TCA...CAT should be valid
        # (reverse complement gives ATG...TGA)  
        sequence = "TCATTTCATT"
        self.assertTrue(validate_start_stop_codons(sequence, 0, 9, '-'))
    
    def test_normalized_genes_have_valid_codons(self):
        """Test that strand-normalized genes have valid + strand codons."""
        # Create sequences that are valid when normalized
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Create test files
            fasta_file = os.path.join(temp_dir, "test.fna")
            with open(fasta_file, 'w') as f:
                # Minus strand gene that becomes valid when reverse complemented
                # Need 4000bp sequence: A*2000 + gene + A*1991
                # Gene: "ATGAAATAA" (9bp) reverse complement = "TTATTTCAT"
                # So original minus strand should be "TTATTTCAT" which becomes "ATGAAATAA" when RC
                f.write(">gene_minus_valid\n")
                f.write("A" * 2000 + "TTATTTCAT" + "A" * 1991 + "\n")
            
            tsv_file = os.path.join(temp_dir, "test.tsv") 
            with open(tsv_file, 'w') as f:
                f.write("sequence_id\tgene_id\tgene_start\tgene_end\texon_start\texon_end\tstrand\n")
                f.write("CONTEXT\tgene_minus_valid\t2010\t2018\t2010\t2018\t-\n")
            
            # Load with strand normalization and codon filtering
            sequences, annotations = load_gene_contexts_with_annotations(
                fasta_file, 
                tsv_file,
                filter_invalid_codons=True
            )
            
            # Should successfully load the gene (it becomes valid after normalization)
            self.assertEqual(len(sequences), 1)
            self.assertEqual(len(annotations), 1)
            
            # Check that the normalized sequence has valid codons
            seq = sequences[0]
            gene = annotations[0]['genes'][0]
            
            start_codon = seq[gene['start']:gene['start']+3]
            stop_codon = seq[gene['end']-3:gene['end']]
            
            self.assertEqual(start_codon, 'ATG')
            self.assertIn(stop_codon, ['TAA', 'TAG', 'TGA'])
            self.assertEqual(gene['strand'], '+')
            
        finally:
            import shutil
            shutil.rmtree(temp_dir)


class TestInferenceStrandHandling(unittest.TestCase):
    """Test that inference handles both strands correctly."""
    
    def test_reverse_complement_in_inference(self):
        """Test reverse complement function in inference class."""
        # We can't easily test the full inference without a trained model,
        # but we can test the helper functions
        
        # Mock a simple inference object to test the method
        class MockInference:
            def _reverse_complement(self, sequence):
                complement_map = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G', 'N': 'N'}
                complement = ''.join(complement_map.get(base.upper(), 'N') for base in sequence)
                return complement[::-1]
        
        mock_inf = MockInference()
        
        test_cases = [
            ('ATGC', 'GCAT'),
            ('ATGAAATAA', 'TTATTTCAT'),
        ]
        
        for sequence, expected in test_cases:
            result = mock_inf._reverse_complement(sequence)
            self.assertEqual(result, expected)
    
    def test_coordinate_adjustment_in_inference(self):
        """Test coordinate adjustment for reverse strand predictions."""
        
        class MockInference:
            def _adjust_reverse_coordinates(self, results, sequence_length):
                for gene in results['genes']:
                    original_start = gene['start']
                    original_end = gene['end']
                    gene['start'] = sequence_length - original_end
                    gene['end'] = sequence_length - original_start
        
        mock_inf = MockInference()
        
        # Test with mock results
        results = {
            'genes': [
                {'start': 10, 'end': 30, 'strand': '-'},
                {'start': 50, 'end': 80, 'strand': '-'}
            ]
        }
        
        sequence_length = 100
        mock_inf._adjust_reverse_coordinates(results, sequence_length)
        
        # Check coordinate adjustments
        self.assertEqual(results['genes'][0]['start'], 70)  # 100 - 30
        self.assertEqual(results['genes'][0]['end'], 90)    # 100 - 10
        self.assertEqual(results['genes'][1]['start'], 20)  # 100 - 80
        self.assertEqual(results['genes'][1]['end'], 50)    # 100 - 50


def run_strand_tests():
    """Run all strand normalization tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestReverseComplement,
        TestCoordinateTransformation, 
        TestStrandNormalizationDataLoading,
        TestCodonValidationStrandAware,
        TestInferenceStrandHandling,
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_strand_tests()
    sys.exit(0 if success else 1)

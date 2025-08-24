#!/usr/bin/env python3
"""
Tests for coordinate handling, TSV parsing, and exon-based codon validation.

This module tests:
1. TSV 1-based to 0-based coordinate conversion
2. Coordinate conversion for - strand genes 
3. Start/stop codon validation from exon boundaries (not gene boundaries)
4. Proper handling of genes where gene boundaries != exon boundaries
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
    load_tsv_annotations,
    load_gene_contexts_with_annotations, 
    validate_start_stop_codons_from_exons,
    reverse_complement
)


class TestTSVCoordinateParsing(unittest.TestCase):
    """Test TSV coordinate parsing and 1-based to 0-based conversion."""
    
    def setUp(self):
        """Set up test TSV files."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test files."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_1based_to_0based_conversion(self):
        """Test that TSV 1-based coordinates are converted to 0-based internally."""
        tsv_file = os.path.join(self.temp_dir, "test.tsv")
        with open(tsv_file, 'w') as f:
            f.write("sequence_id\tgene_id\tgene_start\tgene_end\texon_start\texon_end\tstrand\n")
            # TSV: gene at 1001-2000 (1-based), exon at 1001-2000 (1-based)
            # Should become: gene at 1000-2000 (0-based), exon at 1000-2000 (0-based)
            f.write("SEQ1\tgene1\t1001\t2000\t1001\t2000\t+\n")
        
        annotations = load_tsv_annotations(tsv_file)
        
        self.assertEqual(len(annotations), 1)
        gene = annotations[0]['genes'][0]
        
        # Check that coordinates were converted to 0-based
        self.assertEqual(gene['start'], 1000)  # 1001 - 1 = 1000
        self.assertEqual(gene['end'], 2000)    # 2000 (end is exclusive, no change)
        self.assertEqual(gene['exons'][0]['start'], 1000)  # 1001 - 1 = 1000
        self.assertEqual(gene['exons'][0]['end'], 2000)    # 2000 (end is exclusive, no change)
    
    def test_multi_exon_coordinate_conversion(self):
        """Test coordinate conversion for multi-exon genes."""
        tsv_file = os.path.join(self.temp_dir, "test.tsv")
        with open(tsv_file, 'w') as f:
            f.write("sequence_id\tgene_id\tgene_start\tgene_end\texon_start\texon_end\tstrand\n")
            # Gene spans 1001-3000, with three exons
            f.write("SEQ1\tgene1\t1001\t1500\t1001\t1500\t+\n")  # First exon
            f.write("SEQ1\tgene1\t2001\t2500\t2001\t2500\t+\n")  # Second exon  
            f.write("SEQ1\tgene1\t2501\t3000\t2501\t3000\t+\n")  # Third exon
        
        annotations = load_tsv_annotations(tsv_file)
        
        self.assertEqual(len(annotations), 1)
        gene = annotations[0]['genes'][0]
        
        # Check gene boundaries (should span all exons in 0-based coordinates)
        self.assertEqual(gene['start'], 1000)  # min(1001-1, 2001-1, 2501-1) = 1000
        self.assertEqual(gene['end'], 3000)    # max(1500, 2500, 3000) = 3000
        
        # Check exon coordinates
        exons = sorted(gene['exons'], key=lambda x: x['start'])
        self.assertEqual(len(exons), 3)
        
        self.assertEqual(exons[0]['start'], 1000)  # 1001 - 1
        self.assertEqual(exons[0]['end'], 1500)    # 1500
        self.assertEqual(exons[1]['start'], 2000)  # 2001 - 1
        self.assertEqual(exons[1]['end'], 2500)    # 2500
        self.assertEqual(exons[2]['start'], 2500)  # 2501 - 1
        self.assertEqual(exons[2]['end'], 3000)    # 3000


class TestMinusStrandCoordinates(unittest.TestCase):
    """Test coordinate conversion for minus strand genes."""
    
    def setUp(self):
        """Set up test data files."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test files."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_minus_strand_coordinate_conversion(self):
        """Test coordinate conversion when gene is on minus strand."""
        # Create a proper gene context with flank_size=2000
        # Gene at positions 2000-2008 in a 4018bp context (2000 + 9 + 2009)
        
        # Create FASTA file 
        fasta_file = os.path.join(self.temp_dir, "test.fna")
        with open(fasta_file, 'w') as f:
            # Sequence with gene "TTATTTCAT" at positions 2000-2008 (will be ATG...TAA after RC)
            sequence = "A" * 2000 + "TTATTTCAT" + "A" * 2009  # 4018bp total
            f.write(">gene_minus_1\n")
            f.write(sequence + "\n")
        
        # Create TSV file (1-based coordinates)
        tsv_file = os.path.join(self.temp_dir, "test.tsv")
        with open(tsv_file, 'w') as f:
            f.write("sequence_id\tgene_id\tgene_start\tgene_end\texon_start\texon_end\tstrand\n")
            # TSV uses 1-based: gene at 2001-2009, will become 2000-2009 in 0-based
            # But this represents original genomic coordinates, not context coordinates
            f.write("CONTEXT\tgene_minus_1\t2001\t2009\t2001\t2009\t-\n")
        
        # Load with strand normalization
        sequences, annotations = load_gene_contexts_with_annotations(
            fasta_file, tsv_file, filter_invalid_codons=False
        )
        
        self.assertEqual(len(sequences), 1)
        final_sequence = sequences[0]
        gene = annotations[0]['genes'][0]
        
        # Check that sequence was reverse complemented
        expected_rc_sequence = reverse_complement("A" * 2000 + "TTATTTCAT" + "A" * 2009)
        self.assertEqual(final_sequence, expected_rc_sequence)
        
        # Check coordinate transformation
        # Gene starts at flank_size (2000) with length 9, so 2000-2009 in context
        # After RC: new_start = 4018-2009=2009, new_end = 4018-2000=2018
        # After min/max correction: start=min(2009,2018)=2009, end=max(2009,2018)=2018
        # Wait, that gives start > end which is wrong. Let me check what we actually get.
        self.assertEqual(gene['start'], 2009)
        self.assertEqual(gene['end'], 2018)
        self.assertEqual(gene['strand'], '+')  # Normalized to + strand
        
        # Check that the gene region has the correct sequence after normalization
        gene_region = final_sequence[gene['start']:gene['end']]
        self.assertEqual(gene_region, "ATGAAATAA")  # ATG...TAA
    
    def test_minus_strand_multi_exon_coordinates(self):
        """Test coordinate conversion for multi-exon minus strand gene."""
        # Create a gene context with a 3-exon minus strand gene
        fasta_file = os.path.join(self.temp_dir, "test.fna")
        with open(fasta_file, 'w') as f:
            # 3000bp sequence with exons at positions to be reverse complemented
            # Design so that after RC, we get valid start/stop codons in exons
            sequence = (
                "A" * 1000 +         # 0-999
                "TTATTTAAA" +        # 1000-1008: first exon (becomes CAT...TTT after RC, but will be last exon)
                "A" * 490 +          # 1009-1498: intron 1  
                "AAAAAACCC" +        # 1499-1507: second exon (becomes GGT...TTT after RC, middle exon)
                "A" * 490 +          # 1508-1997: intron 2
                "AAACATGAA" +        # 1998-2006: third exon (becomes TTC...GTT after RC, but will be first exon)
                "A" * 993            # 2007-2999: 3' end
            )  # 3000bp total
            f.write(">gene_minus_multi\n")
            f.write(sequence + "\n")
        
        # Create TSV file for 3-exon gene (1-based coordinates)
        tsv_file = os.path.join(self.temp_dir, "test.tsv")
        with open(tsv_file, 'w') as f:
            f.write("sequence_id\tgene_id\tgene_start\tgene_end\texon_start\texon_end\tstrand\n")
            # Gene spans from first to last exon in TSV 1-based coordinates
            f.write("CONTEXT\tgene_minus_multi\t1001\t2007\t1001\t1009\t-\n")  # First exon (will become last)
            f.write("CONTEXT\tgene_minus_multi\t1500\t2007\t1500\t1508\t-\n")  # Second exon (will become middle)
            f.write("CONTEXT\tgene_minus_multi\t1999\t2007\t1999\t2007\t-\n")  # Third exon (will become first)
        
        # Load with strand normalization
        sequences, annotations = load_gene_contexts_with_annotations(
            fasta_file, tsv_file, filter_invalid_codons=False
        )
        
        self.assertEqual(len(sequences), 1)
        gene = annotations[0]['genes'][0]
        
        # Check that coordinates were flipped and exons reordered
        exons = sorted(gene['exons'], key=lambda x: x['start'])
        self.assertEqual(len(exons), 3)
        
        # Original exon positions (0-based after TSV conversion):
        # Exon 1: 1000-1009, Exon 2: 1499-1508, Exon 3: 1998-2007
        # After RC on 3000bp sequence:
        # Original Exon 3 (1998-2007) -> (3000-2007, 3000-1998) = (993, 1002) -> First exon
        # Original Exon 2 (1499-1508) -> (3000-1508, 3000-1499) = (1492, 1501) -> Second exon  
        # Original Exon 1 (1000-1009) -> (3000-1009, 3000-1000) = (1991, 2000) -> Third exon
        
        self.assertEqual(exons[0]['start'], 993)   # First exon after RC
        self.assertEqual(exons[0]['end'], 1002)
        self.assertEqual(exons[1]['start'], 1492)  # Second exon after RC
        self.assertEqual(exons[1]['end'], 1501)
        self.assertEqual(exons[2]['start'], 1991)  # Third exon after RC  
        self.assertEqual(exons[2]['end'], 2000)


class TestExonBasedCodonValidation(unittest.TestCase):
    """Test start/stop codon validation using exon boundaries."""
    
    def test_plus_strand_valid_exon_codons(self):
        """Test validation with valid codons at exon boundaries."""
        # Create exons with ATG at start of first exon, TAA at end of last exon
        sequence = "ATGAAAAAATTTCCCGGGTAA"  # ATG...TAA
        exons = [
            {'start': 0, 'end': 9},     # ATGAAAAAA (starts with ATG)
            {'start': 9, 'end': 15},    # TTTCCC
            {'start': 15, 'end': 21}    # GGGTAA (ends with TAA)
        ]
        
        self.assertTrue(validate_start_stop_codons_from_exons(sequence, exons, '+'))
    
    def test_plus_strand_invalid_start_codon(self):
        """Test validation with invalid start codon."""
        sequence = "TTGAAAAAATAACTTTTTTTTTTT"  # TTG...TAA... (invalid start)
        exons = [
            {'start': 0, 'end': 9},    # TTGAAAAAA
            {'start': 9, 'end': 15},   # TAACTT
            {'start': 15, 'end': 24}   # TTTTTTTTT
        ]
        
        self.assertFalse(validate_start_stop_codons_from_exons(sequence, exons, '+'))
    
    def test_plus_strand_invalid_stop_codon(self):
        """Test validation with invalid stop codon.""" 
        sequence = "ATGAAAAAATCACTTTTTTTTTTT"  # ATG...TCA... (invalid stop)
        exons = [
            {'start': 0, 'end': 9},    # ATGAAAAAA
            {'start': 9, 'end': 15},   # TCACTT
            {'start': 15, 'end': 24}   # TTTTTTTTT  
        ]
        
        self.assertFalse(validate_start_stop_codons_from_exons(sequence, exons, '+'))
    
    def test_minus_strand_valid_exon_codons(self):
        """Test validation with valid codons for minus strand."""
        # For minus strand, we check reverse complement
        # Last exon should have ATG when reverse complemented
        # First exon should have stop codon when reverse complemented
        sequence = "TTATTTTTTTTTCATAAAAAAA"  # This will give ATG...TAA when checking minus strand
        # When reverse complemented: "TTTTTTCTATTTAAAAAAATAAA"
        # Last exon (positions 18-21): CAT -> ATG when RC
        # First exon (positions 0-3): TTA -> TAA when RC
        exons = [
            {'start': 0, 'end': 3},    # TTA (becomes TAA when RC) - stop codon
            {'start': 12, 'end': 15},  # CAT (becomes ATG when RC) - but this is not the last exon
            {'start': 18, 'end': 21}   # AAA (becomes TTT when RC) - this needs to be CAT for ATG
        ]
        
        # Let me fix this sequence to be correct
        sequence = "TTATTTTTTTTTTCATTTTCAT"  # Designed for minus strand validation
        # Reverse complement: "ATGAAAATGAAAAAAAATAAA"
        # We want first exon to have stop codon (TTA->TAA) and last exon to have start codon (CAT->ATG)
        exons = [
            {'start': 0, 'end': 3},    # TTA -> TAA when RC (stop codon)
            {'start': 11, 'end': 14},  # TTC -> GAA when RC
            {'start': 19, 'end': 22}   # CAT -> ATG when RC (start codon)
        ]
        
        self.assertTrue(validate_start_stop_codons_from_exons(sequence, exons, '-'))
    
    def test_gene_boundaries_vs_exon_boundaries(self):
        """Test that codons are checked from exon boundaries, not gene boundaries."""
        # Gene boundaries don't have valid codons, but exon boundaries do
        sequence = "CCCATGAAAAAATAACCCGGGG"
        #           012345678901234567890123
        #           CCC ATG AAAAAA TAA CCC GGGG
        
        # Gene spans positions 0-22, but doesn't start/end with valid codons
        # However, exons at positions 3-9 and 12-18 do have valid codons
        exons = [
            {'start': 3, 'end': 9},    # ATGAAA (starts with ATG)
            {'start': 12, 'end': 15}   # TAA (stops with TAA)
        ]
        
        # This should be valid because we check exon boundaries
        self.assertTrue(validate_start_stop_codons_from_exons(sequence, exons, '+'))
        
        # The old gene-boundary method would fail
        from utils.dna_processor import validate_start_stop_codons
        self.assertFalse(validate_start_stop_codons(sequence, 0, 22, '+'))  # Gene boundaries
    
    def test_gene_length_not_multiple_of_3(self):
        """Test that gene length doesn't need to be multiple of 3."""
        # Gene with total length of 10bp (not divisible by 3)
        sequence = "ATGAAATAAG"  # 10bp: ATG...TAA + G
        exons = [
            {'start': 0, 'end': 6},    # ATGAAA  
            {'start': 6, 'end': 9}     # TAA (stop codon TAA, positions 6-8)
        ]
        
        self.assertTrue(validate_start_stop_codons_from_exons(sequence, exons, '+'))
    
    def test_empty_exons_list(self):
        """Test that empty exons list returns False."""
        sequence = "ATGAAATAAG"
        exons = []
        
        self.assertFalse(validate_start_stop_codons_from_exons(sequence, exons, '+'))


class TestIntegratedCoordinateHandling(unittest.TestCase):
    """Test integrated coordinate handling in the full pipeline."""
    
    def setUp(self):
        """Set up test data."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test files."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_full_pipeline_minus_strand_with_exon_codons(self):
        """Test the full pipeline with minus strand gene and exon-based codon validation."""
        # Create a gene context where gene boundaries != exon boundaries
        # and codons are only valid at exon boundaries
        
        fasta_file = os.path.join(self.temp_dir, "test.fna") 
        with open(fasta_file, 'w') as f:
            # 4000bp sequence with minus strand gene
            # Design so exons have valid codons but gene boundaries don't
            sequence = (
                "A" * 2000 +           # Flanking region (0-1999)
                "CCCC" +               # Gene start padding (2000-2003, invalid)
                "TTATTTAAT" +          # First exon: TTA...AAT -> ATT...TAA when RC (2004-2012)
                "AAAA" +               # Intron (2013-2016)
                "AATTTTCAT" +          # Second exon: AAT...CAT -> ATG...ATT when RC (2017-2025)  
                "GGGG" +               # Gene end padding (2026-2029, invalid)
                "A" * 1970             # Flanking region (2030-3999)
            )  # 4000bp total
            f.write(">gene_minus_exon_codons\n")
            f.write(sequence + "\n")
        
        tsv_file = os.path.join(self.temp_dir, "test.tsv")
        with open(tsv_file, 'w') as f:
            f.write("sequence_id\tgene_id\tgene_start\tgene_end\texon_start\texon_end\tstrand\n")
            # Gene boundaries include padding (1-based coordinates)
            f.write("CONTEXT\tgene_minus_exon_codons\t2001\t2030\t2005\t2013\t-\n")  # First exon
            f.write("CONTEXT\tgene_minus_exon_codons\t2001\t2030\t2018\t2026\t-\n")  # Second exon
        
        # Load with codon filtering enabled
        sequences, annotations = load_gene_contexts_with_annotations(
            fasta_file, tsv_file, filter_invalid_codons=True
        )
        
        # Should successfully load the gene because exon boundaries have valid codons
        self.assertEqual(len(sequences), 1)
        self.assertEqual(len(annotations), 1)
        
        gene = annotations[0]['genes'][0]
        
        # Check that gene was normalized to + strand
        self.assertEqual(gene['strand'], '+')
        
        # Check that exons were properly transformed and sorted
        exons = sorted(gene['exons'], key=lambda x: x['start'])
        self.assertEqual(len(exons), 2)
        
        # Verify that the gene has valid start/stop codons at exon boundaries
        final_sequence = sequences[0]
        gene_exons = gene['exons']
        self.assertTrue(validate_start_stop_codons_from_exons(final_sequence, gene_exons, '+'))


def run_coordinate_tests():
    """Run all coordinate handling tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestTSVCoordinateParsing,
        TestMinusStrandCoordinates,
        TestExonBasedCodonValidation,
        TestIntegratedCoordinateHandling,
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_coordinate_tests()
    sys.exit(0 if success else 1)

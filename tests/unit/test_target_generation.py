#!/usr/bin/env python3
"""
Unit tests for gene prediction target generation.

Tests the GenePredictionTargetGenerator to ensure it correctly assigns
START, STOP, GENE_BODY, UTR5, UTR3, and INTERGENIC labels for both
single and multi-exon genes.
"""

import unittest
import numpy as np
from typing import List, Dict

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from utils.gene_prediction_processor import GenePredictionTargetGenerator
from utils.constants import GenePredictionClass


class TestTargetGeneration(unittest.TestCase):
    """Test target generation for gene boundary detection."""

    def setUp(self):
        """Set up test fixtures."""
        self.generator = GenePredictionTargetGenerator()
        self.class_names = {
            GenePredictionClass.INTERGENIC: 'INTERGENIC',
            GenePredictionClass.UTR5: 'UTR5',
            GenePredictionClass.START: 'START',
            GenePredictionClass.GENE_BODY: 'GENE_BODY',
            GenePredictionClass.STOP: 'STOP',
            GenePredictionClass.UTR3: 'UTR3'
        }

    def create_test_sequence_with_gene(self, gene_start: int, cds_start: int, 
                                     cds_end: int, gene_end: int, 
                                     start_codon: str = "ATG", 
                                     stop_codon: str = "TAG") -> str:
        """
        Create a test DNA sequence with a single gene.
        
        Args:
            gene_start: Start of gene (0-based)
            cds_start: Start of CDS (0-based)
            cds_end: End of CDS (0-based, inclusive)
            gene_end: End of gene (0-based, inclusive)
            start_codon: START codon (default: ATG)
            stop_codon: STOP codon (default: TAG)
            
        Returns:
            DNA sequence string
        """
        total_length = gene_end + 100  # Add some buffer
        sequence = ['N'] * total_length
        
        # Fill intergenic regions with random bases
        intergenic_bases = ['A', 'T', 'G', 'C']
        for i in range(total_length):
            if i < gene_start or i > gene_end:
                sequence[i] = intergenic_bases[i % 4]
        
        # Fill UTR regions with random bases (no ATG or stop codons)
        utr_bases = ['A', 'A', 'C', 'C']  # Avoid creating random ATG/stops
        for i in range(gene_start, cds_start):
            sequence[i] = utr_bases[i % 4]
        for i in range(cds_end + 1, gene_end + 1):
            sequence[i] = utr_bases[i % 4]
        
        # Place START codon
        for i, base in enumerate(start_codon):
            sequence[cds_start + i] = base
        
        # Fill CDS body with random bases (avoid stop codons)
        cds_body_bases = ['G', 'C', 'G', 'C', 'A', 'A']
        for i in range(cds_start + 3, cds_end - 2):
            sequence[i] = cds_body_bases[i % 6]
        
        # Place STOP codon
        for i, base in enumerate(stop_codon):
            sequence[cds_end - 2 + i] = base
        
        return ''.join(sequence)

    def test_single_exon_gene_with_utrs(self):
        """Test target generation for single exon gene with UTRs."""
        # Create test sequence: UTR5(500bp) + CDS(900bp) + UTR3(500bp)
        gene_start = 1000
        cds_start = gene_start + 500  # 500bp UTR5
        cds_end = cds_start + 899     # 900bp CDS (0-based inclusive)
        gene_end = cds_end + 500      # 500bp UTR3
        
        sequence = self.create_test_sequence_with_gene(
            gene_start, cds_start, cds_end, gene_end
        )
        
        # Create gene data
        genes = [{
            'sequence_id': 'test_seq',
            'gene_id': 'test_gene_001',
            'start': gene_start,
            'end': gene_end,
            'strand': '+',
            'exons': [{'start': cds_start, 'end': cds_end + 1}]  # Exclusive end
        }]
        
        # Generate targets
        targets = self.generator.generate_targets(sequence, genes)
        
        # Verify sequence length
        self.assertEqual(len(targets), len(sequence))
        
        # Check UTR5 region
        utr5_targets = targets[gene_start:cds_start]
        self.assertTrue(np.all(utr5_targets == GenePredictionClass.UTR5),
                       "UTR5 region should all be UTR5 class")
        
        # Check START codon (3 bp)
        start_targets = targets[cds_start:cds_start + 3]
        self.assertTrue(np.all(start_targets == GenePredictionClass.START),
                       "START codon should all be START class")
        
        # Check GENE_BODY region
        gene_body_targets = targets[cds_start + 3:cds_end - 2]
        self.assertTrue(np.all(gene_body_targets == GenePredictionClass.GENE_BODY),
                       "Gene body should all be GENE_BODY class")
        
        # Check STOP codon (3 bp)
        stop_targets = targets[cds_end - 2:cds_end + 1]
        self.assertTrue(np.all(stop_targets == GenePredictionClass.STOP),
                       "STOP codon should all be STOP class")
        
        # Check UTR3 region
        utr3_targets = targets[cds_end + 1:gene_end + 1]
        self.assertTrue(np.all(utr3_targets == GenePredictionClass.UTR3),
                       "UTR3 region should all be UTR3 class")
        
        # Check intergenic regions
        intergenic_before = targets[:gene_start]
        intergenic_after = targets[gene_end + 1:]
        self.assertTrue(np.all(intergenic_before == GenePredictionClass.INTERGENIC),
                       "Region before gene should be INTERGENIC")
        self.assertTrue(np.all(intergenic_after == GenePredictionClass.INTERGENIC),
                       "Region after gene should be INTERGENIC")
        
        # Verify actual codons in sequence
        start_codon = sequence[cds_start:cds_start + 3]
        stop_codon = sequence[cds_end - 2:cds_end + 1]
        self.assertEqual(start_codon, "ATG", "START codon should be ATG")
        self.assertEqual(stop_codon, "TAG", "STOP codon should be TAG")

    def test_single_exon_gene_no_utrs(self):
        """Test target generation for single exon gene without UTRs."""
        # Gene boundaries exactly match CDS boundaries
        gene_start = 2000
        cds_start = gene_start
        cds_end = gene_start + 899  # 900bp CDS
        gene_end = cds_end
        
        sequence = self.create_test_sequence_with_gene(
            gene_start, cds_start, cds_end, gene_end, stop_codon="TAA"
        )
        
        genes = [{
            'sequence_id': 'test_seq',
            'gene_id': 'test_gene_002',
            'start': gene_start,
            'end': gene_end,
            'strand': '+',
            'exons': [{'start': cds_start, 'end': cds_end + 1}]  # Exclusive end
        }]
        
        targets = self.generator.generate_targets(sequence, genes)
        
        # Should have no UTR regions
        utr5_count = np.sum(targets == GenePredictionClass.UTR5)
        utr3_count = np.sum(targets == GenePredictionClass.UTR3)
        self.assertEqual(utr5_count, 0, "Should have no UTR5 regions")
        self.assertEqual(utr3_count, 0, "Should have no UTR3 regions")
        
        # Check gene regions
        start_count = np.sum(targets == GenePredictionClass.START)
        stop_count = np.sum(targets == GenePredictionClass.STOP)
        gene_body_count = np.sum(targets == GenePredictionClass.GENE_BODY)
        
        self.assertEqual(start_count, 3, "Should have exactly 3 START positions")
        self.assertEqual(stop_count, 3, "Should have exactly 3 STOP positions")
        self.assertEqual(gene_body_count, 894, "Should have 894 GENE_BODY positions")
        
        # Verify codons
        start_codon = sequence[cds_start:cds_start + 3]
        stop_codon = sequence[cds_end - 2:cds_end + 1]
        self.assertEqual(start_codon, "ATG")
        self.assertEqual(stop_codon, "TAA")

    def test_multi_exon_gene(self):
        """Test target generation for multi-exon gene."""
        # Create a gene with 3 exons
        gene_start = 3000
        gene_end = 8000
        
        # Define exons with introns between them (exclusive end coordinates)
        exons = [
            {'start': 3500, 'end': 3801},   # Exon 1: 301bp (3500-3800 inclusive)
            {'start': 5000, 'end': 5201},   # Exon 2: 201bp (5000-5200 inclusive)
            {'start': 7000, 'end': 7401}    # Exon 3: 401bp (7000-7400 inclusive)
        ]
        
        # Create sequence with multiple exons
        total_length = gene_end + 500
        sequence = ['A'] * total_length  # Fill with A's
        
        # Place START codon in first exon
        first_exon = exons[0]
        sequence[first_exon['start']:first_exon['start'] + 3] = ['A', 'T', 'G']
        
        # Fill exon bodies with non-stop codons
        for exon in exons:
            for i in range(exon['start'] + 3, exon['end'] - 3):  # Leave space for STOP
                sequence[i] = ['G', 'C', 'A'][i % 3]
        
        # Place STOP codon in last exon (last 3 bp before exclusive end)
        last_exon = exons[-1]
        sequence[last_exon['end'] - 3:last_exon['end']] = ['T', 'G', 'A']
        
        sequence = ''.join(sequence)
        
        genes = [{
            'sequence_id': 'test_seq',
            'gene_id': 'test_gene_multi',
            'start': gene_start,
            'end': gene_end,
            'strand': '+',
            'exons': exons
        }]
        
        targets = self.generator.generate_targets(sequence, genes)
        
        # Check that START is only in first exon
        start_positions = np.where(targets == GenePredictionClass.START)[0]
        self.assertEqual(len(start_positions), 3)
        self.assertTrue(np.all(start_positions >= first_exon['start']))
        self.assertTrue(np.all(start_positions < first_exon['start'] + 3))
        
        # Check that STOP is only in last exon
        stop_positions = np.where(targets == GenePredictionClass.STOP)[0]
        self.assertEqual(len(stop_positions), 3)
        self.assertTrue(np.all(stop_positions >= last_exon['end'] - 3))
        self.assertTrue(np.all(stop_positions < last_exon['end']))  # Exclusive end
        
        # Check that all exon positions are marked (except START/STOP)
        for exon in exons:
            exon_targets = targets[exon['start']:exon['end']]  # Exclusive end
            # Should contain START, GENE_BODY, and/or STOP
            valid_classes = {GenePredictionClass.START, GenePredictionClass.GENE_BODY, 
                           GenePredictionClass.STOP}
            self.assertTrue(np.all(np.isin(exon_targets, list(valid_classes))),
                           f"All positions in exon {exon} should be exonic classes")
        
        # Check UTR regions
        utr5_targets = targets[gene_start:first_exon['start']]
        utr3_targets = targets[last_exon['end']:gene_end + 1]  # Exclusive end
        
        if len(utr5_targets) > 0:
            self.assertTrue(np.all(utr5_targets == GenePredictionClass.UTR5))
        if len(utr3_targets) > 0:
            self.assertTrue(np.all(utr3_targets == GenePredictionClass.UTR3))
        
        # Verify actual codons
        start_codon = sequence[first_exon['start']:first_exon['start'] + 3]
        stop_codon = sequence[last_exon['end'] - 3:last_exon['end']]  # Exclusive end
        self.assertEqual(start_codon, "ATG")
        self.assertEqual(stop_codon, "TGA")

    def test_reverse_strand_gene(self):
        """Test target generation for reverse strand gene."""
        gene_start = 4000
        cds_start = gene_start + 200  # UTR3 for reverse strand
        cds_end = cds_start + 599     # 600bp CDS
        gene_end = cds_end + 300      # UTR5 for reverse strand
        
        sequence = self.create_test_sequence_with_gene(
            gene_start, cds_start, cds_end, gene_end, stop_codon="TGA"
        )
        
        genes = [{
            'sequence_id': 'test_seq',
            'gene_id': 'test_gene_reverse',
            'start': gene_start,
            'end': gene_end,
            'strand': '-',
            'exons': [{'start': cds_start, 'end': cds_end + 1}]  # Exclusive end
        }]
        
        targets = self.generator.generate_targets(sequence, genes)
        
        # For reverse strand:
        # - Biologically, the first exon is the last by coordinate
        # - UTR5 is at the end (high coordinates)
        # - UTR3 is at the beginning (low coordinates)
        
        # Check UTR3 (at gene start for reverse strand)
        utr3_targets = targets[gene_start:cds_start]
        self.assertTrue(np.all(utr3_targets == GenePredictionClass.UTR3),
                       "UTR3 should be at gene start for reverse strand")
        
        # Check UTR5 (at gene end for reverse strand)  
        utr5_targets = targets[cds_end + 1:gene_end + 1]
        self.assertTrue(np.all(utr5_targets == GenePredictionClass.UTR5),
                       "UTR5 should be at gene end for reverse strand")
        
        # START and STOP should still be properly assigned
        start_count = np.sum(targets == GenePredictionClass.START)
        stop_count = np.sum(targets == GenePredictionClass.STOP)
        self.assertEqual(start_count, 3, "Should have 3 START positions")
        self.assertEqual(stop_count, 3, "Should have 3 STOP positions")

    def test_multiple_genes(self):
        """Test target generation with multiple genes."""
        # Create two non-overlapping genes
        sequence_length = 15000
        sequence = ['A'] * sequence_length
        
        # Gene 1: positions 1000-2000
        gene1_start, gene1_cds_start = 1000, 1200
        gene1_cds_end, gene1_end = 1799, 2000
        
        # Gene 2: positions 5000-6500  
        gene2_start, gene2_cds_start = 5000, 5100
        gene2_cds_end, gene2_end = 6399, 6500
        
        # Place codons for both genes
        sequence[gene1_cds_start:gene1_cds_start + 3] = ['A', 'T', 'G']
        sequence[gene1_cds_end - 2:gene1_cds_end + 1] = ['T', 'A', 'G']
        sequence[gene2_cds_start:gene2_cds_start + 3] = ['A', 'T', 'G'] 
        sequence[gene2_cds_end - 2:gene2_cds_end + 1] = ['T', 'A', 'A']
        
        sequence = ''.join(sequence)
        
        genes = [
            {
                'sequence_id': 'test_seq',
                'gene_id': 'gene1',
                'start': gene1_start,
                'end': gene1_end,
                'strand': '+',
                'exons': [{'start': gene1_cds_start, 'end': gene1_cds_end}]
            },
            {
                'sequence_id': 'test_seq', 
                'gene_id': 'gene2',
                'start': gene2_start,
                'end': gene2_end,
                'strand': '+',
                'exons': [{'start': gene2_cds_start, 'end': gene2_cds_end}]
            }
        ]
        
        targets = self.generator.generate_targets(sequence, genes)
        
        # Should have exactly 2 START and 2 STOP regions
        start_count = np.sum(targets == GenePredictionClass.START)
        stop_count = np.sum(targets == GenePredictionClass.STOP)
        self.assertEqual(start_count, 6, "Should have 6 START positions (2 genes × 3bp)")
        self.assertEqual(stop_count, 6, "Should have 6 STOP positions (2 genes × 3bp)")
        
        # Check intergenic region between genes
        intergenic_between = targets[gene1_end + 1:gene2_start]
        self.assertTrue(np.all(intergenic_between == GenePredictionClass.INTERGENIC),
                       "Region between genes should be INTERGENIC")

    def test_codon_validation_comprehensive(self):
        """Comprehensive test to validate all START/STOP codons are correct."""
        # Create sequence with multiple genes and different stop codons
        sequence_length = 20000
        sequence = ['G'] * sequence_length  # Fill with G to avoid random codons
        
        genes = []
        stop_codons = ['TAA', 'TAG', 'TGA']
        
        for i, stop_codon in enumerate(stop_codons):
            gene_start = 2000 + i * 5000
            cds_start = gene_start + 100
            cds_end = cds_start + 899  # 900bp CDS
            gene_end = cds_end + 100
            
            # Place START codon
            sequence[cds_start:cds_start + 3] = ['A', 'T', 'G']
            
            # Place STOP codon
            sequence[cds_end - 2:cds_end + 1] = list(stop_codon)
            
            genes.append({
                'sequence_id': 'test_seq',
                'gene_id': f'gene_{i+1}',
                'start': gene_start,
                'end': gene_end,
                'strand': '+',
                'exons': [{'start': cds_start, 'end': cds_end + 1}]  # Exclusive end
            })
        
        sequence = ''.join(sequence)
        targets = self.generator.generate_targets(sequence, genes)
        
        # Validate all START positions contain ATG
        start_positions = np.where(targets == GenePredictionClass.START)[0]
        for pos in start_positions[::3]:  # Every 3rd position is start of codon
            codon = sequence[pos:pos + 3]
            self.assertEqual(codon, 'ATG', f"Position {pos} should contain ATG, got {codon}")
        
        # Validate all STOP positions contain valid stop codons
        stop_positions = np.where(targets == GenePredictionClass.STOP)[0]
        valid_stops = {'TAA', 'TAG', 'TGA'}
        for pos in stop_positions[::3]:  # Every 3rd position is start of codon
            codon = sequence[pos:pos + 3]
            self.assertIn(codon, valid_stops, f"Position {pos} should contain valid stop codon, got {codon}")
        
        # Count total genes and validate counts
        self.assertEqual(len(start_positions), 9, "Should have 9 START positions (3 genes × 3bp)")
        self.assertEqual(len(stop_positions), 9, "Should have 9 STOP positions (3 genes × 3bp)")


if __name__ == '__main__':
    unittest.main()

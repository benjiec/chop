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

    def test_single_exon_gene_with_assumed_utrs(self):
        """Test target generation for single exon gene with assumed 500bp UTRs."""
        # With new logic: UTRs are assumed to be 500bp upstream/downstream of CDS
        # regardless of gene annotation boundaries
        cds_start = 1500  # Position of START codon
        cds_end = cds_start + 899  # 900bp CDS (0-based inclusive)
        
        # Create sequence with enough space for 500bp UTRs on each side
        sequence_start = cds_start - 600  # Extra space before UTR5
        sequence_end = cds_end + 600      # Extra space after UTR3
        
        # Create test sequence
        total_length = sequence_end + 100
        sequence = ['A'] * total_length  # Fill with A's
        
        # Place START codon
        sequence[cds_start:cds_start + 3] = ['A', 'T', 'G']
        
        # Fill CDS body with non-stop codons
        for i in range(cds_start + 3, cds_end - 2):
            sequence[i] = ['G', 'C', 'A'][i % 3]
        
        # Place STOP codon
        sequence[cds_end - 2:cds_end + 1] = ['T', 'A', 'G']
        
        sequence = ''.join(sequence)
        
        # Create gene data (gene boundaries are ignored, only CDS boundaries matter)
        genes = [{
            'sequence_id': 'test_seq',
            'gene_id': 'test_gene_001',
            'start': cds_start - 100,  # Gene annotation (ignored by new logic)
            'end': cds_end + 100,      # Gene annotation (ignored by new logic)
            'strand': '+',
            'exons': [{'start': cds_start, 'end': cds_end + 1}]  # Exclusive end
        }]
        
        # Generate targets
        targets = self.generator.generate_targets(sequence, genes)
        
        # Verify sequence length
        self.assertEqual(len(targets), len(sequence))
        
        # With new logic: UTR5 is 500bp upstream of CDS start
        utr5_start = max(0, cds_start - 500)
        utr5_end = cds_start - 1
        
        # UTR3 is 500bp downstream of CDS end
        utr3_start = cds_end + 1
        utr3_end = min(len(sequence) - 1, cds_end + 500)
        
        # Check UTR5 region (500bp upstream of CDS)
        if utr5_start <= utr5_end:
            utr5_targets = targets[utr5_start:utr5_end + 1]
            self.assertTrue(np.all(utr5_targets == GenePredictionClass.UTR5),
                           f"UTR5 region ({utr5_start}-{utr5_end}) should all be UTR5 class")
        
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
        
        # Check UTR3 region (500bp downstream of CDS)
        if utr3_start <= utr3_end:
            utr3_targets = targets[utr3_start:utr3_end + 1]
            self.assertTrue(np.all(utr3_targets == GenePredictionClass.UTR3),
                           f"UTR3 region ({utr3_start}-{utr3_end}) should all be UTR3 class")
        
        # Check intergenic regions
        if utr5_start > 0:
            intergenic_before = targets[:utr5_start]
            self.assertTrue(np.all(intergenic_before == GenePredictionClass.INTERGENIC),
                           "Region before UTR5 should be INTERGENIC")
        
        if utr3_end < len(sequence) - 1:
            intergenic_after = targets[utr3_end + 1:]
            self.assertTrue(np.all(intergenic_after == GenePredictionClass.INTERGENIC),
                           "Region after UTR3 should be INTERGENIC")
        
        # Verify actual codons in sequence
        start_codon = sequence[cds_start:cds_start + 3]
        stop_codon = sequence[cds_end - 2:cds_end + 1]
        self.assertEqual(start_codon, "ATG", "START codon should be ATG")
        self.assertEqual(stop_codon, "TAG", "STOP codon should be TAG")

    def test_single_exon_gene_at_sequence_boundary(self):
        """Test target generation for gene at sequence boundary where UTRs are truncated."""
        # Place gene near the start of sequence so UTR5 gets truncated
        cds_start = 200  # Only 200bp available for UTR5 (instead of 500bp)
        cds_end = cds_start + 899  # 900bp CDS
        
        # Create shorter sequence to test boundary conditions
        total_length = cds_end + 200  # Only 200bp available for UTR3
        sequence = ['A'] * total_length
        
        # Place START codon
        sequence[cds_start:cds_start + 3] = ['A', 'T', 'G']
        
        # Fill CDS body with non-stop codons
        for i in range(cds_start + 3, cds_end - 2):
            sequence[i] = ['G', 'C', 'A'][i % 3]
        
        # Place STOP codon
        sequence[cds_end - 2:cds_end + 1] = ['T', 'A', 'A']
        
        sequence = ''.join(sequence)
        
        genes = [{
            'sequence_id': 'test_seq',
            'gene_id': 'test_gene_boundary',
            'start': 0,  # Gene annotation (ignored)
            'end': total_length - 1,  # Gene annotation (ignored)
            'strand': '+',
            'exons': [{'start': cds_start, 'end': cds_end + 1}]  # Exclusive end
        }]
        
        targets = self.generator.generate_targets(sequence, genes)
        
        # With new logic: UTRs are limited by sequence boundaries
        expected_utr5_start = max(0, cds_start - 500)  # Should be 0
        expected_utr5_end = cds_start - 1  # Should be 199
        
        expected_utr3_start = cds_end + 1
        expected_utr3_end = min(len(sequence) - 1, cds_end + 500)  # Should be limited
        
        # Check truncated UTR5 region
        utr5_targets = targets[expected_utr5_start:expected_utr5_end + 1]
        self.assertTrue(np.all(utr5_targets == GenePredictionClass.UTR5),
                       f"UTR5 region ({expected_utr5_start}-{expected_utr5_end}) should all be UTR5")
        
        # Check gene regions
        start_count = np.sum(targets == GenePredictionClass.START)
        stop_count = np.sum(targets == GenePredictionClass.STOP)
        gene_body_count = np.sum(targets == GenePredictionClass.GENE_BODY)
        
        self.assertEqual(start_count, 3, "Should have exactly 3 START positions")
        self.assertEqual(stop_count, 3, "Should have exactly 3 STOP positions")
        self.assertEqual(gene_body_count, 894, "Should have 894 GENE_BODY positions")
        
        # Check truncated UTR3 region
        utr3_targets = targets[expected_utr3_start:expected_utr3_end + 1]
        self.assertTrue(np.all(utr3_targets == GenePredictionClass.UTR3),
                       f"UTR3 region ({expected_utr3_start}-{expected_utr3_end}) should all be UTR3")
        
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
        
        # Check UTR regions (now based on 500bp assumption from CDS boundaries)
        expected_utr5_start = max(0, first_exon['start'] - 500)
        expected_utr5_end = first_exon['start'] - 1
        
        expected_utr3_start = last_exon['end']
        expected_utr3_end = min(len(sequence) - 1, last_exon['end'] + 500 - 1)
        
        if expected_utr5_start <= expected_utr5_end:
            utr5_targets = targets[expected_utr5_start:expected_utr5_end + 1]
            self.assertTrue(np.all(utr5_targets == GenePredictionClass.UTR5),
                           f"UTR5 region ({expected_utr5_start}-{expected_utr5_end}) should all be UTR5")
        
        if expected_utr3_start <= expected_utr3_end:
            utr3_targets = targets[expected_utr3_start:expected_utr3_end + 1]
            self.assertTrue(np.all(utr3_targets == GenePredictionClass.UTR3),
                           f"UTR3 region ({expected_utr3_start}-{expected_utr3_end}) should all be UTR3")
        
        # Verify actual codons
        start_codon = sequence[first_exon['start']:first_exon['start'] + 3]
        stop_codon = sequence[last_exon['end'] - 3:last_exon['end']]  # Exclusive end
        self.assertEqual(start_codon, "ATG")
        self.assertEqual(stop_codon, "TGA")

    def test_reverse_strand_gene(self):
        """Test target generation for reverse strand gene."""
        cds_start = 4200  # Start of CDS in genomic coordinates
        cds_end = cds_start + 599  # 600bp CDS
        
        # Create sequence with space for 500bp UTRs
        total_length = cds_end + 600
        sequence = ['A'] * total_length
        
        # Place START codon at end of CDS (biologically 5' for reverse strand)
        sequence[cds_end - 3:cds_end] = ['A', 'T', 'G']  # START codon
        
        # Fill CDS body
        for i in range(cds_start + 3, cds_end - 3):
            sequence[i] = ['G', 'C', 'A'][i % 3]
        
        # Place STOP codon at start of CDS (biologically 3' for reverse strand)
        sequence[cds_start:cds_start + 3] = ['T', 'G', 'A']  # STOP codon
        
        sequence = ''.join(sequence)
        
        genes = [{
            'sequence_id': 'test_seq',
            'gene_id': 'test_gene_reverse',
            'start': cds_start - 100,  # Gene annotation (ignored)
            'end': cds_end + 100,      # Gene annotation (ignored)
            'strand': '-',
            'exons': [{'start': cds_start, 'end': cds_end + 1}]  # Exclusive end
        }]
        
        targets = self.generator.generate_targets(sequence, genes)
        
        # For reverse strand with new logic:
        # - UTR3 is 500bp upstream of CDS start (genomically)
        # - UTR5 is 500bp downstream of CDS end (genomically)
        
        expected_utr3_start = max(0, cds_start - 500)
        expected_utr3_end = cds_start - 1
        
        expected_utr5_start = cds_end + 1
        expected_utr5_end = min(len(sequence) - 1, cds_end + 500)
        
        # Check UTR3 (upstream of CDS for reverse strand)
        if expected_utr3_start <= expected_utr3_end:
            utr3_targets = targets[expected_utr3_start:expected_utr3_end + 1]
            self.assertTrue(np.all(utr3_targets == GenePredictionClass.UTR3),
                           f"UTR3 region ({expected_utr3_start}-{expected_utr3_end}) should all be UTR3")
        
        # Check UTR5 (downstream of CDS for reverse strand)  
        if expected_utr5_start <= expected_utr5_end:
            utr5_targets = targets[expected_utr5_start:expected_utr5_end + 1]
            self.assertTrue(np.all(utr5_targets == GenePredictionClass.UTR5),
                           f"UTR5 region ({expected_utr5_start}-{expected_utr5_end}) should all be UTR5")
        
        # START and STOP should still be properly assigned
        start_count = np.sum(targets == GenePredictionClass.START)
        stop_count = np.sum(targets == GenePredictionClass.STOP)
        self.assertEqual(start_count, 3, "Should have 3 START positions")
        self.assertEqual(stop_count, 3, "Should have 3 STOP positions")

    def test_multiple_genes(self):
        """Test target generation with multiple genes."""
        # Create two non-overlapping genes with sufficient space between UTR regions
        sequence_length = 15000
        sequence = ['A'] * sequence_length
        
        # Position genes far enough apart that their UTR regions don't overlap
        gene1_cds_start = 2000
        gene1_cds_end = gene1_cds_start + 599  # 600bp CDS
        
        # Place gene2 far enough that UTR regions don't overlap
        # Gene1 UTR3 ends at gene1_cds_end + 500 = 2600
        # Gene2 UTR5 starts at gene2_cds_start - 500
        # Need gap of 2600 < X < gene2_cds_start - 500, so gene2_cds_start > 3100
        gene2_cds_start = 4000  # Safe distance
        gene2_cds_end = gene2_cds_start + 599  # 600bp CDS
        
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
                'start': gene1_cds_start - 100,  # Gene annotation (ignored)
                'end': gene1_cds_end + 100,      # Gene annotation (ignored)
                'strand': '+',
                'exons': [{'start': gene1_cds_start, 'end': gene1_cds_end + 1}]  # Exclusive end
            },
            {
                'sequence_id': 'test_seq', 
                'gene_id': 'gene2',
                'start': gene2_cds_start - 100,  # Gene annotation (ignored)
                'end': gene2_cds_end + 100,      # Gene annotation (ignored)
                'strand': '+',
                'exons': [{'start': gene2_cds_start, 'end': gene2_cds_end + 1}]  # Exclusive end
            }
        ]
        
        targets = self.generator.generate_targets(sequence, genes)
        
        # Should have exactly 2 START and 2 STOP regions
        start_count = np.sum(targets == GenePredictionClass.START)
        stop_count = np.sum(targets == GenePredictionClass.STOP)
        self.assertEqual(start_count, 6, "Should have 6 START positions (2 genes × 3bp)")
        self.assertEqual(stop_count, 6, "Should have 6 STOP positions (2 genes × 3bp)")
        
        # With new UTR logic, check intergenic region between UTR regions
        gene1_utr3_end = gene1_cds_end + 500
        gene2_utr5_start = gene2_cds_start - 500
        
        # Region between UTR3 of gene1 and UTR5 of gene2 should be intergenic
        intergenic_between = targets[gene1_utr3_end + 1:gene2_utr5_start]
        self.assertTrue(np.all(intergenic_between == GenePredictionClass.INTERGENIC),
                       f"Region between gene UTRs ({gene1_utr3_end + 1}-{gene2_utr5_start - 1}) should be INTERGENIC")

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

    def test_utr_assumption_ignores_gene_boundaries(self):
        """Test that new UTR logic ignores gene annotation boundaries."""
        # Create a gene where the gene annotation boundaries are much wider than CDS
        cds_start = 5000
        cds_end = cds_start + 599  # 600bp CDS
        
        # Gene annotation has very wide boundaries (should be ignored)
        gene_start = cds_start - 2000  # 2000bp before CDS
        gene_end = cds_end + 2000      # 2000bp after CDS
        
        total_length = gene_end + 500
        sequence = ['A'] * total_length
        
        # Place codons
        sequence[cds_start:cds_start + 3] = ['A', 'T', 'G']
        sequence[cds_end - 2:cds_end + 1] = ['T', 'A', 'G']
        
        sequence = ''.join(sequence)
        
        genes = [{
            'sequence_id': 'test_seq',
            'gene_id': 'test_gene_wide',
            'start': gene_start,  # Wide gene annotation
            'end': gene_end,      # Wide gene annotation
            'strand': '+',
            'exons': [{'start': cds_start, 'end': cds_end + 1}]
        }]
        
        targets = self.generator.generate_targets(sequence, genes)
        
        # UTRs should be exactly 500bp from CDS, not based on gene boundaries
        expected_utr5_start = cds_start - 500
        expected_utr5_end = cds_start - 1
        expected_utr3_start = cds_end + 1
        expected_utr3_end = cds_end + 500
        
        # Check UTR5 is exactly 500bp
        utr5_positions = np.where(targets == GenePredictionClass.UTR5)[0]
        expected_utr5_positions = list(range(expected_utr5_start, expected_utr5_end + 1))
        self.assertEqual(list(utr5_positions), expected_utr5_positions,
                        "UTR5 should be exactly 500bp upstream of CDS")
        
        # Check UTR3 is exactly 500bp
        utr3_positions = np.where(targets == GenePredictionClass.UTR3)[0]
        expected_utr3_positions = list(range(expected_utr3_start, expected_utr3_end + 1))
        self.assertEqual(list(utr3_positions), expected_utr3_positions,
                        "UTR3 should be exactly 500bp downstream of CDS")
        
        # Check that regions beyond 500bp UTRs are intergenic
        far_upstream = targets[gene_start:expected_utr5_start]
        far_downstream = targets[expected_utr3_end + 1:gene_end + 1]
        
        self.assertTrue(np.all(far_upstream == GenePredictionClass.INTERGENIC),
                       "Regions beyond 500bp UTR5 should be intergenic")
        self.assertTrue(np.all(far_downstream == GenePredictionClass.INTERGENIC),
                       "Regions beyond 500bp UTR3 should be intergenic")

    def test_overlapping_utr_regions(self):
        """Test behavior when 500bp UTR regions from adjacent genes overlap."""
        # Create two genes close enough that their UTR regions would overlap
        gene1_cds_start = 2000
        gene1_cds_end = gene1_cds_start + 299  # 300bp CDS
        
        gene2_cds_start = gene1_cds_end + 800  # 800bp gap between CDS regions
        gene2_cds_end = gene2_cds_start + 299  # 300bp CDS
        
        total_length = gene2_cds_end + 1000
        sequence = ['A'] * total_length
        
        # Place codons for both genes
        sequence[gene1_cds_start:gene1_cds_start + 3] = ['A', 'T', 'G']
        sequence[gene1_cds_end - 2:gene1_cds_end + 1] = ['T', 'A', 'A']
        sequence[gene2_cds_start:gene2_cds_start + 3] = ['A', 'T', 'G']
        sequence[gene2_cds_end - 2:gene2_cds_end + 1] = ['T', 'A', 'G']
        
        sequence = ''.join(sequence)
        
        genes = [
            {
                'sequence_id': 'test_seq',
                'gene_id': 'gene1',
                'start': gene1_cds_start - 100,
                'end': gene1_cds_end + 100,
                'strand': '+',
                'exons': [{'start': gene1_cds_start, 'end': gene1_cds_end + 1}]
            },
            {
                'sequence_id': 'test_seq',
                'gene_id': 'gene2',
                'start': gene2_cds_start - 100,
                'end': gene2_cds_end + 100,
                'strand': '+',
                'exons': [{'start': gene2_cds_start, 'end': gene2_cds_end + 1}]
            }
        ]
        
        targets = self.generator.generate_targets(sequence, genes)
        
        # Calculate expected UTR regions
        gene1_utr3_start = gene1_cds_end + 1
        gene1_utr3_end = gene1_cds_end + 500
        
        gene2_utr5_start = gene2_cds_start - 500
        gene2_utr5_end = gene2_cds_start - 1
        
        # Check for overlap: gene1 UTR3 and gene2 UTR5
        overlap_start = max(gene1_utr3_start, gene2_utr5_start)
        overlap_end = min(gene1_utr3_end, gene2_utr5_end)
        
        if overlap_start <= overlap_end:
            # There is overlap - later gene processing should overwrite
            overlap_targets = targets[overlap_start:overlap_end + 1]
            # With our current implementation, last gene processed wins
            # This tests that the system handles overlapping UTRs gracefully
            self.assertTrue(len(set(overlap_targets)) <= 2, 
                           "Overlapping UTR region should have consistent labeling")

    def test_introns_classified_as_gene_body_not_intergenic(self):
        """
        CRITICAL TEST: Ensure introns are classified as GENE_BODY, not INTERGENIC.
        
        This test prevents a critical regression where introns were misclassified as
        intergenic regions, which would prevent the model from learning proper gene
        boundaries in multi-exon genes.
        """
        # Create a multi-exon gene with clear introns
        total_length = 8000
        sequence = ['A'] * total_length
        
        # Define a 3-exon gene with introns between exons
        exons = [
            {'start': 2000, 'end': 2301},  # Exon 1: 301bp (2000-2300 inclusive)
            {'start': 3000, 'end': 3201},  # Exon 2: 201bp (3000-3200 inclusive) 
            {'start': 4000, 'end': 4501}   # Exon 3: 501bp (4000-4500 inclusive)
        ]
        
        # Introns are:
        # Intron 1: 2301-2999 (699bp)
        # Intron 2: 3201-3999 (799bp)
        
        # Place START codon in first exon
        sequence[2000:2003] = ['A', 'T', 'G']
        
        # Fill exons with non-stop codons
        for exon in exons:
            for i in range(exon['start'] + 3, exon['end'] - 3):
                sequence[i] = ['G', 'C', 'A'][i % 3]
        
        # Place STOP codon in last exon
        sequence[4498:4501] = ['T', 'A', 'G']
        
        sequence = ''.join(sequence)
        
        genes = [{
            'sequence_id': 'test_seq',
            'gene_id': 'multi_exon_gene',
            'start': 1500,  # Gene boundaries (ignored by new logic)
            'end': 5000,    # Gene boundaries (ignored by new logic)
            'strand': '+',
            'exons': exons
        }]
        
        targets = self.generator.generate_targets(sequence, genes)
        
        # CRITICAL TEST: Check that introns are labeled as GENE_BODY, NOT INTERGENIC
        
        # Intron 1: positions 2301-2999
        intron1_start = 2301
        intron1_end = 2999
        intron1_targets = targets[intron1_start:intron1_end + 1]
        
        # ALL positions in intron should be GENE_BODY
        self.assertTrue(np.all(intron1_targets == GenePredictionClass.GENE_BODY),
                       f"Intron 1 ({intron1_start}-{intron1_end}) should ALL be GENE_BODY, "
                       f"but found classes: {np.unique(intron1_targets)}")
        
        # Intron 2: positions 3201-3999  
        intron2_start = 3201
        intron2_end = 3999
        intron2_targets = targets[intron2_start:intron2_end + 1]
        
        # ALL positions in intron should be GENE_BODY
        self.assertTrue(np.all(intron2_targets == GenePredictionClass.GENE_BODY),
                       f"Intron 2 ({intron2_start}-{intron2_end}) should ALL be GENE_BODY, "
                       f"but found classes: {np.unique(intron2_targets)}")
        
        # Verify that NO intron positions are classified as INTERGENIC
        gene_span_start = exons[0]['start'] 
        gene_span_end = exons[-1]['end']
        gene_span_targets = targets[gene_span_start:gene_span_end]
        
        intergenic_count_in_gene = np.sum(gene_span_targets == GenePredictionClass.INTERGENIC)
        self.assertEqual(intergenic_count_in_gene, 0,
                        f"Found {intergenic_count_in_gene} INTERGENIC positions within gene span "
                        f"({gene_span_start}-{gene_span_end}). Introns should be GENE_BODY!")
        
        # Double-check: verify exons are still properly labeled
        for i, exon in enumerate(exons):
            exon_targets = targets[exon['start']:exon['end']]
            valid_exon_classes = {GenePredictionClass.START, GenePredictionClass.GENE_BODY, 
                                GenePredictionClass.STOP}
            self.assertTrue(np.all(np.isin(exon_targets, list(valid_exon_classes))),
                           f"Exon {i+1} should only contain START/GENE_BODY/STOP classes")
        
        # Verify true intergenic regions are still properly labeled
        upstream_intergenic = targets[:gene_span_start - 500]  # Before UTR5
        downstream_intergenic = targets[gene_span_end + 500:]  # After UTR3
        
        if len(upstream_intergenic) > 0:
            self.assertTrue(np.all(upstream_intergenic == GenePredictionClass.INTERGENIC),
                           "True intergenic regions upstream should be INTERGENIC")
        
        if len(downstream_intergenic) > 0:
            self.assertTrue(np.all(downstream_intergenic == GenePredictionClass.INTERGENIC),
                           "True intergenic regions downstream should be INTERGENIC")
        
        print(f"✓ CRITICAL TEST PASSED: All introns correctly classified as GENE_BODY")
        print(f"  Intron 1 ({intron1_start}-{intron1_end}): {len(intron1_targets)} positions as GENE_BODY")
        print(f"  Intron 2 ({intron2_start}-{intron2_end}): {len(intron2_targets)} positions as GENE_BODY")


if __name__ == '__main__':
    unittest.main()

#!/usr/bin/env python3
"""
Tests for the new preprocessing pipeline.

This module tests the complete preprocessing workflow:
1. GFF parsing 
2. Gene context extraction
3. Strand normalization
4. Coordinate adjustment to gene context coordinates
5. Output file generation
"""

import unittest
import sys
from pathlib import Path
import tempfile
import os
import subprocess

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.preprocess_gene_data import (
    extract_gene_contexts_with_normalization,
    load_genomic_sequences,
    group_genes_by_sequence,
    write_normalized_fasta,
    write_normalized_tsv
)
from scripts.preprocess_gene_data import parse_gff_file
from utils.dna_processor import load_gene_contexts, reverse_complement
from Bio import SeqIO


class TestPreprocessingPipeline(unittest.TestCase):
    """Test the complete preprocessing pipeline."""
    
    def setUp(self):
        """Set up test data files."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create test genomic FASTA
        self.fasta_file = os.path.join(self.temp_dir, "test_genome.fna")
        with open(self.fasta_file, 'w') as f:
            # Sequence with multiple genes, both strands
            # SEQ1: 10000bp with genes at various positions
            f.write(">SEQ1\n")
            sequence = (
                "A" * 1000 +           # 0-999: upstream
                "ATGAAATAA" +          # 1000-1008: + strand gene (ATG...TAA)
                "T" * 2001 +           # 1009-3009: intergenic (extra base to fix coordinates)
                "TTATTTCAT" +          # 3010-3018: - strand gene (will be ATG...TAA when RC)
                "G" * 2999 +           # 3019-6017: intergenic  
                "ATGCCCGGGCCCCCCTAA" + # 6018-6035: + strand multi-exon gene (ATG...TAA)
                "C" * 3000             # 6036-9035: downstream
            )  # Total: 9037bp
            # Break into lines of 80 characters
            for i in range(0, len(sequence), 80):
                f.write(sequence[i:i+80] + "\n")
        
        # Create test GFF file
        self.gff_file = os.path.join(self.temp_dir, "test_annotations.gff")
        with open(self.gff_file, 'w') as f:
            f.write("##gff-version 3\n")
            # Plus strand single-exon gene (ATG...TAA)
            f.write("SEQ1\ttest\tCDS\t1001\t1009\t.\t+\t0\tParent=gene1\n")
            # Minus strand single-exon gene (will be ATG...TAA when RC)  
            f.write("SEQ1\ttest\tCDS\t3011\t3019\t.\t-\t0\tParent=gene2\n")
            # Plus strand multi-exon gene (ATG...CCC in first exon, CCC...TAA in second exon)
            f.write("SEQ1\ttest\tCDS\t6019\t6027\t.\t+\t0\tParent=gene3\n")  # First exon: ATG...CCC
            f.write("SEQ1\ttest\tCDS\t6028\t6036\t.\t+\t0\tParent=gene3\n")  # Second exon: CCC...TAA
    
    def tearDown(self):
        """Clean up test files."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_gff_parsing(self):
        """Test GFF file parsing."""
        genes = parse_gff_file(self.gff_file)
        
        # Should find 3 genes
        self.assertEqual(len(genes), 3)
        
        # Check gene1 (+ strand, single exon)
        gene1 = next(g for g in genes if g['gene_id'] == 'gene1')
        self.assertEqual(gene1['sequence_id'], 'SEQ1')
        self.assertEqual(gene1['start'], 1000)  # 0-based conversion
        self.assertEqual(gene1['end'], 1009)  # GFF 1009 -> 0-based exclusive 1009
        self.assertEqual(gene1['strand'], '+')
        self.assertEqual(len(gene1['exons']), 1)
        
        # Check gene2 (- strand, single exon)
        gene2 = next(g for g in genes if g['gene_id'] == 'gene2')
        self.assertEqual(gene2['sequence_id'], 'SEQ1')
        self.assertEqual(gene2['start'], 3010)  # GFF 3011 -> 0-based 3010  # 0-based conversion
        self.assertEqual(gene2['end'], 3019)  # GFF 3019 -> 0-based exclusive 3019
        self.assertEqual(gene2['strand'], '-')
        self.assertEqual(len(gene2['exons']), 1)
        
        # Check gene3 (+ strand, multi-exon)
        gene3 = next(g for g in genes if g['gene_id'] == 'gene3')
        self.assertEqual(gene3['sequence_id'], 'SEQ1')
        self.assertEqual(gene3['start'], 6018)  # 0-based conversion
        self.assertEqual(gene3['end'], 6036)  # GFF 6036 -> 0-based exclusive 6036
        self.assertEqual(gene3['strand'], '+')
        self.assertEqual(len(gene3['exons']), 2)
    
    def test_gene_context_extraction(self):
        """Test gene context extraction with strand normalization."""
        # Parse genes and load sequences
        genes = parse_gff_file(self.gff_file)
        genomic_sequences = load_genomic_sequences(self.fasta_file)
        genes_by_sequence = group_genes_by_sequence(genes)
        
        # Extract gene contexts
        sequences, annotations = extract_gene_contexts_with_normalization(
            genomic_sequences, genes_by_sequence, flanking_bp=500
        )
        
        # Should get 3 gene contexts (assuming all pass validation)
        self.assertEqual(len(sequences), 3)
        self.assertEqual(len(annotations), 3)
        
        # Check that all gene IDs are represented
        gene_ids = {ann['gene_id'] for ann in annotations}
        expected_ids = {'gene1', 'gene2', 'gene3'}
        self.assertEqual(gene_ids, expected_ids)
        
        # Check that all strands are normalized to '+'
        for annotation in annotations:
            self.assertEqual(annotation['strand'], '+')
    
    def test_plus_strand_gene_context(self):
        """Test gene context extraction for + strand gene."""
        genes = parse_gff_file(self.gff_file)
        genomic_sequences = load_genomic_sequences(self.fasta_file)
        genes_by_sequence = group_genes_by_sequence(genes)
        
        sequences, annotations = extract_gene_contexts_with_normalization(
            genomic_sequences, genes_by_sequence, flanking_bp=100
        )
        
        # Find gene1 (+ strand)
        gene1_ann = next(ann for ann in annotations if ann['gene_id'] == 'gene1')
        gene1_seq_idx = next(i for i, ann in enumerate(annotations) if ann['gene_id'] == 'gene1')
        gene1_seq = sequences[gene1_seq_idx].seq
        
        # Gene should be at position 100 in the context (flanking_bp)
        self.assertEqual(gene1_ann['gene_start'], 100)
        self.assertEqual(gene1_ann['gene_end'], 109)
        
        # Sequence should contain ATG...TAA
        gene_region = str(gene1_seq[100:109])
        self.assertEqual(gene_region, "ATGAAATAA")
        
        # Check exon coordinates
        exons = gene1_ann['exons']
        self.assertEqual(len(exons), 1)
        self.assertEqual(exons[0]['start'], 100)
        self.assertEqual(exons[0]['end'], 109)
    
    def test_minus_strand_gene_context(self):
        """Test gene context extraction for - strand gene with normalization."""
        genes = parse_gff_file(self.gff_file)
        genomic_sequences = load_genomic_sequences(self.fasta_file)
        genes_by_sequence = group_genes_by_sequence(genes)
        
        sequences, annotations = extract_gene_contexts_with_normalization(
            genomic_sequences, genes_by_sequence, flanking_bp=100
        )
        
        # Find gene2 (originally - strand, now normalized to +)
        gene2_ann = next(ann for ann in annotations if ann['gene_id'] == 'gene2')
        gene2_seq_idx = next(i for i, ann in enumerate(annotations) if ann['gene_id'] == 'gene2')
        gene2_seq = sequences[gene2_seq_idx].seq
        
        # Gene should be normalized to + strand
        self.assertEqual(gene2_ann['strand'], '+')
        
        # After strand normalization, coordinates should be flipped
        # Original context was 3010-100 to 3019+100 = 2910-3119 (length 209)
        # After RC: gene at positions 209-9 to 209-0 = 200-209... wait, let me recalculate
        
        # Context length: (3019-3010) + 2*100 = 9 + 200 = 209bp
        # Original gene at positions 100-109 in context
        # After RC: 209-109=100, 209-100=109, so still 100-109
        # But coordinates get reordered by min/max
        expected_start = 100  # This should be the position after normalization
        expected_end = 109
        
        # The gene sequence should be ATG...TAA after reverse complement
        gene_region = str(gene2_seq[gene2_ann['gene_start']:gene2_ann['gene_end']])
        self.assertTrue(gene_region.startswith('ATG'))
        self.assertTrue(gene_region.endswith('TAA'))
    
    def test_multi_exon_gene_context(self):
        """Test gene context extraction for multi-exon gene."""
        genes = parse_gff_file(self.gff_file)
        genomic_sequences = load_genomic_sequences(self.fasta_file)
        genes_by_sequence = group_genes_by_sequence(genes)
        
        sequences, annotations = extract_gene_contexts_with_normalization(
            genomic_sequences, genes_by_sequence, flanking_bp=100
        )
        
        # Find gene3 (multi-exon)
        gene3_ann = next(ann for ann in annotations if ann['gene_id'] == 'gene3')
        
        # Should have 2 exons
        exons = gene3_ann['exons']
        self.assertEqual(len(exons), 2)
        
        # Exons should be in order
        self.assertLess(exons[0]['start'], exons[1]['start'])
        self.assertLess(exons[0]['end'], exons[1]['end'])
        
        # Gene coordinates should span both exons
        self.assertEqual(gene3_ann['gene_start'], exons[0]['start'])
        self.assertEqual(gene3_ann['gene_end'], exons[-1]['end'])
    
    def test_tsv_output_format(self):
        """Test TSV output format and coordinate conversion."""
        genes = parse_gff_file(self.gff_file)
        genomic_sequences = load_genomic_sequences(self.fasta_file)
        genes_by_sequence = group_genes_by_sequence(genes)
        
        sequences, annotations = extract_gene_contexts_with_normalization(
            genomic_sequences, genes_by_sequence, flanking_bp=100
        )
        
        # Write TSV and read it back
        tsv_file = os.path.join(self.temp_dir, "output.tsv")
        write_normalized_tsv(annotations, tsv_file)
        
        # Check TSV format
        with open(tsv_file, 'r') as f:
            lines = f.readlines()
        
        # Should have header + data rows
        self.assertGreater(len(lines), 1)
        
        # Check header
        header = lines[0].strip().split('\t')
        expected_columns = ['sequence_id', 'gene_id', 'gene_start', 'gene_end', 'exon_start', 'exon_end', 'strand']
        self.assertEqual(header, expected_columns)
        
        # Check that all strands are '+'
        for line in lines[1:]:
            if line.strip():
                fields = line.strip().split('\t')
                strand = fields[6]
                self.assertEqual(strand, '+')
        
        # Check that coordinates are 1-based in TSV
        gene1_line = next(line for line in lines[1:] if 'gene1' in line)
        fields = gene1_line.strip().split('\t')
        gene_start = int(fields[2])
        exon_start = int(fields[4])
        self.assertEqual(gene_start, 101)  # 100 + 1 for 1-based conversion
        self.assertEqual(exon_start, 101)  # 100 + 1 for 1-based conversion
    
    def test_fasta_output_format(self):
        """Test FASTA output format."""
        genes = parse_gff_file(self.gff_file)
        genomic_sequences = load_genomic_sequences(self.fasta_file)
        genes_by_sequence = group_genes_by_sequence(genes)
        
        sequences, annotations = extract_gene_contexts_with_normalization(
            genomic_sequences, genes_by_sequence, flanking_bp=100
        )
        
        # Write FASTA and read it back
        fasta_file = os.path.join(self.temp_dir, "output.fna")
        write_normalized_fasta(sequences, fasta_file)
        
        # Read back sequences
        with open(fasta_file) as f:
            records = list(SeqIO.parse(f, "fasta"))
        
        self.assertEqual(len(records), len(sequences))
        
        # Check that sequence IDs match gene IDs
        seq_ids = {record.id for record in records}
        gene_ids = {ann['gene_id'] for ann in annotations}
        self.assertEqual(seq_ids, gene_ids)
        
        # Check sequence content matches
        for record in records:
            # Find corresponding annotation
            annotation = next(ann for ann in annotations if ann['gene_id'] == record.id)
            
            # Check that gene region in sequence has valid start codon
            gene_start = annotation['gene_start']
            gene_region = str(record.seq[gene_start:gene_start+3])
            self.assertEqual(gene_region, 'ATG')


class TestDataLoaderValidation(unittest.TestCase):
    """Test the simplified data loader with validation."""
    
    def setUp(self):
        """Set up test data."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test files."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_normalized_data_loading(self):
        """Test loading of properly normalized data."""
        # Create test normalized FASTA
        fasta_file = os.path.join(self.temp_dir, "normalized.fna")
        with open(fasta_file, 'w') as f:
            f.write(">gene1\n")
            f.write("A" * 100 + "ATGAAATAA" + "G" * 91 + "\n")  # 200bp context
        
        # Create test normalized TSV
        tsv_file = os.path.join(self.temp_dir, "normalized.tsv")
        with open(tsv_file, 'w') as f:
            f.write("sequence_id\tgene_id\tgene_start\tgene_end\texon_start\texon_end\tstrand\n")
            f.write("gene1\tgene1\t101\t109\t101\t109\t+\n")  # 1-based coordinates
        
        # Load data
        sequences, annotations = load_gene_contexts(
            fasta_file, tsv_file, validate_normalization=True
        )
        
        self.assertEqual(len(sequences), 1)
        self.assertEqual(len(annotations), 1)
        
        # Check that data loaded correctly
        gene = annotations[0]['genes'][0]
        self.assertEqual(gene['gene_id'], 'gene1')
        self.assertEqual(gene['strand'], '+')
        self.assertEqual(gene['start'], 100)  # 0-based internally
        self.assertEqual(gene['end'], 109)
    
    def test_validation_rejects_minus_strand(self):
        """Test that validation rejects data with - strand."""
        # Create test FASTA
        fasta_file = os.path.join(self.temp_dir, "invalid.fna")
        with open(fasta_file, 'w') as f:
            f.write(">gene1\n")
            f.write("A" * 200 + "\n")
        
        # Create test TSV with - strand (should be rejected)
        tsv_file = os.path.join(self.temp_dir, "invalid.tsv")
        with open(tsv_file, 'w') as f:
            f.write("sequence_id\tgene_id\tgene_start\tgene_end\texon_start\texon_end\tstrand\n")
            f.write("gene1\tgene1\t101\t109\t101\t109\t-\n")  # Invalid - strand
        
        # Load data with validation
        sequences, annotations = load_gene_contexts(
            fasta_file, tsv_file, validate_normalization=True
        )
        
        # Should reject the gene with - strand
        self.assertEqual(len(sequences), 0)
        self.assertEqual(len(annotations), 0)
    
    def test_validation_rejects_invalid_coordinates(self):
        """Test that validation rejects invalid coordinates."""
        # Create test FASTA (100bp)
        fasta_file = os.path.join(self.temp_dir, "invalid.fna")
        with open(fasta_file, 'w') as f:
            f.write(">gene1\n")
            f.write("A" * 100 + "\n")
        
        # Create test TSV with coordinates outside sequence bounds
        tsv_file = os.path.join(self.temp_dir, "invalid.tsv")
        with open(tsv_file, 'w') as f:
            f.write("sequence_id\tgene_id\tgene_start\tgene_end\texon_start\texon_end\tstrand\n")
            f.write("gene1\tgene1\t101\t200\t101\t200\t+\n")  # End=200 > seq_length=100
        
        # Load data with validation
        sequences, annotations = load_gene_contexts(
            fasta_file, tsv_file, validate_normalization=True
        )
        
        # Should reject the gene with invalid coordinates
        self.assertEqual(len(sequences), 0)
        self.assertEqual(len(annotations), 0)


class TestEndToEndPreprocessing(unittest.TestCase):
    """Test the complete preprocessing pipeline end-to-end."""
    
    def setUp(self):
        """Set up test data."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test files."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_complete_pipeline(self):
        """Test the complete preprocessing pipeline from GFF+FASTA to normalized outputs."""
        # Create test input files
        fasta_file = os.path.join(self.temp_dir, "input.fna")
        with open(fasta_file, 'w') as f:
            f.write(">CHR1\n")
            # Create a sequence with genes on both strands
            sequence = (
                "N" * 2000 +           # Padding
                "ATGAAATAA" +          # + strand gene: ATG...TAA
                "G" * 1000 +           # Spacer
                "TTATTTCAT" +          # - strand gene: will be ATG...TAA when RC
                "N" * 2000             # Padding
            )
            for i in range(0, len(sequence), 80):
                f.write(sequence[i:i+80] + "\n")
        
        gff_file = os.path.join(self.temp_dir, "input.gff")
        with open(gff_file, 'w') as f:
            f.write("##gff-version 3\n")
            f.write("CHR1\ttest\tCDS\t2001\t2009\t.\t+\t0\tParent=gene_plus\n")
            f.write("CHR1\ttest\tCDS\t3010\t3018\t.\t-\t0\tParent=gene_minus\n")
        
        # Run preprocessing script
        output_fasta = os.path.join(self.temp_dir, "output.fna")
        output_tsv = os.path.join(self.temp_dir, "output.tsv")
        
        # Import and run the preprocessing function
        from scripts.preprocess_gene_data import main as preprocess_main
        
        # Mock command line arguments
        import sys
        old_argv = sys.argv
        try:
            sys.argv = [
                'preprocess_gene_data.py',
                '--fasta', fasta_file,
                '--gff', gff_file,
                '--output-fasta', output_fasta,
                '--output-tsv', output_tsv,
                '--flanking-bp', '500'
            ]
            preprocess_main()
        finally:
            sys.argv = old_argv
        
        # Verify outputs exist
        self.assertTrue(os.path.exists(output_fasta))
        self.assertTrue(os.path.exists(output_tsv))
        
        # Load and validate outputs using the data loader
        sequences, annotations = load_gene_contexts(
            output_fasta, output_tsv, validate_normalization=True
        )
        
        # Should successfully load both genes
        self.assertEqual(len(sequences), 2)
        self.assertEqual(len(annotations), 2)
        
        # All genes should be + strand
        for annotation in annotations:
            for gene in annotation['genes']:
                self.assertEqual(gene['strand'], '+')
        
        # All genes should have valid start codons
        for i, annotation in enumerate(annotations):
            sequence = sequences[i]
            gene = annotation['genes'][0]
            start_codon = sequence[gene['start']:gene['start']+3]
            self.assertEqual(start_codon, 'ATG')


def run_preprocessing_tests():
    """Run all preprocessing pipeline tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestPreprocessingPipeline,
        TestDataLoaderValidation,
        TestEndToEndPreprocessing,
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_preprocessing_tests()
    sys.exit(0 if success else 1)

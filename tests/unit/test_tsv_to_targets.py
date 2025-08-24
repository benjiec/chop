#!/usr/bin/env python3
"""
Regression tests for TSV to targets generation.

Tests the conversion of TSV annotations to training tensors,
including sliding window mapping and gene ID tracking.
"""

import unittest
import tempfile
import os
import sys
from pathlib import Path
import csv
import torch

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from utils.dna_processor import DNADataset, load_tsv_annotations
from utils.constants import (
    TSV_COLUMNS, GeneBoundaryClass, ExonIntronClass, 
    UNKNOWN_GENE_ID, DEFAULT_WINDOW_SIZE, DEFAULT_STRIDE
)


class TestTSVToTargets(unittest.TestCase):
    """Test cases for TSV to targets conversion."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def _create_test_tsv(self, rows: list) -> str:
        """Create a temporary TSV file with given rows."""
        tsv_path = os.path.join(self.temp_dir, "test_annotations.tsv")
        with open(tsv_path, 'w', newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(TSV_COLUMNS)
            for row in rows:
                writer.writerow(row)
        return tsv_path
    
    def test_single_exon_gene_targets(self):
        """Test target generation for single-exon gene."""
        # Create test data
        test_sequence = "A" * 2000  # 2kb sequence
        # TSV uses 1-based coordinates, so gene at 501-1000 becomes 500-1000 in 0-based
        tsv_rows = [
            ["test_seq", "gene1", 501, 1000, 501, 1000, "+"]
        ]
        tsv_path = self._create_test_tsv(tsv_rows)
        
        # Load annotations and create dataset
        annotations = load_tsv_annotations(tsv_path)
        dataset = DNADataset(
            sequences=[test_sequence],
            annotations=annotations,
            max_length=2000,
            use_sliding_windows=False
        )
        
        # Get sample
        sample = dataset[0]
        targets = sample['targets']
        
        # Verify gene boundaries
        gene_boundaries = targets['gene_boundaries']
        self.assertEqual(gene_boundaries[500].item(), GeneBoundaryClass.START)
        self.assertEqual(gene_boundaries[1000].item(), GeneBoundaryClass.END)
        
        # Verify exon regions
        exon_intron = targets['exon_intron']
        self.assertTrue(torch.all(exon_intron[500:1000] == ExonIntronClass.EXON))
        self.assertTrue(torch.all(exon_intron[:500] == ExonIntronClass.INTERGENIC))
        self.assertTrue(torch.all(exon_intron[1000:] == ExonIntronClass.INTERGENIC))
        
        # Verify gene ID tracking
        gene_ids = targets['gene_ids']
        self.assertTrue(torch.all(gene_ids[500:1000] == 0))  # First gene gets ID 0
        self.assertTrue(torch.all(gene_ids[:500] == UNKNOWN_GENE_ID))
        self.assertTrue(torch.all(gene_ids[1000:] == UNKNOWN_GENE_ID))
    
    def test_multi_exon_gene_targets(self):
        """Test target generation for multi-exon gene."""
        test_sequence = "A" * 3000
        # TSV uses 1-based coordinates, adjust for conversion to 0-based
        tsv_rows = [
            ["test_seq", "gene1", 501, 2500, 501, 800, "+"],   # Exon 1: 500-799 in 0-based
            ["test_seq", "gene1", 501, 2500, 1201, 1500, "+"], # Exon 2: 1200-1499 in 0-based
            ["test_seq", "gene1", 501, 2500, 2001, 2500, "+"]  # Exon 3: 2000-2499 in 0-based
        ]
        tsv_path = self._create_test_tsv(tsv_rows)
        
        annotations = load_tsv_annotations(tsv_path)
        dataset = DNADataset(
            sequences=[test_sequence],
            annotations=annotations,
            max_length=3000,
            use_sliding_windows=False
        )
        
        sample = dataset[0]
        targets = sample['targets']
        
        # Verify exon regions (0-based coordinates after conversion)
        exon_intron = targets['exon_intron']
        self.assertTrue(torch.all(exon_intron[500:800] == ExonIntronClass.EXON))    # Exon 1: 500-799
        self.assertTrue(torch.all(exon_intron[1200:1500] == ExonIntronClass.EXON))  # Exon 2: 1200-1499
        self.assertTrue(torch.all(exon_intron[2000:2500] == ExonIntronClass.EXON))  # Exon 3: 2000-2499
        
        # Verify intron regions
        self.assertTrue(torch.all(exon_intron[800:1200] == ExonIntronClass.INTRON))   # Intron 1: 800-1199
        self.assertTrue(torch.all(exon_intron[1500:2000] == ExonIntronClass.INTRON))  # Intron 2: 1500-1999
        
        # Verify gene ID consistency
        gene_ids = targets['gene_ids']
        self.assertTrue(torch.all(gene_ids[500:2500] == 0))  # Entire gene region has same ID
    
    def test_overlapping_genes_tracking(self):
        """Test gene ID tracking for overlapping genes."""
        test_sequence = "A" * 3000
        # TSV uses 1-based coordinates, adjust for conversion to 0-based
        tsv_rows = [
            ["test_seq", "gene1", 501, 1500, 501, 1500, "+"],   # Gene 1: 500-1499 in 0-based
            ["test_seq", "gene2", 1001, 2000, 1001, 2000, "-"] # Gene 2: 1000-1999 in 0-based (overlaps with gene 1)
        ]
        tsv_path = self._create_test_tsv(tsv_rows)
        
        annotations = load_tsv_annotations(tsv_path)
        dataset = DNADataset(
            sequences=[test_sequence],
            annotations=annotations,
            max_length=3000,
            use_sliding_windows=False
        )
        
        sample = dataset[0]
        targets = sample['targets']
        gene_ids = targets['gene_ids']
        
        # In overlapping region (1000-1499), the later gene should overwrite
        # This tests that gene ID tracking can handle overlaps
        self.assertTrue(torch.all(gene_ids[500:1000] == 0))    # Gene 1 only
        self.assertTrue(torch.all(gene_ids[1000:1500] == 1))   # Gene 2 overwrites (overlapping region)
        self.assertTrue(torch.all(gene_ids[1500:2000] == 1))   # Gene 2 only
    
    def test_sliding_window_annotation_mapping(self):
        """Test annotation mapping in sliding windows."""
        test_sequence = "A" * 20000  # 20kb sequence
        tsv_rows = [
            ["test_seq", "gene1", 5000, 15000, 5000, 7000, "+"],   # Exon 1
            ["test_seq", "gene1", 5000, 15000, 10000, 12000, "+"], # Exon 2
            ["test_seq", "gene1", 5000, 15000, 13000, 15000, "+"]  # Exon 3
        ]
        tsv_path = self._create_test_tsv(tsv_rows)
        
        annotations = load_tsv_annotations(tsv_path)
        dataset = DNADataset(
            sequences=[test_sequence],
            annotations=annotations,
            max_length=DEFAULT_WINDOW_SIZE,
            use_sliding_windows=True,
            window_size=DEFAULT_WINDOW_SIZE,
            stride=DEFAULT_STRIDE,
            min_gene_coverage=0.3  # Lower threshold for testing
        )
        
        # Should create multiple windows
        self.assertGreater(len(dataset), 1)
        
        # Check that gene appears in multiple windows
        gene_found_in_windows = 0
        for i in range(len(dataset)):
            sample = dataset[i]
            gene_ids = sample['targets']['gene_ids']
            if torch.any(gene_ids != UNKNOWN_GENE_ID):
                gene_found_in_windows += 1
        
        self.assertGreater(gene_found_in_windows, 1, "Gene should appear in multiple windows")
    
    def test_gene_coverage_filtering(self):
        """Test that genes with insufficient coverage are filtered."""
        test_sequence = "A" * 5000  # Shorter sequence
        tsv_rows = [
            ["test_seq", "gene1", 4000, 8000, 4000, 8000, "+"]  # Gene extends beyond sequence
        ]
        tsv_path = self._create_test_tsv(tsv_rows)
        
        annotations = load_tsv_annotations(tsv_path)
        
        # Test with sliding windows to trigger the coverage filtering logic
        dataset = DNADataset(
            sequences=[test_sequence],
            annotations=annotations,
            max_length=5000,
            use_sliding_windows=True,
            window_size=3000,
            stride=1500,
            min_gene_coverage=0.8  # High threshold
        )
        
        # Check if any windows contain the gene
        gene_found = False
        for i in range(len(dataset)):
            sample = dataset[i]
            gene_ids = sample['targets']['gene_ids']
            if torch.any(gene_ids != UNKNOWN_GENE_ID):
                gene_found = True
                break
        
        # Gene should be filtered due to insufficient coverage in windows
        self.assertFalse(gene_found, "Gene should be filtered due to insufficient coverage")
    
    def test_window_metadata_tracking(self):
        """Test that window metadata is correctly tracked."""
        test_sequences = ["A" * 15000, "T" * 8000]  # One long, one short
        annotations = [{}] * len(test_sequences)  # Empty annotations
        
        dataset = DNADataset(
            sequences=test_sequences,
            annotations=annotations,
            max_length=DEFAULT_WINDOW_SIZE,
            use_sliding_windows=True,
            window_size=DEFAULT_WINDOW_SIZE,
            stride=DEFAULT_STRIDE
        )
        
        # First sequence should create multiple windows, second should create one
        total_windows = len(dataset)
        self.assertGreater(total_windows, 2)
        
        # Check metadata
        metadata = dataset.window_metadata
        self.assertEqual(len(metadata), total_windows)
        
        # First sequence should have multiple windows
        seq0_windows = [m for m in metadata if m['original_idx'] == 0]
        seq1_windows = [m for m in metadata if m['original_idx'] == 1]
        
        self.assertGreater(len(seq0_windows), 1)  # Multiple windows for long sequence
        self.assertEqual(len(seq1_windows), 1)    # Single window for short sequence
    
    def test_tsv_header_validation(self):
        """Test TSV header validation."""
        # Create TSV with wrong header
        tsv_path = os.path.join(self.temp_dir, "invalid_header.tsv")
        with open(tsv_path, 'w', newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(['wrong', 'header', 'columns'])
            writer.writerow(['test_seq', 'gene1', '1000', '2000', '1000', '2000', '+'])
        
        # Should raise ValueError for invalid header
        with self.assertRaises(ValueError):
            load_tsv_annotations(tsv_path)


if __name__ == '__main__':
    unittest.main()

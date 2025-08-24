#!/usr/bin/env python3
"""
Test micro-exons (very short exons) to ensure the system handles them correctly.

This test validates that exons as short as 1bp don't break the tensor operations
or data processing pipeline.
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
from utils.constants import TSV_COLUMNS, ExonIntronClass, UNKNOWN_GENE_ID


class TestMicroExons(unittest.TestCase):
    """Test cases for very short exons."""
    
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
    
    def test_1bp_exon(self):
        """Test that 1bp exons work correctly."""
        test_sequence = "A" * 2000
        tsv_rows = [
            ["test_seq", "gene1", 500, 1500, 500, 501, "+"],   # 1bp exon
            ["test_seq", "gene1", 500, 1500, 1000, 1500, "+"] # Normal exon
        ]
        tsv_path = self._create_test_tsv(tsv_rows)
        
        annotations = load_tsv_annotations(tsv_path)
        dataset = DNADataset(
            sequences=[test_sequence],
            annotations=annotations,
            max_length=2000,
            use_sliding_windows=False
        )
        
        sample = dataset[0]
        targets = sample['targets']
        exon_intron = targets['exon_intron']
        
        # Check that 1bp exon is properly labeled
        self.assertEqual(exon_intron[500].item(), ExonIntronClass.EXON)  # 1bp exon
        self.assertTrue(torch.all(exon_intron[1000:1500] == ExonIntronClass.EXON))  # Normal exon
        
        # Check that intron between exons is labeled correctly
        self.assertTrue(torch.all(exon_intron[501:1000] == ExonIntronClass.INTRON))
    
    def test_2bp_exon(self):
        """Test that 2bp exons work correctly."""
        test_sequence = "A" * 1000
        tsv_rows = [
            ["test_seq", "gene1", 100, 900, 100, 102, "+"],   # 2bp exon
            ["test_seq", "gene1", 100, 900, 200, 202, "+"],   # 2bp exon
            ["test_seq", "gene1", 100, 900, 800, 900, "+"]    # Normal exon
        ]
        tsv_path = self._create_test_tsv(tsv_rows)
        
        annotations = load_tsv_annotations(tsv_path)
        dataset = DNADataset(
            sequences=[test_sequence],
            annotations=annotations,
            max_length=1000,
            use_sliding_windows=False
        )
        
        sample = dataset[0]
        targets = sample['targets']
        exon_intron = targets['exon_intron']
        
        # Check that all exons are properly labeled
        self.assertTrue(torch.all(exon_intron[100:102] == ExonIntronClass.EXON))  # 2bp exon 1
        self.assertTrue(torch.all(exon_intron[200:202] == ExonIntronClass.EXON))  # 2bp exon 2  
        self.assertTrue(torch.all(exon_intron[800:900] == ExonIntronClass.EXON))  # Normal exon
        
        # Check introns
        self.assertTrue(torch.all(exon_intron[102:200] == ExonIntronClass.INTRON))  # Intron 1
        self.assertTrue(torch.all(exon_intron[202:800] == ExonIntronClass.INTRON))  # Intron 2
    
    def test_mixed_exon_sizes(self):
        """Test gene with mix of very short and normal exons."""
        test_sequence = "A" * 3000
        tsv_rows = [
            ["test_seq", "gene1", 500, 2500, 500, 501, "+"],    # 1bp exon
            ["test_seq", "gene1", 500, 2500, 600, 603, "+"],    # 3bp exon
            ["test_seq", "gene1", 500, 2500, 700, 706, "+"],    # 6bp exon
            ["test_seq", "gene1", 500, 2500, 1000, 1200, "+"],  # 200bp exon
            ["test_seq", "gene1", 500, 2500, 2000, 2500, "+"]   # 500bp exon
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
        exon_intron = targets['exon_intron']
        gene_ids = targets['gene_ids']
        
        # All exon regions should be labeled correctly regardless of size
        self.assertEqual(exon_intron[500].item(), ExonIntronClass.EXON)     # 1bp
        self.assertTrue(torch.all(exon_intron[600:603] == ExonIntronClass.EXON))   # 3bp
        self.assertTrue(torch.all(exon_intron[700:706] == ExonIntronClass.EXON))   # 6bp
        self.assertTrue(torch.all(exon_intron[1000:1200] == ExonIntronClass.EXON)) # 200bp
        self.assertTrue(torch.all(exon_intron[2000:2500] == ExonIntronClass.EXON)) # 500bp
        
        # Gene ID should be consistent across all exons
        self.assertTrue(torch.all(gene_ids[500:2500] == 0))
        
        # Introns should be labeled correctly
        self.assertTrue(torch.all(exon_intron[501:600] == ExonIntronClass.INTRON))
        self.assertTrue(torch.all(exon_intron[603:700] == ExonIntronClass.INTRON))
        self.assertTrue(torch.all(exon_intron[706:1000] == ExonIntronClass.INTRON))
        self.assertTrue(torch.all(exon_intron[1200:2000] == ExonIntronClass.INTRON))
    
    def test_edge_case_0bp_exon(self):
        """Test edge case where exon_start == exon_end (0bp)."""
        test_sequence = "A" * 1000
        tsv_rows = [
            ["test_seq", "gene1", 100, 500, 100, 100, "+"],   # 0bp "exon" (edge case)
            ["test_seq", "gene1", 100, 500, 200, 500, "+"]    # Normal exon
        ]
        tsv_path = self._create_test_tsv(tsv_rows)
        
        annotations = load_tsv_annotations(tsv_path)
        dataset = DNADataset(
            sequences=[test_sequence],
            annotations=annotations,
            max_length=1000,
            use_sliding_windows=False
        )
        
        sample = dataset[0]
        targets = sample['targets']
        exon_intron = targets['exon_intron']
        
        # 0bp exon should not affect tensor (empty slice)
        # tensor[100:100] is valid but empty
        self.assertTrue(torch.all(exon_intron[200:500] == ExonIntronClass.EXON))  # Normal exon works
        
        # Gene should still be valid
        gene_ids = targets['gene_ids']
        self.assertTrue(torch.all(gene_ids[100:500] == 0))


if __name__ == '__main__':
    unittest.main()

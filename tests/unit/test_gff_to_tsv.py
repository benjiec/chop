#!/usr/bin/env python3
"""
Regression tests for GFF to TSV conversion.

Tests various GFF formats and edge cases to ensure robust parsing
of gene annotations into the standardized TSV format.
"""

import unittest
import tempfile
import os
import sys
from pathlib import Path
import csv

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from scripts.gff_to_tsv import GFFToTSVConverter
from utils.constants import TSV_COLUMNS


class TestGFFToTSV(unittest.TestCase):
    """Test cases for GFF to TSV conversion."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.converter = GFFToTSVConverter(validate_structure=True, verbose=False)
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def _create_test_gff(self, content: str) -> str:
        """Create a temporary GFF file with given content."""
        gff_path = os.path.join(self.temp_dir, "test.gff")
        with open(gff_path, 'w') as f:
            f.write(content)
        return gff_path
    
    def _read_tsv_output(self, tsv_path: str) -> list:
        """Read TSV output and return as list of dictionaries."""
        with open(tsv_path, 'r', newline='') as f:
            reader = csv.DictReader(f, delimiter='\t')
            return list(reader)
    
    def test_single_exon_gene(self):
        """Test parsing of single-exon gene."""
        gff_content = """##gff-version 3
scaffold1	maker	CDS	1000	1500	.	+	0	Parent=mRNA1;locus_tag=gene1
"""
        gff_path = self._create_test_gff(gff_content)
        tsv_path = os.path.join(self.temp_dir, "output.tsv")
        
        self.converter.convert(gff_path, tsv_path)
        
        # Verify output
        rows = self._read_tsv_output(tsv_path)
        self.assertEqual(len(rows), 1)
        
        row = rows[0]
        self.assertEqual(row['sequence_id'], "scaffold1")  # From GFF first column
        self.assertEqual(row['gene_id'], "mRNA1")
        self.assertEqual(int(row['gene_start']), 999)  # 0-based
        self.assertEqual(int(row['gene_end']), 1500)
        self.assertEqual(int(row['exon_start']), 999)
        self.assertEqual(int(row['exon_end']), 1500)
        self.assertEqual(row['strand'], "+")
    
    def test_multi_exon_gene(self):
        """Test parsing of multi-exon gene."""
        gff_content = """##gff-version 3
scaffold1	maker	CDS	1000	1200	.	+	0	Parent=mRNA1;locus_tag=gene1
scaffold1	maker	CDS	1500	1700	.	+	0	Parent=mRNA1;locus_tag=gene1
scaffold1	maker	CDS	2000	2300	.	+	0	Parent=mRNA1;locus_tag=gene1
"""
        gff_path = self._create_test_gff(gff_content)
        tsv_path = os.path.join(self.temp_dir, "output.tsv")
        
        self.converter.convert(gff_path, tsv_path)
        
        # Verify output
        rows = self._read_tsv_output(tsv_path)
        self.assertEqual(len(rows), 3)  # One row per exon
        
        # Check gene boundaries are consistent
        gene_starts = [int(row['gene_start']) for row in rows]
        gene_ends = [int(row['gene_end']) for row in rows]
        self.assertTrue(all(gs == 999 for gs in gene_starts))  # All should have same gene start
        self.assertTrue(all(ge == 2300 for ge in gene_ends))   # All should have same gene end
        
        # Check exon coordinates
        exon_coords = [(int(row['exon_start']), int(row['exon_end'])) for row in rows]
        expected_coords = [(999, 1200), (1499, 1700), (1999, 2300)]
        self.assertEqual(exon_coords, expected_coords)
    
    def test_multiple_genes(self):
        """Test parsing of multiple genes."""
        gff_content = """##gff-version 3
scaffold1	maker	CDS	1000	1500	.	+	0	Parent=mRNA1;locus_tag=gene1
scaffold1	maker	CDS	2000	2500	.	-	0	Parent=mRNA2;locus_tag=gene2
scaffold1	maker	CDS	3000	3200	.	+	0	Parent=mRNA2;locus_tag=gene2
"""
        gff_path = self._create_test_gff(gff_content)
        tsv_path = os.path.join(self.temp_dir, "output.tsv")
        
        self.converter.convert(gff_path, tsv_path)
        
        # Verify output
        rows = self._read_tsv_output(tsv_path)
        self.assertEqual(len(rows), 3)
        
        # Group by gene_id
        gene1_rows = [row for row in rows if row['gene_id'] == 'mRNA1']
        gene2_rows = [row for row in rows if row['gene_id'] == 'mRNA2']
        
        self.assertEqual(len(gene1_rows), 1)  # Single exon
        self.assertEqual(len(gene2_rows), 2)  # Two exons
        
        # Check strands
        self.assertEqual(gene1_rows[0]['strand'], '+')
        self.assertTrue(all(row['strand'] == '-' for row in gene2_rows))
    
    def test_malformed_gff_entries(self):
        """Test handling of malformed GFF entries."""
        gff_content = """##gff-version 3
scaffold1	maker	CDS	1000	1500	.	+	0	Parent=mRNA1;locus_tag=gene1
scaffold1	maker	CDS	invalid	2500	.	-	0	Parent=mRNA2;locus_tag=gene2
scaffold1	maker	CDS	3000	3200	.	+	0	Parent=mRNA3;locus_tag=gene3
"""
        gff_path = self._create_test_gff(gff_content)
        tsv_path = os.path.join(self.temp_dir, "output.tsv")
        
        # Should not raise exception, but should skip invalid entry
        self.converter.convert(gff_path, tsv_path)
        
        rows = self._read_tsv_output(tsv_path)
        self.assertEqual(len(rows), 1)  # Only valid entries (mRNA3 gets filtered for being too short)
        
        gene_ids = [row['gene_id'] for row in rows]
        self.assertIn('mRNA1', gene_ids)
        self.assertNotIn('mRNA2', gene_ids)  # Invalid entry should be skipped
        # mRNA3 gets filtered by validation for being too short (200bp < 300bp minimum)
    
    def test_missing_parent_id(self):
        """Test handling of CDS entries without Parent ID."""
        gff_content = """##gff-version 3
scaffold1	maker	CDS	1000	1500	.	+	0	locus_tag=gene1
scaffold1	maker	CDS	2000	2500	.	-	0	ID=cds2
"""
        gff_path = self._create_test_gff(gff_content)
        tsv_path = os.path.join(self.temp_dir, "output.tsv")
        
        self.converter.convert(gff_path, tsv_path)
        
        rows = self._read_tsv_output(tsv_path)
        self.assertEqual(len(rows), 2)
        
        # Should use locus_tag as gene_id when Parent is missing
        gene_ids = [row['gene_id'] for row in rows]
        self.assertIn('gene1', gene_ids)
        self.assertTrue(any('unknown_gene' in gid for gid in gene_ids))
    
    def test_overlapping_genes(self):
        """Test handling of overlapping genes."""
        gff_content = """##gff-version 3
scaffold1	maker	CDS	1000	2000	.	+	0	Parent=mRNA1;locus_tag=gene1
scaffold1	maker	CDS	1500	2500	.	-	0	Parent=mRNA2;locus_tag=gene2
"""
        gff_path = self._create_test_gff(gff_content)
        tsv_path = os.path.join(self.temp_dir, "output.tsv")
        
        self.converter.convert(gff_path, tsv_path)
        
        rows = self._read_tsv_output(tsv_path)
        self.assertEqual(len(rows), 2)
        
        # Both genes should be present with correct coordinates
        gene1_row = next(row for row in rows if row['gene_id'] == 'mRNA1')
        gene2_row = next(row for row in rows if row['gene_id'] == 'mRNA2')
        
        self.assertEqual(int(gene1_row['gene_start']), 999)
        self.assertEqual(int(gene1_row['gene_end']), 2000)
        self.assertEqual(int(gene2_row['gene_start']), 1499)
        self.assertEqual(int(gene2_row['gene_end']), 2500)
    
    def test_validation_filters(self):
        """Test that validation correctly filters invalid genes."""
        gff_content = """##gff-version 3
scaffold1	maker	CDS	1000	1050	.	+	0	Parent=mRNA1;locus_tag=gene1
scaffold1	maker	CDS	2000	2500	.	+	0	Parent=mRNA2;locus_tag=gene2
"""
        gff_path = self._create_test_gff(gff_content)
        tsv_path = os.path.join(self.temp_dir, "output.tsv")
        
        self.converter.convert(gff_path, tsv_path)
        
        rows = self._read_tsv_output(tsv_path)
        # Gene1 should be filtered (too short: 50bp < 300bp minimum)
        # Gene2 should pass (500bp >= 300bp minimum)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['gene_id'], 'mRNA2')


if __name__ == '__main__':
    unittest.main()

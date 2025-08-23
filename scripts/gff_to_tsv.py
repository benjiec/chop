#!/usr/bin/env python3
"""
GFF to TSV Converter

Converts GFF annotation files to intermediate TSV format for gene prediction training.
The TSV format provides a clean, standardized representation of gene structures.

Output TSV columns: accession_id, gene_id, gene_start, gene_end, exon_start, exon_end, strand
"""

import argparse
import sys
import os
from pathlib import Path
from typing import Dict, List, Tuple, Set
import csv

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))
from utils.constants import (
    TSV_COLUMNS, TSV_HEADER, MIN_GENE_LENGTH, MIN_EXON_LENGTH, 
    MAX_GENE_EXONS, MAX_INTRON_LENGTH
)


class GFFToTSVConverter:
    """Converts GFF files to standardized TSV format for gene prediction."""
    
    def __init__(self, validate_structure: bool = True, verbose: bool = False):
        self.validate_structure = validate_structure
        self.verbose = verbose
        self.stats = {
            'total_cds_entries': 0,
            'total_genes': 0,
            'single_exon_genes': 0,
            'multi_exon_genes': 0,
            'invalid_genes': 0,
            'total_exons': 0
        }
        
    def convert(self, gff_path: str, output_path: str) -> None:
        """Convert GFF file to TSV format."""
        
        if not os.path.exists(gff_path):
            raise FileNotFoundError(f"GFF file not found: {gff_path}")
            
        if self.verbose:
            print(f"Converting {gff_path} to {output_path}")
            
        # Parse GFF and group CDS entries by gene
        gene_structures = self._parse_gff(gff_path)
        
        # Write to TSV format
        self._write_tsv(gene_structures, output_path)
        
        # Print statistics
        self._print_stats()
        
    def _parse_gff(self, gff_path: str) -> List[Dict]:
        """Parse GFF file and group CDS entries by Parent ID (gene)."""
        
        # Group CDS entries by Parent (transcript/mRNA ID)
        transcripts = {}  # parent_id -> list of CDS entries
        
        with open(gff_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                if line.startswith('#'):
                    continue
                    
                parts = line.strip().split('\t')
                if len(parts) < 9:
                    continue
                    
                if parts[2] == 'CDS':
                    self.stats['total_cds_entries'] += 1
                    
                    try:
                        sequence_id = parts[0]  # Use sequence ID from GFF first column
                        start = int(parts[3]) - 1  # Convert to 0-based
                        end = int(parts[4])
                        strand = parts[6]
                        attributes = parts[8]
                        
                        # Extract Parent ID and other identifiers
                        parent_id, locus_tag = self._extract_gene_identifiers(attributes)
                        gene_id = parent_id or locus_tag or f"unknown_gene_{line_num}"
                        
                        if gene_id not in transcripts:
                            transcripts[gene_id] = []
                            
                        transcripts[gene_id].append({
                            'start': start,
                            'end': end,
                            'strand': strand,
                            'sequence_id': sequence_id  # Use sequence ID from GFF
                        })
                        
                    except (ValueError, IndexError) as e:
                        if self.verbose:
                            print(f"Warning: Skipping malformed line {line_num}: {e}")
                        continue
                        
        if self.verbose:
            print(f"Found {len(transcripts)} genes/transcripts from {self.stats['total_cds_entries']} CDS entries")
            
        # Convert to gene structures
        return self._build_gene_structures(transcripts)
        
    def _extract_gene_identifiers(self, attributes: str) -> Tuple[str, str]:
        """Extract Parent ID and locus_tag from GFF attributes."""
        parent_id = None
        locus_tag = None
        
        for attr in attributes.split(';'):
            attr = attr.strip()
            if attr.startswith('Parent='):
                parent_id = attr.split('Parent=')[1].split(',')[0]
            elif attr.startswith('locus_tag='):
                locus_tag = attr.split('locus_tag=')[1]
                
        return parent_id, locus_tag
        
    def _build_gene_structures(self, transcripts: Dict[str, List[Dict]]) -> List[Dict]:
        """Build gene structures from grouped CDS entries."""
        gene_structures = []
        
        for gene_id, cds_list in transcripts.items():
            if not cds_list:
                continue
                
            # Sort CDS by position
            cds_list.sort(key=lambda x: x['start'])
            
            # Calculate gene boundaries
            gene_start = min(cds['start'] for cds in cds_list)
            gene_end = max(cds['end'] for cds in cds_list)
            strand = cds_list[0]['strand']
            sequence_id = cds_list[0]['sequence_id']
            
            # Validate gene structure if enabled
            if self.validate_structure:
                if not self._validate_gene_structure(gene_id, gene_start, gene_end, cds_list):
                    self.stats['invalid_genes'] += 1
                    continue
                    
            # Create gene structure
            gene_structure = {
                'sequence_id': sequence_id,
                'gene_id': gene_id,
                'gene_start': gene_start,
                'gene_end': gene_end,
                'strand': strand,
                'exons': cds_list  # Each CDS is an exon
            }
            
            gene_structures.append(gene_structure)
            self.stats['total_genes'] += 1
            self.stats['total_exons'] += len(cds_list)
            
            if len(cds_list) == 1:
                self.stats['single_exon_genes'] += 1
            else:
                self.stats['multi_exon_genes'] += 1
                
        return gene_structures
        
    def _validate_gene_structure(self, gene_id: str, gene_start: int, gene_end: int, 
                                cds_list: List[Dict]) -> bool:
        """Validate gene structure for biological plausibility."""
        
        # Check minimum gene length
        gene_length = gene_end - gene_start
        if gene_length < MIN_GENE_LENGTH:
            if self.verbose:
                print(f"Warning: Gene {gene_id} too short ({gene_length} bp)")
            return False
            
        # Check maximum number of exons
        if len(cds_list) > MAX_GENE_EXONS:
            if self.verbose:
                print(f"Warning: Gene {gene_id} has too many exons ({len(cds_list)})")
            return False
            
        # Check exon lengths and overlaps
        for i, exon in enumerate(cds_list):
            exon_length = exon['end'] - exon['start']
            if exon_length < MIN_EXON_LENGTH:
                if self.verbose:
                    print(f"Warning: Gene {gene_id} exon {i} too short ({exon_length} bp)")
                return False
                
            # Check for overlapping exons (should not happen)
            if i > 0:
                prev_exon = cds_list[i-1]
                if exon['start'] < prev_exon['end']:
                    if self.verbose:
                        print(f"Warning: Gene {gene_id} has overlapping exons")
                    return False
                    
                # Check intron length
                intron_length = exon['start'] - prev_exon['end']
                if intron_length > MAX_INTRON_LENGTH:
                    if self.verbose:
                        print(f"Warning: Gene {gene_id} has very long intron ({intron_length} bp)")
                    # Don't reject, just warn - algae may have long introns
                    
        return True
        
    def _write_tsv(self, gene_structures: List[Dict], output_path: str) -> None:
        """Write gene structures to TSV format."""
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f, delimiter='\t')
            
            # Write header
            writer.writerow(TSV_COLUMNS)
            
            # Write gene data
            for gene in gene_structures:
                for exon in gene['exons']:
                    row = [
                        gene['sequence_id'],  # Use sequence ID instead of accession ID
                        gene['gene_id'],
                        gene['gene_start'],
                        gene['gene_end'],
                        exon['start'],
                        exon['end'],
                        gene['strand']
                    ]
                    writer.writerow(row)
                    
        if self.verbose:
            print(f"TSV file written to: {output_path}")
            
    def _print_stats(self) -> None:
        """Print conversion statistics."""
        print("\n=== Conversion Statistics ===")
        print(f"Total CDS entries processed: {self.stats['total_cds_entries']:,}")
        print(f"Total genes created: {self.stats['total_genes']:,}")
        print(f"  - Single-exon genes: {self.stats['single_exon_genes']:,}")
        print(f"  - Multi-exon genes: {self.stats['multi_exon_genes']:,}")
        print(f"Total exons: {self.stats['total_exons']:,}")
        print(f"Invalid genes filtered: {self.stats['invalid_genes']:,}")
        
        if self.stats['total_genes'] > 0:
            avg_exons = self.stats['total_exons'] / self.stats['total_genes']
            multi_exon_pct = (self.stats['multi_exon_genes'] / self.stats['total_genes']) * 100
            print(f"Average exons per gene: {avg_exons:.1f}")
            print(f"Multi-exon gene percentage: {multi_exon_pct:.1f}%")


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description='Convert GFF annotation files to TSV format for gene prediction training'
    )
    parser.add_argument('input_gff', help='Input GFF file path')
    parser.add_argument('output_tsv', help='Output TSV file path')
    parser.add_argument('--no-validation', action='store_true', 
                       help='Disable gene structure validation')
    parser.add_argument('--verbose', '-v', action='store_true', 
                       help='Enable verbose output')
    
    args = parser.parse_args()
    
    # Create converter
    converter = GFFToTSVConverter(
        validate_structure=not args.no_validation,
        verbose=args.verbose
    )
    
    try:
        # Convert file
        converter.convert(args.input_gff, args.output_tsv)
        print(f"\nConversion completed successfully!")
        
    except Exception as e:
        print(f"Error during conversion: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()

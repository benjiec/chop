#!/usr/bin/env python3
"""
Comprehensive gene data preprocessing script.

This script combines GFF→TSV conversion, gene context extraction, and strand normalization
into a single preprocessing step. It produces normalized gene context sequences and 
annotations that are ready for training.

Input:
  - .fna file (genomic sequences)  
  - .gff file (gene annotations)

Output:
  - .fna file with gene contexts (gene_id as sequence names, strand normalized)
  - .tsv file with annotations adjusted to gene context coordinates (all + strand)

Features:
  - Extracts gene ± flanking_bp contexts
  - Normalizes all genes to + strand (5'→3' orientation)
  - Adjusts all coordinates to gene context coordinate system
  - Validates start/stop codons after normalization
  - Filters out invalid genes
"""

import argparse
import sys
from pathlib import Path
import os
from collections import defaultdict
from typing import Dict, List, Tuple, Set
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
import csv

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.sequences import reverse_complement, validate_start_stop_codons_from_exons


def parse_gff_file(gff_path: str) -> List[Dict]:
    """Parse GFF file and return list of gene dictionaries."""
    from collections import defaultdict
    
    # Group CDS entries by Parent ID (gene)
    genes_data = defaultdict(lambda: {
        'sequence_id': None,
        'gene_start': float('inf'),
        'gene_end': 0,
        'strand': '+',
        'exons': []
    })
    
    with open(gff_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            if line.startswith('#'):
                continue
                
            parts = line.strip().split('\t')
            if len(parts) < 9:
                continue
                
            if parts[2] == 'CDS':
                try:
                    sequence_id = parts[0]
                    start = int(parts[3]) - 1  # Convert to 0-based
                    end = int(parts[4])        # Convert to 0-based exclusive (Python slice style)
                    strand = parts[6]
                    attributes = parts[8]
                    
                    # Extract Parent ID
                    parent_id = None
                    for attr in attributes.split(';'):
                        if attr.startswith('Parent='):
                            parent_id = attr.split('=', 1)[1]
                            break
                    
                    if not parent_id:
                        parent_id = f"gene_{line_num}"
                    
                    # Update gene boundaries
                    genes_data[parent_id]['sequence_id'] = sequence_id
                    genes_data[parent_id]['gene_start'] = min(genes_data[parent_id]['gene_start'], start)
                    genes_data[parent_id]['gene_end'] = max(genes_data[parent_id]['gene_end'], end)
                    genes_data[parent_id]['strand'] = strand
                    
                    # Add exon
                    genes_data[parent_id]['exons'].append({
                        'start': start,
                        'end': end
                    })
                    
                except (ValueError, IndexError) as e:
                    print(f"Warning: Skipping malformed line {line_num}: {e}")
                    continue
    
    # Convert to list of gene dictionaries
    processed_genes = []
    for gene_id, gene_data in genes_data.items():
        if gene_data['sequence_id'] is None:
            continue  # Skip genes with no valid CDS entries
            
        gene = {
            'sequence_id': gene_data['sequence_id'],
            'gene_id': gene_id,
            'start': gene_data['gene_start'],
            'end': gene_data['gene_end'],
            'strand': gene_data['strand'],
            'exons': sorted(gene_data['exons'], key=lambda x: x['start'])
        }
        processed_genes.append(gene)
    
    print(f"Parsed {len(processed_genes)} genes from GFF file")
    return processed_genes


def extract_gene_contexts_with_normalization(
    genomic_sequences: Dict[str, str],
    genes_by_sequence: Dict[str, List[Dict]],
    flanking_bp: int = 2000
) -> Tuple[List[SeqRecord], List[Dict]]:
    """
    Extract gene contexts with strand normalization.
    
    Args:
        genomic_sequences: Dict mapping sequence_id -> sequence
        genes_by_sequence: Dict mapping sequence_id -> list of gene dicts
        flanking_bp: Number of flanking base pairs around each gene
        
    Returns:
        Tuple of (normalized_sequences, normalized_annotations)
    """
    normalized_sequences = []
    normalized_annotations = []
    
    total_genes = 0
    processed_genes = 0
    invalid_genes = 0
    
    for sequence_id, genes in genes_by_sequence.items():
        if sequence_id not in genomic_sequences:
            print(f"Warning: Sequence {sequence_id} not found in genomic sequences")
            continue
            
        genomic_sequence = genomic_sequences[sequence_id]
        total_genes += len(genes)
        
        for gene in genes:
            gene_id = gene['gene_id']
            gene_start = gene['start']  # 0-based coordinates from GFF parsing
            gene_end = gene['end']
            strand = gene['strand']
            exons = gene.get('exons', [])
            
            # Calculate context boundaries
            context_start = max(0, gene_start - flanking_bp)
            context_end = min(len(genomic_sequence), gene_end + flanking_bp)
            
            # Extract context sequence
            context_sequence = genomic_sequence[context_start:context_end]
            
            # Adjust gene and exon coordinates to context coordinate system
            adjusted_gene_start = gene_start - context_start
            adjusted_gene_end = gene_end - context_start
            
            adjusted_exons = []
            for exon in exons:
                adjusted_exon = {
                    'start': exon['start'] - context_start,
                    'end': exon['end'] - context_start
                }
                adjusted_exons.append(adjusted_exon)
            
            # Handle strand normalization
            if strand == '-':
                # Reverse complement the sequence
                context_sequence = reverse_complement(context_sequence)
                
                # Flip coordinates for reverse complement
                seq_length = len(context_sequence)
                
                # Gene coordinates
                new_gene_start = seq_length - adjusted_gene_end
                new_gene_end = seq_length - adjusted_gene_start
                adjusted_gene_start = min(new_gene_start, new_gene_end)
                adjusted_gene_end = max(new_gene_start, new_gene_end)
                
                # Exon coordinates
                flipped_exons = []
                for exon in adjusted_exons:
                    new_exon_start = seq_length - exon['end']
                    new_exon_end = seq_length - exon['start']
                    flipped_exons.append({
                        'start': min(new_exon_start, new_exon_end),
                        'end': max(new_exon_start, new_exon_end)
                    })
                
                # Sort exons by start position (they may be in reverse order after flipping)
                adjusted_exons = sorted(flipped_exons, key=lambda x: x['start'])
                
                # Update strand to + (normalized)
                strand = '+'
            
            # Validate start/stop codons after normalization
            if not validate_start_stop_codons_from_exons(context_sequence, adjusted_exons, strand):
                print(f"Warning: Gene {gene_id} has invalid start/stop codons after normalization")
                invalid_genes += 1
                continue
            
            # Create normalized sequence record
            # Note: Use empty description to ensure FASTA header only contains the gene_id
            seq_record = SeqRecord(
                seq=context_sequence,
                id=gene_id,
                description=""  # Empty description ensures header is just ">gene_id"
            )
            normalized_sequences.append(seq_record)
            
            # Create normalized annotation
            normalized_annotation = {
                'sequence_id': gene_id,  # Use gene_id as sequence identifier
                'gene_id': gene_id,
                'gene_start': adjusted_gene_start,
                'gene_end': adjusted_gene_end,
                'strand': strand,  # Always '+' after normalization
                'exons': adjusted_exons
            }
            normalized_annotations.append(normalized_annotation)
            processed_genes += 1
    
    print(f"\n=== Gene Context Extraction Summary ===")
    print(f"Total genes found: {total_genes}")
    print(f"Successfully processed: {processed_genes}")
    print(f"Invalid genes filtered: {invalid_genes}")
    print(f"Success rate: {processed_genes/total_genes*100:.1f}%")
    
    return normalized_sequences, normalized_annotations


def write_normalized_fasta(sequences: List[SeqRecord], output_path: str):
    """Write normalized gene context sequences to FASTA file."""
    with open(output_path, 'w') as f:
        SeqIO.write(sequences, f, "fasta")
    print(f"Wrote {len(sequences)} normalized gene contexts to {output_path}")


def write_normalized_tsv(annotations: List[Dict], output_path: str):
    """Write normalized annotations to TSV file."""
    
    # Define TSV columns
    columns = ['sequence_id', 'gene_id', 'gene_start', 'gene_end', 'exon_start', 'exon_end', 'strand']
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns, delimiter='\t')
        writer.writeheader()
        
        for annotation in annotations:
            gene_id = annotation['gene_id']
            sequence_id = annotation['sequence_id']
            gene_start = annotation['gene_start']
            gene_end = annotation['gene_end']
            strand = annotation['strand']
            exons = annotation['exons']
            
            # Write one row per exon
            for exon in exons:
                row = {
                    'sequence_id': sequence_id,
                    'gene_id': gene_id,
                    'gene_start': gene_start + 1,  # Convert back to 1-based for TSV output
                    'gene_end': gene_end,          # End is already exclusive
                    'exon_start': exon['start'] + 1,  # Convert back to 1-based for TSV output
                    'exon_end': exon['end'],          # End is already exclusive
                    'strand': strand
                }
                writer.writerow(row)
    
    print(f"Wrote {len(annotations)} gene annotations to {output_path}")


def load_genomic_sequences(fasta_path: str) -> Dict[str, str]:
    """Load genomic sequences from FASTA file."""
    sequences = {}
    
    print(f"Loading genomic sequences from {fasta_path}")
    with open(fasta_path) as f:
        for record in SeqIO.parse(f, "fasta"):
            sequences[record.id] = str(record.seq).upper()
    
    print(f"Loaded {len(sequences)} genomic sequences")
    return sequences


def group_genes_by_sequence(genes: List[Dict]) -> Dict[str, List[Dict]]:
    """Group genes by their sequence ID."""
    genes_by_sequence = defaultdict(list)
    
    for gene in genes:
        sequence_id = gene['sequence_id']
        genes_by_sequence[sequence_id].append(gene)
    
    return dict(genes_by_sequence)


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess gene data: GFF→TSV + gene context extraction + strand normalization"
    )
    parser.add_argument('--fasta', required=True, help='Input genomic FASTA file')
    parser.add_argument('--gff', required=True, help='Input GFF annotation file')
    parser.add_argument('--output-fasta', required=True, help='Output normalized gene context FASTA file')
    parser.add_argument('--output-tsv', required=True, help='Output normalized annotation TSV file')
    parser.add_argument('--flanking-bp', type=int, default=2000, 
                       help='Number of flanking base pairs around each gene (default: 2000)')
    
    args = parser.parse_args()
    
    # Validate input files
    if not os.path.exists(args.fasta):
        print(f"Error: FASTA file not found: {args.fasta}")
        sys.exit(1)
    
    if not os.path.exists(args.gff):
        print(f"Error: GFF file not found: {args.gff}")
        sys.exit(1)
    
    # Step 1: Parse GFF file
    print(f"\n=== Step 1: Parsing GFF file ===")
    genes = parse_gff_file(args.gff)
    
    if not genes:
        print("Error: No genes found in GFF file")
        sys.exit(1)
    
    # Step 2: Load genomic sequences
    print(f"\n=== Step 2: Loading genomic sequences ===")
    genomic_sequences = load_genomic_sequences(args.fasta)
    
    # Step 3: Group genes by sequence
    print(f"\n=== Step 3: Grouping genes by sequence ===")
    genes_by_sequence = group_genes_by_sequence(genes)
    
    # Filter to only sequences that have genes
    sequences_with_genes = set(genes_by_sequence.keys())
    available_sequences = set(genomic_sequences.keys())
    
    missing_sequences = sequences_with_genes - available_sequences
    if missing_sequences:
        print(f"Warning: {len(missing_sequences)} gene sequences not found in FASTA:")
        for seq_id in sorted(missing_sequences):
            print(f"  - {seq_id}")
    
    valid_sequences = sequences_with_genes & available_sequences
    print(f"Found {len(valid_sequences)} sequences with genes in FASTA file")
    
    # Step 4: Extract gene contexts with strand normalization
    print(f"\n=== Step 4: Extracting gene contexts with strand normalization ===")
    normalized_sequences, normalized_annotations = extract_gene_contexts_with_normalization(
        genomic_sequences, genes_by_sequence, args.flanking_bp
    )
    
    if not normalized_sequences:
        print("Error: No valid gene contexts extracted")
        sys.exit(1)
    
    # Step 5: Write output files
    print(f"\n=== Step 5: Writing output files ===")
    write_normalized_fasta(normalized_sequences, args.output_fasta)
    write_normalized_tsv(normalized_annotations, args.output_tsv)
    
    print(f"\n=== Preprocessing Complete ===")
    print(f"Generated {len(normalized_sequences)} normalized gene contexts")
    print(f"Output files:")
    print(f"  - FASTA: {args.output_fasta}")
    print(f"  - TSV: {args.output_tsv}")


if __name__ == '__main__':
    main()

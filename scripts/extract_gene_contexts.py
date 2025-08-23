#!/usr/bin/env python3
"""
Extract Gene Contexts Script

This script takes a genomic FASTA file and gene annotations TSV file,
and creates a new FASTA file containing genomic contexts around each gene.

Each output sequence contains:
- Gene start - 2000bp to Gene end + 2000bp
- Labeled with gene ID for traceability

Usage:
    python scripts/extract_gene_contexts.py input.fna annotations.tsv output_contexts.fna
    
    Or with custom flanking distance:
    python scripts/extract_gene_contexts.py input.fna annotations.tsv output_contexts.fna --flank 1500
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))
from utils.dna_processor import load_tsv_annotations, load_fasta_sequences_with_ids


def extract_gene_context(sequence: str, gene_start: int, gene_end: int, 
                        flank_size: int = 2000) -> Tuple[str, int, int]:
    """
    Extract genomic context around a gene.
    
    Args:
        sequence: Full genomic sequence
        gene_start: Gene start position (0-based)
        gene_end: Gene end position (0-based, exclusive)
        flank_size: Number of base pairs to include on each side
        
    Returns:
        Tuple of (context_sequence, context_start, context_end)
    """
    seq_length = len(sequence)
    
    # Calculate context boundaries, ensuring we don't go beyond sequence limits
    context_start = max(0, gene_start - flank_size)
    context_end = min(seq_length, gene_end + flank_size)
    
    # Extract context sequence
    context_sequence = sequence[context_start:context_end]
    
    return context_sequence, context_start, context_end


def create_gene_context_record(gene: Dict, context_sequence: str, 
                              context_start: int, context_end: int,
                              original_seq_id: str) -> SeqRecord:
    """
    Create a SeqRecord for a gene context.
    
    Args:
        gene: Gene annotation dictionary
        context_sequence: Extracted genomic context
        context_start: Start position of context in original sequence
        context_end: End position of context in original sequence
        original_seq_id: ID of original sequence
        
    Returns:
        SeqRecord with appropriate ID and description
    """
    # Use gene ID directly as sequence ID - clean header with no description
    gene_id = gene['gene_id']
    
    return SeqRecord(
        Seq(context_sequence),
        id=gene_id,
        description=""  # No description - just gene ID as header
    )


def extract_all_gene_contexts(fasta_path: str, tsv_path: str, 
                             output_path: str, flank_size: int = 2000) -> None:
    """
    Extract genomic contexts for all genes and write to new FASTA file.
    
    Args:
        fasta_path: Path to input genomic FASTA file
        tsv_path: Path to gene annotations TSV file
        output_path: Path for output gene contexts FASTA file
        flank_size: Number of base pairs to include on each side of genes
    """
    print(f"🧬 Extracting gene contexts...")
    print(f"  Input FASTA: {fasta_path}")
    print(f"  Input TSV: {tsv_path}")
    print(f"  Output FASTA: {output_path}")
    print(f"  Flanking size: ±{flank_size} bp")
    print()
    
    # Load genomic sequences
    print("📖 Loading genomic sequences...")
    sequences_with_ids = load_fasta_sequences_with_ids(fasta_path)
    sequence_dict = {seq_id: sequence for seq_id, sequence in sequences_with_ids}
    print(f"  Loaded {len(sequence_dict)} genomic sequences")
    
    # Load gene annotations
    print("📊 Loading gene annotations...")
    annotations = load_tsv_annotations(tsv_path)
    print(f"  Loaded annotations for {len(annotations)} sequences")
    
    # Extract gene contexts
    print("✂️  Extracting gene contexts...")
    gene_contexts = []
    total_genes = 0
    skipped_genes = 0
    total_original_size = 0
    total_context_size = 0
    
    for seq_obj in annotations:
        seq_id = seq_obj['sequence_id']
        
        if seq_id not in sequence_dict:
            print(f"  ⚠️  Warning: Sequence {seq_id} not found in FASTA file")
            continue
            
        genomic_sequence = sequence_dict[seq_id]
        total_original_size += len(genomic_sequence)
        
        genes = seq_obj.get('genes', [])
        for gene in genes:
            total_genes += 1
            
            try:
                # Extract genomic context around this gene
                context_seq, context_start, context_end = extract_gene_context(
                    genomic_sequence, gene['start'], gene['end'], flank_size
                )
                
                # Create SeqRecord for this gene context
                context_record = create_gene_context_record(
                    gene, context_seq, context_start, context_end, seq_id
                )
                
                gene_contexts.append(context_record)
                total_context_size += len(context_seq)
                
                if total_genes % 100 == 0:
                    print(f"  Processed {total_genes} genes...")
                    
            except Exception as e:
                print(f"  ⚠️  Error processing gene {gene.get('gene_id', 'unknown')}: {e}")
                skipped_genes += 1
    
    # Write gene contexts to output FASTA
    print(f"💾 Writing {len(gene_contexts)} gene contexts to {output_path}...")
    with open(output_path, 'w') as output_file:
        SeqIO.write(gene_contexts, output_file, "fasta")
    
    # Calculate size reduction
    size_reduction = (total_original_size - total_context_size) / total_original_size * 100
    avg_context_size = total_context_size / len(gene_contexts) if gene_contexts else 0
    
    # Print summary
    print("\n📈 Extraction Summary:")
    print(f"  Total genes processed: {total_genes}")
    print(f"  Gene contexts created: {len(gene_contexts)}")
    print(f"  Genes skipped (errors): {skipped_genes}")
    print(f"  Original total size: {total_original_size:,} bp ({total_original_size/1e6:.1f} MB)")
    print(f"  Context total size: {total_context_size:,} bp ({total_context_size/1e6:.1f} MB)")
    print(f"  Size reduction: {size_reduction:.1f}%")
    print(f"  Average context size: {avg_context_size:.0f} bp")
    print(f"✅ Extraction complete! Use {output_path} for training.")


def main():
    """Main function to parse arguments and run extraction."""
    parser = argparse.ArgumentParser(
        description="Extract genomic contexts around annotated genes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Extract contexts with default ±2000bp flanking
    python scripts/extract_gene_contexts.py data/genome.fna data/annotations.tsv data/gene_contexts.fna
    
    # Extract contexts with custom ±1500bp flanking
    python scripts/extract_gene_contexts.py data/genome.fna data/annotations.tsv data/gene_contexts.fna --flank 1500
        """
    )
    
    parser.add_argument("fasta", help="Input genomic FASTA file")
    parser.add_argument("tsv", help="Input gene annotations TSV file")
    parser.add_argument("output", help="Output gene contexts FASTA file")
    parser.add_argument("--flank", type=int, default=2000, 
                       help="Flanking size in base pairs (default: 2000)")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose output")
    
    args = parser.parse_args()
    
    # Validate input files exist
    if not Path(args.fasta).exists():
        print(f"❌ Error: FASTA file not found: {args.fasta}")
        sys.exit(1)
        
    if not Path(args.tsv).exists():
        print(f"❌ Error: TSV file not found: {args.tsv}")
        sys.exit(1)
    
    # Create output directory if needed
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        extract_all_gene_contexts(args.fasta, args.tsv, args.output, args.flank)
    except Exception as e:
        print(f"❌ Error during extraction: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Generate synthetic test fixtures for end-to-end testing.

Creates realistic genomic sequences with genes, exons, and introns for testing
the complete preprocessing → training → prediction workflow.
"""

import random
import argparse
from pathlib import Path
from typing import List, Tuple, Dict
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord


def generate_random_dna(length: int) -> str:
    """Generate random DNA sequence of specified length."""
    return ''.join(random.choices('ATGC', k=length))


def generate_codon_sequence(num_codons: int, include_stop: bool = False) -> str:
    """Generate a valid coding sequence with proper codons."""
    # Common codons for realistic gene simulation
    codons = [
        'AAA', 'AAG', 'GAA', 'GAG', 'CAA', 'CAG', 'TTC', 'TTT',  # Common AA codons
        'GCT', 'GCC', 'GCA', 'GCG', 'AGT', 'AGC', 'TCT', 'TCC',  # Common other codons
        'CTT', 'CTC', 'CTA', 'CTG', 'GTT', 'GTC', 'GTA', 'GTG',  # More variety
        'ATT', 'ATC', 'ATA', 'ACT', 'ACC', 'ACA', 'ACG', 'CCT',  # Additional codons
    ]
    
    sequence = ""
    for _ in range(num_codons):
        sequence += random.choice(codons)
    
    if include_stop:
        stop_codons = ['TAA', 'TAG', 'TGA']
        sequence += random.choice(stop_codons)
    
    return sequence


class SyntheticGene:
    def __init__(self, gene_id: str, start: int, strand: str = '+'):
        self.gene_id = gene_id
        self.start = start
        self.strand = strand
        self.exons = []  # List of (start, end) tuples
        self.sequence = ""
        self.end = start
        
    def add_exon(self, exon_length: int, intron_length: int = 0):
        """Add an exon and optionally an intron after it."""
        exon_start = self.end
        exon_end = exon_start + exon_length
        self.exons.append((exon_start, exon_end))
        
        # Generate exon sequence
        if len(self.exons) == 1:
            # First exon must start with ATG
            if exon_length < 3:
                raise ValueError("First exon must be at least 3bp to include start codon")
            exon_seq = 'ATG' + generate_codon_sequence((exon_length - 3) // 3)
            # Pad to exact length if needed
            while len(exon_seq) < exon_length:
                exon_seq += random.choice('ATGC')
        else:
            # Other exons - ensure they maintain reading frame
            # Generate complete codons to maintain frame
            num_codons = exon_length // 3
            exon_seq = generate_codon_sequence(num_codons)
            # Pad to exact length if needed
            while len(exon_seq) < exon_length:
                exon_seq += random.choice('ATGC')
        
        self.sequence += exon_seq
        self.end = exon_end
        
        # Add intron if specified
        if intron_length > 0:
            self.end += intron_length
    
    def finalize_gene(self):
        """Finalize the gene by ensuring it ends with a stop codon."""
        # Make sure the last exon ends with a stop codon
        if len(self.sequence) >= 3:
            # Replace last 3 bases with stop codon
            stop_codons = ['TAA', 'TAG', 'TGA']
            self.sequence = self.sequence[:-3] + random.choice(stop_codons)
    
    def get_gene_length(self):
        """Get total gene length including introns."""
        return self.end - self.start


def create_synthetic_contig(contig_id: str, length: int, num_genes: int) -> Tuple[str, List[SyntheticGene]]:
    """
    Create a synthetic genomic contig with specified number of genes.
    
    Args:
        contig_id: Identifier for the contig
        length: Total length of the contig (7-8kb)
        num_genes: Number of genes to place in the contig
        
    Returns:
        Tuple of (DNA sequence, list of genes)
    """
    # Start with random DNA background
    sequence = list(generate_random_dna(length))
    genes = []
    
    # Calculate approximate space per gene
    space_per_gene = length // (num_genes + 1)  # +1 for spacing
    
    for i in range(num_genes):
        gene_id = f"{contig_id}_gene_{i+1:02d}"
        
        # Position gene with some randomness
        base_start = (i + 1) * space_per_gene
        start_pos = base_start + random.randint(-space_per_gene//4, space_per_gene//4)
        start_pos = max(500, min(start_pos, length - 3000))  # Ensure room for gene
        
        # Create gene with random characteristics
        gene = SyntheticGene(gene_id, start_pos)
        
        # Generate 5-6 exons
        num_exons = random.randint(5, 6)
        max_available_length = length - start_pos - 500  # Leave space at end
        total_gene_length = min(random.randint(2000, 4000), max_available_length)  # 2-4kb as requested
        
        # Distribute length between exons and introns
        exon_lengths = []
        intron_lengths = []
        
        # Generate exon lengths (5-800 bp each)
        for j in range(num_exons):
            if j == 0:  # First exon
                exon_len = random.randint(50, 300)  # Reasonable first exon
            else:
                exon_len = random.randint(5, min(800, total_gene_length // num_exons))  # As requested
            exon_lengths.append(exon_len)
        
        # Calculate remaining space for introns
        total_exon_length = sum(exon_lengths)
        remaining_for_introns = total_gene_length - total_exon_length
        
        # Ensure we have enough space for introns
        min_intron_space = (num_exons - 1) * 50  # Minimum 50bp per intron
        if remaining_for_introns < min_intron_space:
            # Adjust gene length or skip this gene
            total_gene_length = total_exon_length + min_intron_space
            remaining_for_introns = min_intron_space
        
        # Distribute intron lengths
        for j in range(num_exons - 1):  # n-1 introns for n exons
            if j == num_exons - 2:  # Last intron gets remainder
                intron_len = max(50, remaining_for_introns)
            else:
                max_intron = remaining_for_introns // (num_exons - 1 - j)
                if max_intron < 50:
                    intron_len = 50
                else:
                    intron_len = random.randint(50, min(1000, max_intron))
                remaining_for_introns -= intron_len
            intron_lengths.append(intron_len)
        
        # Build the gene
        try:
            for j in range(num_exons):
                intron_len = intron_lengths[j] if j < len(intron_lengths) else 0
                gene.add_exon(exon_lengths[j], intron_len)
            
            gene.finalize_gene()
            
            # Check if gene fits in contig
            if gene.end < length - 500:  # Leave some space at end
                genes.append(gene)
                
                # Place the gene sequence into the contig
                # For now, just mark the positions - we'll handle actual sequence later
                
        except (ValueError, IndexError):
            # Skip genes that don't fit or have issues
            continue
    
    return ''.join(sequence), genes


def write_fasta_file(filename: str, contigs: List[Tuple[str, str]]):
    """Write contigs to FASTA file."""
    records = []
    for contig_id, sequence in contigs:
        record = SeqRecord(Seq(sequence), id=contig_id, description="")
        records.append(record)
    
    with open(filename, 'w') as f:
        SeqIO.write(records, f, "fasta")


def write_gff_file(filename: str, all_genes: Dict[str, List[SyntheticGene]]):
    """Write gene annotations to GFF file."""
    with open(filename, 'w') as f:
        f.write("##gff-version 3\n")
        
        for contig_id, genes in all_genes.items():
            for gene in genes:
                # Write gene feature
                f.write(f"{contig_id}\tsynthetic\tgene\t{gene.start+1}\t{gene.end}\t.\t{gene.strand}\t.\t"
                       f"ID={gene.gene_id};Name={gene.gene_id}\n")
                
                # Write CDS features for each exon
                for i, (exon_start, exon_end) in enumerate(gene.exons):
                    f.write(f"{contig_id}\tsynthetic\tCDS\t{exon_start+1}\t{exon_end}\t.\t{gene.strand}\t0\t"
                           f"ID={gene.gene_id}_CDS_{i+1};Parent={gene.gene_id}\n")


def generate_test_fixture(output_dir: str, num_contigs: int = 3, contig_length: int = 7500):
    """Generate complete test fixture with FASTA and GFF files."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Generate contigs
    contigs = []
    all_genes = {}
    
    for i in range(num_contigs):
        contig_id = f"test_contig_{i+1:02d}"
        num_genes = random.randint(2, 4)  # 2-4 genes per contig
        
        sequence, genes = create_synthetic_contig(contig_id, contig_length, num_genes)
        contigs.append((contig_id, sequence))
        all_genes[contig_id] = genes
    
    # Write files
    fasta_file = output_path / "test_genome.fna"
    gff_file = output_path / "test_annotations.gff"
    
    write_fasta_file(str(fasta_file), contigs)
    write_gff_file(str(gff_file), all_genes)
    
    # Print summary
    total_genes = sum(len(genes) for genes in all_genes.values())
    total_exons = sum(len(gene.exons) for genes in all_genes.values() for gene in genes)
    
    print(f"Generated test fixture in {output_dir}:")
    print(f"  - {num_contigs} contigs, {contig_length} bp each")
    print(f"  - {total_genes} genes total")
    print(f"  - {total_exons} exons total")
    print(f"  - Files: {fasta_file.name}, {gff_file.name}")
    
    return str(fasta_file), str(gff_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic test fixture")
    parser.add_argument("--output-dir", default="tests/fixtures", help="Output directory")
    parser.add_argument("--num-contigs", type=int, default=3, help="Number of contigs")
    parser.add_argument("--contig-length", type=int, default=7500, help="Length of each contig")
    
    args = parser.parse_args()
    
    # Set random seed for reproducible results
    random.seed(42)
    
    generate_test_fixture(args.output_dir, args.num_contigs, args.contig_length)

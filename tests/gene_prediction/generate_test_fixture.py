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
        'GGT', 'GGC', 'GGA', 'GGG', 'CCA', 'CCC', 'CCG', 'TAT',  # More codons
        'TAC', 'TGG', 'TGT', 'TGC', 'AGG', 'AGA', 'CGT', 'CGC',  # Complete set
    ]
    
    sequence = ""
    for _ in range(num_codons):
        sequence += random.choice(codons)
    
    if include_stop:
        stop_codons = ['TAA', 'TAG', 'TGA']
        sequence += random.choice(stop_codons)
    
    return sequence


def generate_intron_sequence(length: int) -> str:
    """Generate an intron sequence with proper splice sites."""
    if length < 4:
        raise ValueError("Intron must be at least 4bp (GT...AG)")
    
    # Splice donor sites (start of intron)
    donor_sites = ['GT', 'GC', 'GA']  # Most common is GT (~99%), GC and GA are rare
    donor = random.choices(donor_sites, weights=[95, 3, 2])[0]  # Realistic distribution
    
    # Splice acceptor site (end of intron) - always AG
    acceptor = 'AG'
    
    # Generate middle sequence
    middle_length = length - 4  # 2 for donor + 2 for acceptor
    middle = generate_random_dna(middle_length)
    
    return donor + middle + acceptor


class SyntheticGene:
    def __init__(self, gene_id: str, start: int, strand: str = '+'):
        self.gene_id = gene_id
        self.start = start
        self.strand = strand
        self.exons = []  # List of (start, end) tuples
        self.introns = []  # List of (start, end) tuples
        self.cds_sequence = ""  # Complete CDS (concatenated exons)
        self.genomic_sequence = {}  # Dict mapping position to nucleotide
        self.end = start
        
    def generate_gene_structure(self, num_exons: int, total_cds_length: int, intron_lengths: List[int]):
        """Generate complete gene structure with proper CDS and splice sites."""
        if num_exons != len(intron_lengths) + 1:
            raise ValueError(f"Need {num_exons-1} intron lengths for {num_exons} exons")
            
        # Generate complete CDS first (must be multiple of 3)
        if total_cds_length % 3 != 0:
            total_cds_length = (total_cds_length // 3) * 3  # Round down to multiple of 3
            
        # Start with ATG
        self.cds_sequence = 'ATG'
        
        # Add middle codons
        num_middle_codons = (total_cds_length - 6) // 3  # -6 for start and stop codons
        if num_middle_codons > 0:
            self.cds_sequence += generate_codon_sequence(num_middle_codons)
        
        # Add stop codon
        stop_codons = ['TAA', 'TAG', 'TGA']
        self.cds_sequence += random.choice(stop_codons)
        
        # Now distribute this CDS across exons with variable lengths
        self.distribute_cds_across_exons(num_exons, intron_lengths)
    
    def distribute_cds_across_exons(self, num_exons: int, intron_lengths: List[int]):
        """Distribute the CDS across exons of variable lengths."""
        cds_pos = 0
        current_pos = self.start
        
        for i in range(num_exons):
            # Determine exon length (variable, but ensure we use all CDS)
            if i == num_exons - 1:
                # Last exon gets remaining CDS
                exon_cds_length = len(self.cds_sequence) - cds_pos
            else:
                # Random exon length between 6bp and 800bp, but ensure we have enough CDS left
                remaining_cds = len(self.cds_sequence) - cds_pos
                remaining_exons = num_exons - i
                min_length = 6  # Minimum exon size
                max_length = min(800, remaining_cds - (remaining_exons - 1) * min_length)
                max_length = max(min_length, max_length)  # Ensure max >= min
                
                exon_cds_length = random.randint(min_length, max_length)
                # Ensure we don't exceed remaining CDS
                exon_cds_length = min(exon_cds_length, remaining_cds)
            
            # Create exon
            exon_start = current_pos
            exon_end = exon_start + exon_cds_length
            self.exons.append((exon_start, exon_end))
            
            # Store exon sequence in genomic positions
            exon_sequence = self.cds_sequence[cds_pos:cds_pos + exon_cds_length]
            for j, base in enumerate(exon_sequence):
                self.genomic_sequence[exon_start + j] = base
            
            cds_pos += exon_cds_length
            current_pos = exon_end
            
            # Add intron if not the last exon
            if i < num_exons - 1:
                intron_length = intron_lengths[i]
                intron_start = current_pos
                intron_end = current_pos + intron_length
                self.introns.append((intron_start, intron_end))
                
                # Generate intron sequence with proper splice sites
                intron_sequence = generate_intron_sequence(intron_length)
                for j, base in enumerate(intron_sequence):
                    self.genomic_sequence[intron_start + j] = base
                
                current_pos = intron_end
        
        self.end = current_pos
    
    def get_gene_length(self):
        """Get total gene length including introns."""
        return self.end - self.start
    
    def get_cds_length(self):
        """Get total CDS length (sum of exons)."""
        return len(self.cds_sequence)


def create_synthetic_contig(contig_id: str, num_genes: int) -> Tuple[str, List[SyntheticGene]]:
    """
    Create a synthetic genomic contig with specified number of genes.
    The contig size is determined by the genes and intergenic spacers.
    
    Args:
        contig_id: Identifier for the contig
        num_genes: Number of genes to place in the contig
        
    Returns:
        Tuple of (DNA sequence, list of genes)
    """
    genes = []
    current_pos = 0
    
    # Add initial spacer
    initial_spacer = random.randint(300, 800)
    current_pos += initial_spacer
   
    for i in range(num_genes):
        gene_id = f"{contig_id}_gene_{i+1:03d}"  # 3 digits for up to 999 genes
        
        # Create gene at current position
        gene = SyntheticGene(gene_id, current_pos)
        
        # Generate 5-6 exons as requested
        num_exons = random.randint(3, 6)

        # Generate CDS length (multiple of 3, realistic gene sizes)
        cds_length = random.randint(150, 1000)

        if random.random() < 0.1:  # 10% chance very large gene
            cds_length = random.randint(1500, 4500)
            num_exons *= 2

        cds_length = (cds_length // 3) * 3  # Ensure multiple of 3
        
        # Generate variable intron lengths (realistic distribution)
        intron_lengths = []
        for j in range(num_exons - 1):
            intron_len = random.randint(60, 200)
            intron_lengths.append(intron_len)

        # Build the gene structure
        try:
            gene.generate_gene_structure(num_exons, cds_length, intron_lengths)
            genes.append(gene)
            
            # Move position to end of this gene
            current_pos = gene.end
            
            # Add intergenic spacer (except after last gene)
            if i < num_genes - 1:
                intergenic_spacer = random.randint(1000, 1500)
                current_pos += intergenic_spacer
                
        except (ValueError, IndexError) as e:
            # Skip genes that have issues, but don't break the process
            print(f"Warning: Skipping gene {gene_id} due to generation error: {e}")
            continue
    
    # Add final spacer
    final_spacer = random.randint(300, 800)
    total_length = current_pos + final_spacer
    
    # Create the actual DNA sequence
    sequence = list(generate_random_dna(total_length))
    
    # Place all gene sequences into the genomic sequence
    for gene in genes:
        for pos, base in gene.genomic_sequence.items():
            if 0 <= pos < len(sequence):
                sequence[pos] = base
    
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


def generate_test_fixture(output_dir: str, num_contigs: int = 10):
    """Generate complete test fixture with FASTA and GFF files.
    
    Args:
        output_dir: Directory to write output files
        num_contigs: Number of contigs to generate
        
    Each contig will have 15-25 genes (random). Contig size is automatically 
    calculated to accommodate the genes with realistic spacing.
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    contigs = []
    all_genes = {}
    
    for i in range(num_contigs):
        contig_id = f"test_contig_{i+1:02d}"
        
        # Random number of genes per contig (15-25)
        genes_per_contig = random.randint(15, 25)
        
        sequence, genes = create_synthetic_contig(contig_id, genes_per_contig)
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
    total_cds_length = sum(gene.get_cds_length() for genes in all_genes.values() for gene in genes)
    
    # Calculate average contig length
    contig_lengths = [len(seq) for _, seq in contigs]
    avg_contig_length = sum(contig_lengths) / len(contig_lengths) if contig_lengths else 0
    total_sequence_length = sum(contig_lengths)
    
    print(f"Generated test fixture in {output_dir}:")
    print(f"  - {num_contigs} contigs, average ~{avg_contig_length/1000:.0f}kb each")
    print(f"  - Total sequence: {total_sequence_length/1000:.0f}kb")
    print(f"  - {total_genes} genes total")
    print(f"  - {total_exons} exons total")
    print(f"  - {total_cds_length} bp total CDS")
    print(f"  - Average {total_exons/total_genes:.1f} exons per gene")
    print(f"  - Average {total_cds_length/total_genes:.0f} bp CDS per gene")
    print(f"  - Files: {fasta_file.name}, {gff_file.name}")
    
    return str(fasta_file), str(gff_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic test fixture")
    parser.add_argument("--output-dir", default="tests/fixtures", help="Output directory")
    parser.add_argument("--num-contigs", type=int, default=10, help="Number of contigs (each will have 15-25 genes randomly)")
    
    args = parser.parse_args()
    
    # Set random seed for reproducible results
    random.seed(42)
    
    generate_test_fixture(args.output_dir, args.num_contigs)

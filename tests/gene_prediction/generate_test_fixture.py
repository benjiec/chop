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


def create_synthetic_contig(contig_id: str, length: int, num_genes: int) -> Tuple[str, List[SyntheticGene]]:
    """
    Create a synthetic genomic contig with specified number of genes.
    
    Args:
        contig_id: Identifier for the contig
        length: Total length of the contig
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
        gene_id = f"{contig_id}_gene_{i+1:03d}"  # 3 digits for up to 999 genes
        
        # Position gene with some randomness
        base_start = (i + 1) * space_per_gene
        start_pos = base_start + random.randint(-space_per_gene//4, space_per_gene//4)
        start_pos = max(500, min(start_pos, length - 4000))  # Ensure room for gene
        
        # Create gene with random characteristics
        gene = SyntheticGene(gene_id, start_pos)
        
        # Generate 5-6 exons as requested
        num_exons = random.randint(5, 6)
        
        # Generate CDS length (multiple of 3, realistic gene sizes)
        cds_length = random.randint(300, 2400)  # 100-800 codons (300-2400 bp)
        cds_length = (cds_length // 3) * 3  # Ensure multiple of 3
        
        # Generate variable intron lengths (realistic distribution)
        intron_lengths = []
        for j in range(num_exons - 1):
            # Realistic intron sizes: mostly small, some large
            if random.random() < 0.7:  # 70% small introns
                intron_len = random.randint(50, 200)
            else:  # 30% larger introns
                intron_len = random.randint(200, 1500)
            intron_lengths.append(intron_len)
        
        # Build the gene structure
        try:
            gene.generate_gene_structure(num_exons, cds_length, intron_lengths)
            
            # Check if gene fits in contig
            max_available = length - start_pos - 500
            if gene.get_gene_length() <= max_available:
                genes.append(gene)
                
                # Place the gene sequence into the genomic sequence
                for pos, base in gene.genomic_sequence.items():
                    if 0 <= pos < len(sequence):
                        sequence[pos] = base
                
        except (ValueError, IndexError) as e:
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


def generate_test_fixture(output_dir: str, num_contigs: int = 10, contig_length: int = 50000):
    """Generate complete test fixture with FASTA and GFF files."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Generate contigs to accommodate ~200 genes total
    contigs = []
    all_genes = {}
    target_total_genes = 200
    genes_per_contig = target_total_genes // num_contigs
    
    for i in range(num_contigs):
        contig_id = f"test_contig_{i+1:02d}"
        # Distribute genes across contigs
        if i == num_contigs - 1:
            # Last contig gets any remaining genes
            num_genes = target_total_genes - (genes_per_contig * (num_contigs - 1))
        else:
            num_genes = genes_per_contig + random.randint(-2, 2)  # Small variation
        
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
    total_cds_length = sum(gene.get_cds_length() for genes in all_genes.values() for gene in genes)
    
    print(f"Generated test fixture in {output_dir}:")
    print(f"  - {num_contigs} contigs, ~{contig_length/1000:.0f}kb each")
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
    parser.add_argument("--num-contigs", type=int, default=10, help="Number of contigs")
    parser.add_argument("--contig-length", type=int, default=50000, help="Length of each contig")
    
    args = parser.parse_args()
    
    # Set random seed for reproducible results
    random.seed(42)
    
    generate_test_fixture(args.output_dir, args.num_contigs, args.contig_length)

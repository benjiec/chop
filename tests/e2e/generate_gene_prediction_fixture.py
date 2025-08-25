#!/usr/bin/env python3
"""
Generate synthetic genomic test fixtures for gene prediction.

This script creates realistic FASTA and GFF files with proper UTR patterns,
gene boundaries, and realistic genomic structure for testing the gene
prediction pipeline.

Gene prediction focuses on: 5'UTR -> START -> GENE_BODY -> STOP -> 3'UTR
"""

import random
import sys
from pathlib import Path
from typing import List, Tuple, Dict, NamedTuple

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from utils.constants import UTR5_SIZE, UTR3_SIZE


class UTRPatterns:
    """Common UTR regulatory patterns found in eukaryotic genes."""
    
    # 5' UTR patterns
    TATA_BOXES = [
        "TATAAA", "TATAWAW", "TATAWAR", "TATAWWA"  # W = A or T, R = A or G
    ]
    
    KOZAK_SEQUENCES = [
        "GCCRCCATGG", "GCCACCATGG", "GCCGCCATGG",  # R = A or G
        "GCAACCATGG", "GCCACCAUG", "ACCACCATGG"
    ]
    
    # 3' UTR patterns
    POLYA_SIGNALS = [
        "AAUAAA", "AAAAAA", "AUUAAA", "AAUACA", 
        "AAUAUA", "AAUAAG", "AAAAAG", "AAWAAA"  # W = A or U
    ]
    
    CDS_SIGNALS = [
        "AUUAAA", "AAUACA", "AAUAUA"
    ]


class SyntheticGene(NamedTuple):
    """Represents a complete gene with UTRs for gene prediction testing."""
    gene_id: str
    start_pos: int
    end_pos: int
    strand: str
    utr5_seq: str
    cds_seq: str  # includes START and STOP
    utr3_seq: str
    full_sequence: str
    
    # Boundary positions within the gene
    utr5_start: int
    utr5_end: int
    start_codon_pos: int
    cds_start: int
    cds_end: int
    stop_codon_pos: int
    utr3_start: int
    utr3_end: int


def generate_random_dna(length: int, gc_content: float = 0.5) -> str:
    """Generate random DNA sequence with specified GC content."""
    gc_bases = int(length * gc_content)
    at_bases = length - gc_bases
    
    bases = ['G'] * (gc_bases // 2) + ['C'] * (gc_bases // 2)
    bases += ['A'] * (at_bases // 2) + ['T'] * (at_bases // 2)
    
    # Add remaining bases if length is odd
    if len(bases) < length:
        bases.extend(random.choices(['A', 'T', 'G', 'C'], k=length - len(bases)))
    
    random.shuffle(bases)
    return ''.join(bases)


def generate_5utr_with_patterns(target_length: int = UTR5_SIZE) -> str:
    """Generate 5' UTR with realistic regulatory patterns."""
    sequence = []
    remaining = target_length
    
    # Add TATA box (around position 25-30 from TSS)
    tata_pos = random.randint(20, 35)
    if tata_pos < remaining:
        # Leading sequence
        if tata_pos > 0:
            sequence.append(generate_random_dna(tata_pos))
            remaining -= tata_pos
        
        # TATA box
        tata = random.choice(UTRPatterns.TATA_BOXES)
        # Convert ambiguous nucleotides
        tata = tata.replace('W', random.choice(['A', 'T']))
        tata = tata.replace('R', random.choice(['A', 'G']))
        sequence.append(tata)
        remaining -= len(tata)
    
    # Add Kozak sequence near the end (just before START)
    kozak_space = 15
    if remaining > kozak_space:
        # Middle filler
        middle_len = remaining - kozak_space
        if middle_len > 0:
            sequence.append(generate_random_dna(middle_len))
            remaining -= middle_len
        
        # Kozak-like sequence (without the ATG)
        kozak = random.choice(UTRPatterns.KOZAK_SEQUENCES)
        kozak = kozak.replace('R', random.choice(['A', 'G']))
        # Remove ATG from Kozak since that's the START codon
        kozak = kozak.replace('ATG', '').replace('AUG', '')
        if len(kozak) <= remaining:
            sequence.append(kozak)
            remaining -= len(kozak)
    
    # Fill remaining space
    if remaining > 0:
        sequence.append(generate_random_dna(remaining))
    
    result = ''.join(sequence)
    # Trim to exact length
    if len(result) > target_length:
        result = result[:target_length]
    elif len(result) < target_length:
        result += generate_random_dna(target_length - len(result))
    
    return result


def generate_3utr_with_patterns(target_length: int = UTR3_SIZE) -> str:
    """Generate 3' UTR with realistic polyadenylation signals."""
    sequence = []
    remaining = target_length
    
    # Add polyadenylation signal early in 3' UTR
    polya_pos = random.randint(10, 30)
    if polya_pos < remaining:
        # Leading sequence
        if polya_pos > 0:
            sequence.append(generate_random_dna(polya_pos))
            remaining -= polya_pos
        
        # PolyA signal (convert T to A for DNA)
        polya = random.choice(UTRPatterns.POLYA_SIGNALS)
        polya = polya.replace('U', 'T').replace('W', random.choice(['A', 'T']))
        sequence.append(polya)
        remaining -= len(polya)
    
    # Fill remaining space
    if remaining > 0:
        sequence.append(generate_random_dna(remaining))
    
    result = ''.join(sequence)
    # Trim to exact length
    if len(result) > target_length:
        result = result[:target_length]
    elif len(result) < target_length:
        result += generate_random_dna(target_length - len(result))
    
    return result


def generate_coding_sequence(target_length: int) -> str:
    """Generate a coding sequence with START and STOP codons."""
    # Ensure length is multiple of 3
    cds_length = (target_length // 3) * 3
    if cds_length < 6:  # Need at least START + STOP
        cds_length = 6
    
    # Start with ATG
    sequence = ["ATG"]
    remaining_codons = (cds_length // 3) - 2  # -2 for START and STOP
    
    # Generate random codons (avoiding stop codons)
    valid_codons = []
    for base1 in 'ATGC':
        for base2 in 'ATGC':
            for base3 in 'ATGC':
                codon = base1 + base2 + base3
                if codon not in ['TAA', 'TAG', 'TGA']:  # Not stop codons
                    valid_codons.append(codon)
    
    # Add random codons
    for _ in range(remaining_codons):
        sequence.append(random.choice(valid_codons))
    
    # End with stop codon
    stop_codon = random.choice(['TAA', 'TAG', 'TGA'])
    sequence.append(stop_codon)
    
    return ''.join(sequence)


def create_synthetic_gene(gene_id: str, target_length: int) -> SyntheticGene:
    """Create a synthetic gene with realistic UTR patterns for gene prediction."""
    
    # Calculate component lengths
    utr5_len = UTR5_SIZE
    utr3_len = UTR3_SIZE
    cds_len = target_length - utr5_len - utr3_len
    
    if cds_len < 6:  # Minimum for START + STOP
        cds_len = 6
        # Adjust total length
        target_length = utr5_len + cds_len + utr3_len
    
    # Generate sequences
    utr5_seq = generate_5utr_with_patterns(utr5_len)
    cds_seq = generate_coding_sequence(cds_len)
    utr3_seq = generate_3utr_with_patterns(utr3_len)
    
    # Combine full sequence
    full_sequence = utr5_seq + cds_seq + utr3_seq
    
    # Calculate boundary positions (0-based, relative to gene start)
    utr5_start = 0
    utr5_end = utr5_len - 1
    start_codon_pos = utr5_len  # Position of ATG
    cds_start = utr5_len
    cds_end = utr5_len + cds_len - 1
    stop_codon_pos = utr5_len + cds_len - 3  # Position of stop codon
    utr3_start = utr5_len + cds_len
    utr3_end = utr5_len + cds_len + utr3_len - 1
    
    return SyntheticGene(
        gene_id=gene_id,
        start_pos=0,  # Will be set when placed in contig
        end_pos=len(full_sequence) - 1,  # Will be adjusted when placed
        strand='+',
        utr5_seq=utr5_seq,
        cds_seq=cds_seq,
        utr3_seq=utr3_seq,
        full_sequence=full_sequence,
        utr5_start=utr5_start,
        utr5_end=utr5_end,
        start_codon_pos=start_codon_pos,
        cds_start=cds_start,
        cds_end=cds_end,
        stop_codon_pos=stop_codon_pos,
        utr3_start=utr3_start,
        utr3_end=utr3_end
    )


def create_synthetic_contig(contig_id: str, contig_length: int, 
                          genes_per_contig: int) -> Tuple[str, List[SyntheticGene]]:
    """Create a synthetic contig with multiple genes for gene prediction."""
    
    genes = []
    contig_sequence = ['N'] * contig_length
    
    # Generate genes with reasonable spacing
    available_space = contig_length
    gene_spacing = available_space // (genes_per_contig + 1)  # +1 for end spacing
    
    for i in range(genes_per_contig):
        gene_id = f"{contig_id}_gene_{i+1:03d}"
        
        # Target gene length (1.5-2.5kb including UTRs, averaging 2kb)
        target_gene_length = random.randint(1500, 2500)
        
        # Create gene
        gene = create_synthetic_gene(gene_id, target_gene_length)
        
        # Position gene in contig
        start_pos = (i + 1) * gene_spacing - len(gene.full_sequence) // 2
        start_pos = max(0, min(start_pos, contig_length - len(gene.full_sequence)))
        end_pos = start_pos + len(gene.full_sequence) - 1
        
        # Check for overlap with existing genes
        overlaps = False
        for existing_gene in genes:
            if not (end_pos < existing_gene.start_pos or start_pos > existing_gene.end_pos):
                overlaps = True
                break
        
        if not overlaps and end_pos < contig_length:
            # Update gene coordinates
            gene = gene._replace(start_pos=start_pos, end_pos=end_pos)
            
            # Insert gene sequence into contig
            for j, base in enumerate(gene.full_sequence):
                if start_pos + j < contig_length:
                    contig_sequence[start_pos + j] = base
            
            genes.append(gene)
    
    # Fill remaining N's with random sequence
    final_sequence = []
    for base in contig_sequence:
        if base == 'N':
            final_sequence.append(random.choice(['A', 'T', 'G', 'C']))
        else:
            final_sequence.append(base)
    
    return ''.join(final_sequence), genes


def generate_gene_prediction_fixture(output_dir: Path, num_contigs: int = 5, 
                                    total_genes: int = 200) -> Tuple[Path, Path]:
    """Generate complete test fixture for gene prediction."""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fasta_file = output_dir / "gene_prediction_test.fna"
    gff_file = output_dir / "gene_prediction_test.gff"
    
    genes_per_contig = total_genes // num_contigs
    # Calculate contig length based on gene size: ~2kb per gene + 50% spacing
    contig_length = max(50000, genes_per_contig * 3000)  # At least 50kb, or 3kb per gene
    
    all_genes = []
    
    # Write FASTA file
    with open(fasta_file, 'w') as f:
        for contig_idx in range(num_contigs):
            contig_id = f"contig_{contig_idx+1:03d}"
            
            # Create contig with genes
            contig_seq, contig_genes = create_synthetic_contig(
                contig_id, contig_length, genes_per_contig
            )
            
            # Write FASTA entry
            f.write(f">{contig_id}\n")
            # Write sequence in 80-character lines
            for i in range(0, len(contig_seq), 80):
                f.write(contig_seq[i:i+80] + "\n")
            
            # Update gene coordinates with contig context
            for gene in contig_genes:
                gene_with_contig = gene._replace(
                    gene_id=f"{contig_id}_{gene.gene_id.split('_', 2)[-1]}"
                )
                all_genes.append((contig_id, gene_with_contig))
    
    # Write GFF file
    with open(gff_file, 'w') as f:
        f.write("##gff-version 3\n")
        
        for contig_id, gene in all_genes:
            gene_start = gene.start_pos + 1  # GFF is 1-based
            gene_end = gene.end_pos + 1
            
            # Gene feature
            f.write(f"{contig_id}\tsynthetic\tgene\t{gene_start}\t{gene_end}\t.\t+\t.\t"
                   f"ID={gene.gene_id};Name={gene.gene_id}\n")
            
            # CDS feature (just the coding part, not UTRs)
            cds_start = gene.start_pos + gene.cds_start + 1
            cds_end = gene.start_pos + gene.cds_end + 1
            f.write(f"{contig_id}\tsynthetic\tCDS\t{cds_start}\t{cds_end}\t.\t+\t0\t"
                   f"ID={gene.gene_id}_CDS;Parent={gene.gene_id}\n")
    
    print(f"Generated gene prediction test fixture:")
    print(f"  FASTA: {fasta_file}")
    print(f"  GFF: {gff_file}")
    print(f"  Contigs: {num_contigs}")
    print(f"  Genes: {len(all_genes)}")
    
    return fasta_file, gff_file


if __name__ == "__main__":
    import tempfile
    
    # Generate test fixture
    with tempfile.TemporaryDirectory() as tmp_dir:
        fasta_file, gff_file = generate_gene_prediction_fixture(
            output_dir=tmp_dir,
            num_contigs=5,
            total_genes=200
        )
        
        print(f"\nTest fixture generated in: {tmp_dir}")
        print("Files created successfully!")

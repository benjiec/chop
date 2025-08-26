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
    """Generate a simple coding sequence with START and STOP codons."""
    if target_length < 6:  # Need at least ATG + XXX + STOP
        target_length = 6
    
    # Start with ATG
    sequence = "ATG"
    
    # Add random sequence in the middle  
    middle_length = target_length - 6  # -3 for ATG, -3 for stop
    if middle_length > 0:
        middle_seq = ''.join(random.choices('ATGC', k=middle_length))
        sequence += middle_seq
    
    # End with a stop codon
    stop_codon = random.choice(['TAA', 'TAG', 'TGA'])
    sequence += stop_codon
    
    # DEBUG: Check if we generated valid START/STOP
    if sequence[:3] != 'ATG':
        print(f"⚠️  WARNING: generate_coding_sequence created invalid START: '{sequence[:3]}'")
    if sequence[-3:] not in ['TAA', 'TAG', 'TGA']:
        print(f"⚠️  WARNING: generate_coding_sequence created invalid STOP: '{sequence[-3:]}'")
    
    return sequence


def create_synthetic_gene(gene_id: str, target_cds_length: int) -> SyntheticGene:
    """Create a synthetic gene with realistic UTR patterns for gene prediction."""
    
    # Calculate component lengths
    utr5_len = UTR5_SIZE
    utr3_len = UTR3_SIZE
    cds_len = target_cds_length  # target_length now refers to CDS only
    
    if cds_len < 6:  # Minimum for START + STOP
        cds_len = 6
    
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
    cds_end = utr5_len + cds_len - 1  # Inclusive end position
    stop_codon_pos = utr5_len + cds_len - 3  # Position of stop codon
    utr3_start = utr5_len + cds_len
    utr3_end = utr5_len + cds_len + utr3_len - 1
    
    # DEBUG: Check CDS extraction from full sequence
    extracted_cds = full_sequence[cds_start:cds_end+1]
    if extracted_cds != cds_seq:
        print(f"⚠️  WARNING: {gene_id} CDS extraction mismatch!")
        print(f"    Expected: '{cds_seq[:10]}...{cds_seq[-10:]}'")
        print(f"    Extracted: '{extracted_cds[:10]}...{extracted_cds[-10:]}'")
    
    if cds_seq[:3] != 'ATG':
        print(f"⚠️  WARNING: {gene_id} has invalid START: '{cds_seq[:3]}'")
    if cds_seq[-3:] not in ['TAA', 'TAG', 'TGA']:
        print(f"⚠️  WARNING: {gene_id} has invalid STOP: '{cds_seq[-3:]}'")
    
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


def create_synthetic_contig(contig_id: str, estimated_length: int, 
                          genes_per_contig: int) -> Tuple[str, List[SyntheticGene]]:
    """Create a synthetic contig with multiple genes for gene prediction."""
    
    sequences = []
    genes = []
    current_position = 0
    
    # Start with intergenic spacer
    spacer_length = random.randint(500, 1500)  # Random intergenic region
    spacer_seq = generate_random_dna(spacer_length)
    sequences.append(spacer_seq)
    current_position += spacer_length
    
    gene_count = 0
    while current_position < estimated_length and gene_count < genes_per_contig:
        gene_count += 1
        gene_id = f"{contig_id}_gene_{gene_count:03d}"
        
        # Create gene with random CDS length
        target_cds_length = random.randint(800, 1200)
        gene = create_synthetic_gene(gene_id, target_cds_length)
        
        # Set gene coordinates
        gene = gene._replace(
            start_pos=current_position,
            end_pos=current_position + len(gene.full_sequence) - 1
        )
        
        # DEBUG: Check gene placement in sequences array
        print(f"DEBUG: {gene_id} placement:")
        print(f"  current_position: {current_position}")
        print(f"  gene.cds_start: {gene.cds_start} (relative)")
        print(f"  gene.cds_end: {gene.cds_end} (relative)")
        print(f"  gene.cds_seq: '{gene.cds_seq[:10]}...{gene.cds_seq[-10:]}'")
        
        # Add gene sequence
        sequences.append(gene.full_sequence)
        genes.append(gene)
        current_position += len(gene.full_sequence)
        
        # DEBUG: Build temp contig and check coordinates
        temp_contig = ''.join(sequences)
        cds_global_start = gene.start_pos + gene.cds_start
        cds_global_end = gene.start_pos + gene.cds_end
        
        if cds_global_end < len(temp_contig):
            extracted_cds = temp_contig[cds_global_start:cds_global_end+1]
            print(f"  temp_contig length: {len(temp_contig)}")
            print(f"  CDS global coords: {cds_global_start}-{cds_global_end}")
            print(f"  extracted CDS: '{extracted_cds[:10]}...{extracted_cds[-10:]}'")
            print(f"  CDS match: {gene.cds_seq == extracted_cds}")
            if gene.cds_seq != extracted_cds:
                print(f"  ❌ CDS MISMATCH in sequences array!")
            print()
        
        # Add spacer after gene (unless we're at the end)
        if current_position < estimated_length:
            spacer_length = random.randint(500, 1500)
            spacer_seq = generate_random_dna(spacer_length)
            sequences.append(spacer_seq)
            current_position += spacer_length
    
    # Combine all sequences
    final_sequence = ''.join(sequences)
    
    return final_sequence, genes


def generate_gene_prediction_fixture(output_dir: Path, num_contigs: int = 5, 
                                    total_genes: int = 200) -> Tuple[Path, Path]:
    """Generate complete test fixture for gene prediction."""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fasta_file = output_dir / "gene_prediction_test.fna"
    gff_file = output_dir / "gene_prediction_test.gff"
    
    genes_per_contig = total_genes // num_contigs
    # Calculate contig length: each gene ~2kb + spacers ~1kb = 3kb per gene
    contig_length = max(30000, genes_per_contig * 3000)  # At least 30kb, or 3kb per gene
    
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
    
    # Read back the FASTA to check CDS sequences in GFF writing
    contig_sequences = {}
    with open(fasta_file, 'r') as f:
        current_contig = None
        current_seq = []
        for line in f:
            if line.startswith('>'):
                if current_contig:
                    contig_sequences[current_contig] = ''.join(current_seq).upper()
                current_contig = line[1:].strip()
                current_seq = []
            else:
                current_seq.append(line.strip())
        if current_contig:
            contig_sequences[current_contig] = ''.join(current_seq).upper()
    
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
            
            # DEBUG: Extract actual CDS using GFF coordinates and compare
            if contig_id in contig_sequences:
                contig_seq = contig_sequences[contig_id]
                # Convert GFF coordinates to 0-based for extraction
                extract_start = cds_start - 1
                extract_end = cds_end
                
                if extract_end <= len(contig_seq):
                    extracted_cds = contig_seq[extract_start:extract_end]
                    
                    # Compare stored vs extracted
                    if gene.cds_seq != extracted_cds:
                        print(f"⚠️  WARNING: {gene.gene_id} CDS MISMATCH!")
                        print(f"    Stored:    '{gene.cds_seq[:10]}...{gene.cds_seq[-10:]}' ({len(gene.cds_seq)}bp)")
                        print(f"    Extracted: '{extracted_cds[:10]}...{extracted_cds[-10:]}' ({len(extracted_cds)}bp)")
                    
                    # Check extracted sequence START/STOP
                    if extracted_cds[:3] != 'ATG':
                        print(f"⚠️  WARNING: {gene.gene_id} extracted CDS has invalid START: '{extracted_cds[:3]}'")
                    if extracted_cds[-3:] not in ['TAA', 'TAG', 'TGA']:
                        print(f"⚠️  WARNING: {gene.gene_id} extracted CDS has invalid STOP: '{extracted_cds[-3:]}'")
                else:
                    print(f"⚠️  WARNING: {gene.gene_id} coordinates {cds_start}-{cds_end} exceed contig length {len(contig_seq)}")
            
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

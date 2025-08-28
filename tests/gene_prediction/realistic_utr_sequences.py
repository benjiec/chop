#!/usr/bin/env python3
"""
Realistic UTR sequences from public databases and literature.

This module contains well-characterized 5' UTR and 3' UTR sequences that can be
used to make synthetic gene fixtures more biologically realistic.
"""

import random
from typing import List

# ==============================================================================
# 5' UTR SEQUENCES
# ==============================================================================

# Kozak consensus sequences (optimal ribosome binding)
KOZAK_SEQUENCES = [
    # Classic Kozak consensus: (gcc)gccRccAUGG -- when this is inserted the ATG is placed to overlap with the actual ATG in the CDS
    "GCCGCCACCATG",  # Perfect Kozak
    "GCCACCACCATG",  # Strong Kozak
    "GCCGCCATCATG",  # Good Kozak
    "ACCGCCACCATG",  # Good Kozak variant
    "GCTGCCACCATG",  # Common variant
    "GCCACCATCATG",  # Moderate Kozak
    "AGCGCCACCATG",  # Natural variant
    "GCCGCTACCATG",  # Moderate variant
]

# Real 5' UTR sequences from well-studied human genes
UTR5_REAL_SEQUENCES = [
    # Human β-globin (HBB) - Classic example, shortened for insertion
    "ACATTTGCTTCTGACACAACTGTGTTCACTAGCAACCTCAAACAGACACC",
    
    # Human GAPDH - Housekeeping gene
    "GGTGAAGGTCGGAGTCAACGGATTTGGTCGTATTGGGCGCCTGGTCACCAGGGCTGCTTTTAACTCTGGTA",
    
    # Human p53 tumor suppressor
    "GTGCAGCTGTGGGTTGATTCCACACCCCCGCCCGGCACCCGCGTCCGCGCC",
    
    # Human c-MYC oncogene
    "CGGCAACAGGCCGGCGGTGGGTCCGGGGCGGGACGGGGCGGGTCTCGTCGGGCGGGGCGGGGCTGGGGCGG",
    
    # Human ACTB (β-actin)
    "GCGGGGCCGGGAGCCGCGCTCGGGGTTGGGGACTGACGCGCGCGCGCCGCGCGCGCGCGCGCGCGCGCCGCC",
    
    # Shorter realistic sequences for easier insertion
    "TGCAGCTGTGGGTTGATTCCACACCCC",
    "GGCAACAGGCCGGCGGTGGGTCCGGGG",
    "GGGCCGGGAGCCGCGCTCGGGGTTGGG",
    "CATTTGCTTCTGACACAACTGTGTTCA",
    "GAAGGTCGGAGTCAACGGATTTGGTCG",
]

# IRES (Internal Ribosome Entry Site) sequences - shortened for practical use
IRES_SEQUENCES = [
    # HCV IRES (simplified core region)
    "GCCAGCCCCCGATTGGGGGCGACACTCCACCATGAATCACTCCCCTGTGAGGAACTACTGTCTTCACGCAG",
    
    # EMCV IRES (core region)
    "TTAAAACAGCCTGGGGTTGTACCCACCCCAGAGGCCCACGTGGCGGCTAGTACTCCGGTATTGCGGTACCC",
    
    # Poliovirus IRES (core region)
    "TTCCCTAATTCTGGGGTAACCGGGCCCCCACCTGAGGGTTTGAACGCCTTTCCCATCTCGGTACCGAAAG",
    
    # Simplified IRES-like sequences
    "CCCCGATTGGGGGCGACACTCCACCATG",
    "CCTGGGGTTGTACCCACCCCAGAGGCCC",
    "CTAATTCTGGGGTAACCGGGCCCCCACC",
]

# ==============================================================================
# 3' UTR SEQUENCES
# ==============================================================================

# Polyadenylation signals and variants
POLYA_SIGNALS = [
    # Canonical poly-A signals
    "AATAAA",  # Most common (~60%)
    "ATTAAA",  # Second most common (~15%)
    "AGTAAA",  # Less common variant
    "TATAAA",  # Less common variant
    "CATAAA",  # Rare variant
    "GATAAA",  # Rare variant
    "AATATA",  # Variant
    "AATACA",  # Variant
    "AATAGA",  # Variant
    "AAAAAG",  # Variant
]

# AU-rich elements (AREs) for mRNA stability regulation
ARE_SEQUENCES = [
    # Core ARE motifs
    "ATTTA",     # Pentamer ARE
    "TATTTAT",   # Extended ARE
    "TATTTATT",  # Longer ARE
    "ATTTATTTAT", # Multiple ATTTA
    "TATTTATTTAT", # Extended multiple
    
    # Realistic ARE contexts
    "TATTTATTTATTTATTTA",  # Multiple repeats
    "ATTTAATTTAATTTAT",     # Spaced AREs
    "TATTTATTTAATTTA",      # Mixed spacing
    "ATTTATTTATTTATTTAT",   # Long ARE cluster
]

# GU-rich elements (GREs)
GRE_SEQUENCES = [
    "TGTTGTTGT",
    "GTTGTTGTTGT", 
    "TGTTTGTTTTGT",
    "GTTGTTTGTTGT",
    "TGTTGTTGTTGTTGT",
]

# Real 3' UTR sequences from human genes (shortened for insertion)
UTR3_REAL_SEQUENCES = [
    # Sequences with poly-A signals embedded
    "TGAGGGGAAACAACATGGCAAATAAAT" + "AATAAA" + "GCAGAAATTAATAAAAACAATAAATAA",
    "CCTTCCACGATACCAAAGTTGTCATGG" + "ATTAAA" + "TGACCTTGGCCAGGGGGCCATCCACAG",
    "GAGTCCTTCCACGATACCAAAGTTGTC" + "AATAAA" + "ATGGATGACCTTGGCCAGGGGGCCATC",
    "CTGAGTCCTTCCACGATACCAAAGTTG" + "AGTAAA" + "TCATGGATGACCTTGGCCAGGGGGCCA",
    
    # Sequences with AREs
    "TGAGGGGAAACAACATGGCAAATAAAT" + "ATTTA" + "GCAGAAATTAATAAAAACAATAAATAA",
    "CCTTCCACGATACCAAAGTTGTCATGG" + "TATTTAT" + "GACCTTGGCCAGGGGGCCATCCACAG",
    "GAGTCCTTCCACGATACCAAAGTTGTC" + "ATTTATTTAT" + "ATGGATGACCTTGGCCAGGGGGCCATC",
    
    # Simple realistic 3' UTR contexts
    "TGAGGGGAAACAACATGGCAAATAAAT",
    "CCTTCCACGATACCAAAGTTGTCATGG", 
    "GAGTCCTTCCACGATACCAAAGTTGTC",
    "CTGAGTCCTTCCACGATACCAAAGTTG",
    "TCCACGATACCAAAGTTGTCATGGATG",
]

# Complete 3' UTR sequences with regulatory elements
UTR3_COMPLETE_SEQUENCES = [
    # Realistic 3' UTR with poly-A signal
    "TGAGGGGAAACAACATGGCAAATAAATAAGGCAGAAATTAATAAAAACAATAAAT" + "AATAAA" + "ATCTTAGTAATAAAATACTTATCTAAAAAA",
    
    # 3' UTR with ARE elements
    "CCTTCCACGATACCAAAGTTGTCATGGATGACCTTGGCCAGGGGGCCATCCACAG" + "TATTTAT" + "TCTTCTGGGTGGCAGTGATGGCATGGACT",
    
    # 3' UTR with GRE elements  
    "GAGTCCTTCCACGATACCAAAGTTGTCATGGATGACCTTGGCCAGGGGGCCATCC" + "TGTTGTTGT" + "ACAGTCTTCTGGGTGGCAGTGATGGCAT",
]

# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def mutate_sequence(sequence: str, mutation_rate: float = 0.1) -> str:
    """
    Randomly mutate 1-2 bases in a sequence with given probability.
    
    Args:
        sequence: DNA sequence to potentially mutate
        mutation_rate: Probability of mutation (default 10%)
        
    Returns:
        Potentially mutated sequence
    """
    if random.random() > mutation_rate:
        return sequence  # No mutation
    
    sequence_list = list(sequence)
    bases = ['A', 'T', 'G', 'C']
    
    # Randomly choose 1 or 2 positions to mutate
    num_mutations = random.choice([1, 2])
    positions = random.sample(range(len(sequence)), min(num_mutations, len(sequence)))
    
    for pos in positions:
        original_base = sequence_list[pos]
        # Choose a different base
        new_bases = [b for b in bases if b != original_base]
        sequence_list[pos] = random.choice(new_bases)
    
    return ''.join(sequence_list)

def get_random_kozak_sequence() -> str:
    """Get a random Kozak sequence, potentially mutated."""
    kozak = random.choice(KOZAK_SEQUENCES)
    return mutate_sequence(kozak)

def get_random_5utr_sequence() -> str:
    """Get a random 5' UTR sequence, potentially mutated."""
    # Mix of different types
    all_5utr = UTR5_REAL_SEQUENCES + IRES_SEQUENCES
    utr = random.choice(all_5utr)
    return mutate_sequence(utr)

def get_random_polya_signal() -> str:
    """Get a random poly-A signal, potentially mutated."""
    signal = random.choice(POLYA_SIGNALS)
    return mutate_sequence(signal)

def get_random_3utr_sequence() -> str:
    """Get a random 3' UTR sequence, potentially mutated."""
    # Mix of different types  
    all_3utr = UTR3_REAL_SEQUENCES + UTR3_COMPLETE_SEQUENCES
    utr = random.choice(all_3utr)
    return mutate_sequence(utr)

def get_random_are_sequence() -> str:
    """Get a random AU-rich element, potentially mutated."""
    are = random.choice(ARE_SEQUENCES)
    return mutate_sequence(are)

def get_random_gre_sequence() -> str:
    """Get a random GU-rich element, potentially mutated."""
    gre = random.choice(GRE_SEQUENCES)
    return mutate_sequence(gre)

# ==============================================================================
# MAIN FUNCTIONS FOR FIXTURE GENERATION
# ==============================================================================

def insert_5utr_elements_near_start(sequence: List[str], start_pos: int, window: int = 50) -> None:
    """
    Insert realistic 5' UTR elements near a START codon position.
    
    Args:
        sequence: List representation of DNA sequence (mutable)
        start_pos: Position of START codon in the sequence
        window: How far upstream to consider for insertion (default 50bp)
    """
    if start_pos < window:
        return  # Not enough space upstream
    
    # 70% chance to insert a Kozak sequence right before START
    if random.random() < 0.7:
        kozak = get_random_kozak_sequence()
        # Insert Kozak ending right at the START codon
        insert_pos = max(0, start_pos - len(kozak) + 3)  # +3 to overlap with ATG
        for i, base in enumerate(kozak[:-3]):  # Don't overwrite the ATG
            if insert_pos + i < start_pos:
                sequence[insert_pos + i] = base
    
    # 30% chance to insert a longer 5' UTR element further upstream
    if random.random() < 0.3:
        utr_element = get_random_5utr_sequence()
        # Insert further upstream
        insert_pos = start_pos - random.randint(20, min(window-10, len(utr_element)+10))
        insert_pos = max(0, insert_pos)
        for i, base in enumerate(utr_element):
            if insert_pos + i < start_pos - 3:  # Don't get too close to START
                sequence[insert_pos + i] = base

def insert_3utr_elements_near_stop(sequence: List[str], stop_pos: int, window: int = 50) -> None:
    """
    Insert realistic 3' UTR elements near a STOP codon position.
    
    Args:
        sequence: List representation of DNA sequence (mutable)
        stop_pos: Position after STOP codon in the sequence  
        window: How far downstream to consider for insertion (default 50bp)
    """
    if stop_pos + window >= len(sequence):
        return  # Not enough space downstream
    
    # 60% chance to insert a poly-A signal downstream
    if random.random() < 0.6:
        polya = get_random_polya_signal()
        # Insert poly-A signal 10-30 bp downstream of STOP
        insert_pos = stop_pos + random.randint(10, 30)
        for i, base in enumerate(polya):
            if insert_pos + i < len(sequence):
                sequence[insert_pos + i] = base
    
    # 40% chance to insert ARE or other 3' UTR elements
    if random.random() < 0.4:
        if random.random() < 0.7:
            element = get_random_are_sequence()
        else:
            element = get_random_3utr_sequence()
            element = element[:min(len(element), 30)]  # Limit length for insertion
        
        # Insert further downstream
        insert_pos = stop_pos + random.randint(15, min(window-5, 40))
        for i, base in enumerate(element):
            if insert_pos + i < len(sequence):
                sequence[insert_pos + i] = base

def insert_regulatory_elements_in_intergenic(sequence: List[str], start: int, end: int) -> None:
    """
    Occasionally insert regulatory-like elements in intergenic spacers.
    
    Args:
        sequence: List representation of DNA sequence (mutable)
        start: Start position of intergenic region
        end: End position of intergenic region
    """
    region_length = end - start
    
    # 20% chance to insert some regulatory elements in intergenic space
    if random.random() < 0.2 and region_length > 100:
        # Choose what type of element to insert
        if random.random() < 0.4:
            element = get_random_5utr_sequence()
        elif random.random() < 0.7:
            element = get_random_3utr_sequence()  
        else:
            element = get_random_are_sequence()
        
        # Limit element size for intergenic insertion
        element = element[:min(len(element), 40)]
        
        # Insert at random position in the middle of intergenic region
        buffer = 50  # Stay away from edges
        if region_length > len(element) + 2 * buffer:
            insert_pos = start + buffer + random.randint(0, region_length - len(element) - 2*buffer)
            for i, base in enumerate(element):
                if insert_pos + i < end:
                    sequence[insert_pos + i] = base

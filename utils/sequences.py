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
    "CCCCGATTGGGGGCGACACTCCACCATC",
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

def mutate_sequence(sequence: str, mutation_prob: float = 0.1) -> str:
    """
    Randomly mutate 1-2 bases in a sequence with given probability.
    
    Args:
        sequence: DNA sequence to potentially mutate
        mutation_prob: Probability of mutation (default 10%)
        
    Returns:
        Potentially mutated sequence
    """
    if random.random() > mutation_prob:
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


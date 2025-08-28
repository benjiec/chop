#!/usr/bin/env python3
"""
Controlled UTR Pattern Detection Dataset.

This creates a highly controlled test:
- 5% UTR5 elements (realistic sequences)
- 5% UTR3 elements (realistic sequences) 
- 90% INTERGENIC (all Cs - simple background)

This tests if transformers can learn sequence specificity when the background
is trivial and the patterns are clear.
"""

import random
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Tuple, List
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from tests.gene_prediction.realistic_utr_sequences import (
    KOZAK_SEQUENCES, UTR5_REAL_SEQUENCES, IRES_SEQUENCES,
    POLYA_SIGNALS, ARE_SEQUENCES, GRE_SEQUENCES, UTR3_REAL_SEQUENCES, UTR3_COMPLETE_SEQUENCES,
    mutate_sequence
)

class ControlledUTRDataset(Dataset):
    """
    Controlled dataset for testing UTR pattern recognition.
    
    Background is all Cs (trivial), UTR elements are realistic sequences.
    This removes any confounding factors and tests pure pattern recognition.
    """
    
    def __init__(self, sequence_length: int = 1000, num_sequences: int = 400, 
                 utr5_density: float = 0.05, utr3_density: float = 0.05):
        """
        Args:
            sequence_length: Length of each DNA sequence
            num_sequences: Number of sequences to generate  
            utr5_density: Fraction of positions that should be UTR5 (default 5%)
            utr3_density: Fraction of positions that should be UTR3 (default 5%)
        """
        self.sequence_length = sequence_length
        self.num_sequences = num_sequences
        self.utr5_density = utr5_density
        self.utr3_density = utr3_density
        
        # Collect all UTR sequences
        self.utr5_sequences = KOZAK_SEQUENCES + UTR5_REAL_SEQUENCES + IRES_SEQUENCES
        self.utr3_sequences = POLYA_SIGNALS + ARE_SEQUENCES + GRE_SEQUENCES + UTR3_REAL_SEQUENCES + UTR3_COMPLETE_SEQUENCES
        
        # Generate all sequences and targets upfront
        self.sequences = []
        self.targets = []
        
        print(f"Generating {num_sequences} controlled sequences...")
        print(f"  Target composition: {utr5_density:.1%} UTR5, {utr3_density:.1%} UTR3, {1-utr5_density-utr3_density:.1%} INTERGENIC (all Cs)")
        
        utr5_counts = []
        utr3_counts = []
        
        for i in range(num_sequences):
            seq, targets = self._generate_sequence()
            self.sequences.append(seq)
            self.targets.append(targets)
            
            # Count UTR elements in this sequence
            utr5_count = np.sum(targets == 1)
            utr3_count = np.sum(targets == 2)
            utr5_counts.append(utr5_count)
            utr3_counts.append(utr3_count)
            
            if (i + 1) % 50 == 0:
                print(f"  Generated {i + 1}/{num_sequences} sequences")
        
        # Calculate statistics
        total_positions = num_sequences * sequence_length
        total_utr5_positions = sum(utr5_counts)
        total_utr3_positions = sum(utr3_counts)
        total_intergenic = total_positions - total_utr5_positions - total_utr3_positions
        
        print(f"Dataset statistics:")
        print(f"  Total positions: {total_positions}")
        print(f"  UTR5 positions: {total_utr5_positions} ({total_utr5_positions/total_positions:.3f})")
        print(f"  UTR3 positions: {total_utr3_positions} ({total_utr3_positions/total_positions:.3f})")
        print(f"  INTERGENIC positions: {total_intergenic} ({total_intergenic/total_positions:.3f})")
        print(f"  UTR5 elements per sequence: min={min(utr5_counts) if utr5_counts else 0}, max={max(utr5_counts) if utr5_counts else 0}, avg={sum(utr5_counts)/len(utr5_counts) if utr5_counts else 0:.1f}")
        print(f"  UTR3 elements per sequence: min={min(utr3_counts) if utr3_counts else 0}, max={max(utr3_counts) if utr3_counts else 0}, avg={sum(utr3_counts)/len(utr3_counts) if utr3_counts else 0:.1f}")
    
    def _generate_sequence(self) -> Tuple[str, np.ndarray]:
        """Generate a single controlled DNA sequence with UTR elements on C background."""
        
        # Start with all Cs (simple intergenic background)
        sequence = ['C'] * self.sequence_length
        
        # Create targets (all INTERGENIC initially)
        targets = np.zeros(self.sequence_length, dtype=np.int64)
        
        # Calculate how many UTR elements to place
        expected_utr5_positions = int(self.sequence_length * self.utr5_density)
        expected_utr3_positions = int(self.sequence_length * self.utr3_density)
        
        # Estimate number of elements based on average UTR lengths
        avg_utr5_length = 25  # Approximate average length
        avg_utr3_length = 20  # Approximate average length
        
        num_utr5_elements = max(1, expected_utr5_positions // avg_utr5_length)
        num_utr3_elements = max(1, expected_utr3_positions // avg_utr3_length)
        
        # Add some randomness but keep it controlled
        num_utr5_elements = random.randint(max(1, num_utr5_elements - 1), num_utr5_elements + 2)
        num_utr3_elements = random.randint(max(1, num_utr3_elements - 1), num_utr3_elements + 2)
        
        # Place UTR5 elements
        placed_positions = []
        for _ in range(num_utr5_elements):
            # Choose a random UTR5 sequence
            utr5_seq = random.choice(self.utr5_sequences)
            # Potentially mutate it (10% chance)
            utr5_seq = mutate_sequence(utr5_seq, mutation_rate=0.1)
            
            # Find a position that doesn't overlap with existing elements
            attempts = 0
            while attempts < 100:  # More attempts for controlled placement
                pos = random.randint(0, self.sequence_length - len(utr5_seq))
                
                # Check for overlap with sufficient buffer
                overlap = any(abs(pos - existing_pos) < max(len(utr5_seq), existing_len) + 5 
                            for existing_pos, existing_len, _ in placed_positions)
                if not overlap:
                    # Place the UTR5 element
                    for i, base in enumerate(utr5_seq):
                        if pos + i < self.sequence_length:
                            sequence[pos + i] = base
                            targets[pos + i] = 1  # UTR5 class
                    
                    placed_positions.append((pos, len(utr5_seq), 1))
                    break
                attempts += 1
        
        # Place UTR3 elements
        for _ in range(num_utr3_elements):
            # Choose a random UTR3 sequence
            utr3_seq = random.choice(self.utr3_sequences)
            # Potentially mutate it (10% chance)
            utr3_seq = mutate_sequence(utr3_seq, mutation_rate=0.1)
            
            # Find a position that doesn't overlap with existing elements
            attempts = 0
            while attempts < 100:  # More attempts for controlled placement
                pos = random.randint(0, self.sequence_length - len(utr3_seq))
                
                # Check for overlap with sufficient buffer
                overlap = any(abs(pos - existing_pos) < max(len(utr3_seq), existing_len) + 5 
                            for existing_pos, existing_len, _ in placed_positions)
                if not overlap:
                    # Place the UTR3 element
                    for i, base in enumerate(utr3_seq):
                        if pos + i < self.sequence_length:
                            sequence[pos + i] = base
                            targets[pos + i] = 2  # UTR3 class
                    
                    placed_positions.append((pos, len(utr3_seq), 2))
                    break
                attempts += 1
        
        return ''.join(sequence), targets
    
    def __len__(self) -> int:
        return self.num_sequences
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sequence = self.sequences[idx]
        targets = self.targets[idx]
        
        # Encode DNA sequence to integers
        dna_vocab = {'A': 0, 'T': 1, 'G': 2, 'C': 3, 'N': 4}
        encoded_seq = np.array([dna_vocab.get(base, 4) for base in sequence])
        
        return torch.tensor(encoded_seq, dtype=torch.long), torch.tensor(targets, dtype=torch.long)

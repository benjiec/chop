#!/usr/bin/env python3
"""
Simple ATG detection dataset for testing basic pattern learning.

This creates synthetic DNA sequences where:
- ATG positions are labeled as START (class 1)
- All other positions are labeled as INTERGENIC (class 0)

This is the simplest possible test to see if the model can learn basic sequence patterns.
"""

import random
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Tuple

class SimpleATGDataset(Dataset):
    """
    Dataset that generates DNA sequences with ATG codons labeled as START.
    
    This is designed to test if the transformer model can learn the most basic pattern:
    ATG = START, everything else = INTERGENIC.
    
    Background can be random DNA or uniform (all Cs) for controlled experiments.
    """
    
    def __init__(self, sequence_length: int = 1000, num_sequences: int = 1000, 
                 atg_density: float = 0.25, background: str = 'random'):
        """
        Args:
            sequence_length: Length of each DNA sequence
            num_sequences: Number of sequences to generate
            atg_density: Approximate fraction of positions that should be ATG starts
            background: 'random' for random DNA, 'uniform' for all Cs
        """
        self.sequence_length = sequence_length
        self.num_sequences = num_sequences
        self.atg_density = atg_density
        self.background = background
        
        # Generate all sequences and targets upfront
        self.sequences = []
        self.targets = []
        
        print(f"Generating {num_sequences} synthetic sequences...")
        atg_counts = []
        for i in range(num_sequences):
            seq, targets = self._generate_sequence()
            self.sequences.append(seq)
            self.targets.append(targets)
            
            # Count ATGs in this sequence
            atg_count = seq.count('ATG')
            atg_counts.append(atg_count)
            
            if (i + 1) % 100 == 0:
                print(f"  Generated {i + 1}/{num_sequences} sequences")
        
        # Calculate statistics
        total_positions = num_sequences * sequence_length
        total_start_positions = sum(np.sum(targets == 1) for targets in self.targets)
        actual_density = total_start_positions / total_positions
        
        print(f"Dataset statistics:")
        print(f"  Total positions: {total_positions}")
        print(f"  START positions: {total_start_positions} ({actual_density:.3f})")
        print(f"  INTERGENIC positions: {total_positions - total_start_positions} ({1-actual_density:.3f})")
        print(f"  ATG codons per sequence: min={min(atg_counts)}, max={max(atg_counts)}, avg={sum(atg_counts)/len(atg_counts):.1f}")
        print(f"  Total ATG codons: {sum(atg_counts)}")
    
    def _generate_sequence(self) -> Tuple[str, np.ndarray]:
        """Generate a single DNA sequence with ATG labels."""
        
        # Create background based on configuration
        if self.background == 'uniform':
            # All Cs for controlled experiments
            sequence = ['C'] * self.sequence_length
        else:
            # Random DNA (default)
            bases = ['A', 'T', 'G', 'C']
            sequence = [random.choice(bases) for _ in range(self.sequence_length)]
            
            # Remove any accidental ATGs by replacing the middle base
            for i in range(self.sequence_length - 2):
                if sequence[i] == 'A' and sequence[i + 1] == 'T' and sequence[i + 2] == 'G':
                    sequence[i + 1] = random.choice(['A', 'G', 'C'])  # Change T to avoid ATG
        
        # Create targets (all INTERGENIC initially)
        targets = np.zeros(self.sequence_length, dtype=np.int64)
        
        # Decide how many ATG codons to place
        expected_atgs = int(self.sequence_length * self.atg_density / 3)  # Divide by 3 since ATG is 3 bases
        num_atgs = random.randint(max(1, expected_atgs // 2), expected_atgs * 2)
        
        # Place ATG codons at random positions (ensure no overlap)
        atg_positions = []
        attempts = 0
        while len(atg_positions) < num_atgs and attempts < num_atgs * 10:
            pos = random.randint(0, self.sequence_length - 3)
            # Check if this position would overlap with existing ATGs
            overlap = any(abs(pos - existing_pos) < 3 for existing_pos in atg_positions)
            if not overlap:
                atg_positions.append(pos)
            attempts += 1
        
        for pos in atg_positions:
            # Place ATG at this position
            sequence[pos] = 'A'
            sequence[pos + 1] = 'T' 
            sequence[pos + 2] = 'G'
            
            # Label all 3 bases of ATG as START (class 1)
            targets[pos] = 1     # A
            targets[pos + 1] = 1 # T
            targets[pos + 2] = 1 # G
        
        # Debug: count actual ATGs in final sequence for verification
        sequence_str = ''.join(sequence)
        actual_atg_count = sequence_str.count('ATG')
        
        return sequence_str, targets
    
    def __len__(self) -> int:
        return self.num_sequences
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sequence = self.sequences[idx]
        targets = self.targets[idx]
        
        # Encode DNA sequence to integers
        dna_vocab = {'A': 0, 'T': 1, 'G': 2, 'C': 3, 'N': 4}
        encoded_seq = np.array([dna_vocab.get(base, 4) for base in sequence])
        
        return torch.tensor(encoded_seq, dtype=torch.long), torch.tensor(targets, dtype=torch.long)

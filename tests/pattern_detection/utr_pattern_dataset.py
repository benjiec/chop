#!/usr/bin/env python3
"""
UTR Pattern Detection Dataset for testing complex pattern learning.

This creates synthetic DNA sequences with realistic UTR elements:
- 5' UTR sequences (Kozak, IRES, etc.)
- 3' UTR sequences (Poly-A signals, AREs, GREs)
- Intergenic sequences (random DNA)

Classes:
- 0: INTERGENIC (random DNA)
- 1: UTR5 (5' UTR elements)
- 2: UTR3 (3' UTR elements)

This tests if transformers can learn complex, statistical patterns with variation
(as opposed to the simple deterministic ATG detection).
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

class UTRPatternDataset(Dataset):
    """
    Dataset that generates random DNA sequences with realistic UTR elements.
    
    This tests if transformers can learn complex, statistical patterns with biological variation.
    """
    
    def __init__(self, sequence_length: int = 1000, num_sequences: int = 1000, utr_density: float = 0.3):
        """
        Args:
            sequence_length: Length of each DNA sequence
            num_sequences: Number of sequences to generate  
            utr_density: Approximate fraction of positions that should be UTR elements
        """
        self.sequence_length = sequence_length
        self.num_sequences = num_sequences
        self.utr_density = utr_density
        
        # Collect all UTR sequences
        self.utr5_sequences = KOZAK_SEQUENCES + UTR5_REAL_SEQUENCES + IRES_SEQUENCES
        self.utr3_sequences = POLYA_SIGNALS + ARE_SEQUENCES + GRE_SEQUENCES + UTR3_REAL_SEQUENCES + UTR3_COMPLETE_SEQUENCES
        
        # Generate all sequences and targets upfront
        self.sequences = []
        self.targets = []
        
        print(f"Generating {num_sequences} synthetic sequences with UTR patterns...")
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
            
            if (i + 1) % 100 == 0:
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
        """Generate a single random DNA sequence with UTR elements."""
        
        # Start with random DNA
        bases = ['A', 'T', 'G', 'C']
        sequence = [random.choice(bases) for _ in range(self.sequence_length)]
        
        # Create targets (all INTERGENIC initially)
        targets = np.zeros(self.sequence_length, dtype=np.int64)
        
        # Calculate how many UTR elements to place
        expected_utr_positions = int(self.sequence_length * self.utr_density)
        # Split roughly equally between UTR5 and UTR3
        expected_utr5_elements = max(1, expected_utr_positions // 40)  # Assume avg length ~20bp
        expected_utr3_elements = max(1, expected_utr_positions // 40)
        
        # Add some randomness
        num_utr5_elements = random.randint(max(1, expected_utr5_elements // 2), expected_utr5_elements * 2)
        num_utr3_elements = random.randint(max(1, expected_utr3_elements // 2), expected_utr3_elements * 2)
        
        # Place UTR5 elements
        placed_positions = []
        for _ in range(num_utr5_elements):
            # Choose a random UTR5 sequence
            utr5_seq = random.choice(self.utr5_sequences)
            # Potentially mutate it
            utr5_seq = mutate_sequence(utr5_seq, mutation_rate=0.1)
            
            # Find a position that doesn't overlap with existing elements
            attempts = 0
            while attempts < 50:  # Avoid infinite loops
                pos = random.randint(0, self.sequence_length - len(utr5_seq))
                
                # Check for overlap
                overlap = any(abs(pos - existing_pos) < max(len(utr5_seq), 10) for existing_pos, _, _ in placed_positions)
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
            # Potentially mutate it
            utr3_seq = mutate_sequence(utr3_seq, mutation_rate=0.1)
            
            # Find a position that doesn't overlap with existing elements
            attempts = 0
            while attempts < 50:  # Avoid infinite loops
                pos = random.randint(0, self.sequence_length - len(utr3_seq))
                
                # Check for overlap
                overlap = any(abs(pos - existing_pos) < max(len(utr3_seq), 10) for existing_pos, _, _ in placed_positions)
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

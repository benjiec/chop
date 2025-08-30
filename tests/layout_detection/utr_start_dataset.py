#!/usr/bin/env python3
"""
UTR-START Layout Dataset for testing biological context learning.

This creates sequences with realistic gene layouts:
- 500bp background (with decoy ATGs)
- 5' UTR sequence 
- ATG START codon (real gene start)
- 500bp background (with decoy ATGs, but not within 30bp of real ATG)

The goal is to test if models can learn that ATGs following UTRs are real STARTs,
while ATGs in random background are just intergenic noise.

Layout per contig:
- 10 layouts per contig
- Each layout: [500bp background] [UTR5] [ATG] [500bp background]
- Total contig length: ~10 * (500 + UTR_length + 3 + 500) = ~10kb per contig
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
    mutate_sequence
)

class UTRStartDataset(Dataset):
    """
    Dataset for testing UTR-START context learning.
    
    Creates sequences where real START codons follow 5' UTRs,
    while decoy ATGs appear in random background regions.
    """
    
    def __init__(self, num_contigs: int = 20, layouts_per_contig: int = 10, 
                 background_length: int = 500, sequence_length: int = 2000):
        """
        Args:
            num_contigs: Number of contigs to generate (default 20)
            layouts_per_contig: Number of UTR-START layouts per contig (default 10)
            background_length: Length of background regions (default 500bp)
            sequence_length: Target length for each contig sequence (default 2000bp)
        """
        self.num_contigs = num_contigs
        self.layouts_per_contig = layouts_per_contig
        self.background_length = background_length
        self.sequence_length = sequence_length
        
        # Collect all UTR5 sequences
        self.utr5_sequences = KOZAK_SEQUENCES + UTR5_REAL_SEQUENCES + IRES_SEQUENCES
        
        # Generate all sequences and targets upfront
        self.sequences = []
        self.targets = []
        
        print(f"Generating {num_contigs} contigs with UTR-START layouts...")
        print(f"  {layouts_per_contig} layouts per contig")
        print(f"  Background regions: {background_length}bp each")
        print(f"  Target sequence length: {sequence_length}bp")
        
        layout_counts = []
        start_counts = []
        decoy_atg_counts = []
        
        for contig_idx in range(num_contigs):
            seq, targets, stats = self._generate_contig()
            self.sequences.append(seq)
            self.targets.append(targets)
            
            layout_counts.append(stats['layouts'])
            start_counts.append(stats['real_starts'])
            decoy_atg_counts.append(stats['decoy_atgs'])
            
            if (contig_idx + 1) % 5 == 0:
                print(f"  Generated {contig_idx + 1}/{num_contigs} contigs")
        
        # Calculate statistics
        total_positions = sum(len(seq) for seq in self.sequences)
        total_starts = sum(np.sum(targets == 2) for targets in self.targets)
        total_utr5 = sum(np.sum(targets == 1) for targets in self.targets)
        total_intergenic = total_positions - total_starts - total_utr5
        
        print(f"Dataset statistics:")
        print(f"  Total contigs: {num_contigs}")
        print(f"  Total positions: {total_positions}")
        print(f"  START positions: {total_starts} ({total_starts/total_positions:.3f})")
        print(f"  UTR5 positions: {total_utr5} ({total_utr5/total_positions:.3f})")
        print(f"  INTERGENIC positions: {total_intergenic} ({total_intergenic/total_positions:.3f})")
        print(f"  Layouts per contig: min={min(layout_counts)}, max={max(layout_counts)}, avg={sum(layout_counts)/len(layout_counts):.1f}")
        print(f"  Real STARTs per contig: min={min(start_counts)}, max={max(start_counts)}, avg={sum(start_counts)/len(start_counts):.1f}")
        print(f"  Decoy ATGs per contig: min={min(decoy_atg_counts)}, max={max(decoy_atg_counts)}, avg={sum(decoy_atg_counts)/len(decoy_atg_counts):.1f}")
    
    def _generate_contig(self) -> Tuple[str, np.ndarray, dict]:
        """Generate a single contig with multiple UTR-START layouts."""
        
        # Start with empty sequence
        full_sequence = []
        full_targets = []
        
        layouts_created = 0
        real_starts = 0
        decoy_atgs = 0
        
        for layout_idx in range(self.layouts_per_contig):
            # Generate one layout: [background] [UTR5] [ATG] [background]
            layout_seq, layout_targets, layout_stats = self._generate_layout()
            
            full_sequence.extend(layout_seq)
            full_targets.extend(layout_targets)
            
            layouts_created += 1
            real_starts += layout_stats['real_starts']
            decoy_atgs += layout_stats['decoy_atgs']
        
        # Trim or pad to target length
        if len(full_sequence) > self.sequence_length:
            full_sequence = full_sequence[:self.sequence_length]
            full_targets = full_targets[:self.sequence_length]
        elif len(full_sequence) < self.sequence_length:
            # Pad with random background
            padding_needed = self.sequence_length - len(full_sequence)
            padding_seq, padding_targets, padding_stats = self._generate_background(
                padding_needed, include_decoy_atgs=True
            )
            full_sequence.extend(padding_seq)
            full_targets.extend(padding_targets)
            decoy_atgs += padding_stats['decoy_atgs']
        
        sequence_str = ''.join(full_sequence)
        targets_array = np.array(full_targets, dtype=np.int64)
        
        stats = {
            'layouts': layouts_created,
            'real_starts': real_starts,
            'decoy_atgs': decoy_atgs
        }
        
        return sequence_str, targets_array, stats
    
    def _generate_layout(self) -> Tuple[List[str], List[int], dict]:
        """Generate one UTR-START layout."""
        
        # 1. First background region (500bp with decoy ATGs)
        bg1_seq, bg1_targets, bg1_stats = self._generate_background(
            self.background_length, include_decoy_atgs=True
        )
        
        # 2. UTR5 sequence
        utr5_seq = random.choice(self.utr5_sequences)
        utr5_seq = mutate_sequence(utr5_seq, mutation_rate=0.1)
        
        # Check if UTR5 ends with ATG (Kozak sequences do)
        if utr5_seq.endswith('ATG'):
            # Use the ATG from UTR5 as the START
            utr5_targets = [1] * (len(utr5_seq) - 3) + [2, 2, 2]  # UTR5 + START
            real_start_length = 3
        else:
            # Add ATG after UTR5
            utr5_seq += 'ATG'
            utr5_targets = [1] * len(utr5_seq[:-3]) + [2, 2, 2]  # UTR5 + START
            real_start_length = 3
        
        # 3. Second background region (500bp with decoy ATGs, but not near real START)
        bg2_seq, bg2_targets, bg2_stats = self._generate_background(
            self.background_length, include_decoy_atgs=True, exclude_near_start=30
        )
        
        # Combine layout
        layout_seq = bg1_seq + list(utr5_seq) + bg2_seq
        layout_targets = bg1_targets + utr5_targets + bg2_targets
        
        stats = {
            'real_starts': 1,
            'decoy_atgs': bg1_stats['decoy_atgs'] + bg2_stats['decoy_atgs']
        }
        
        return layout_seq, layout_targets, stats
    
    def _generate_background(self, length: int, include_decoy_atgs: bool = False, 
                           exclude_near_start: int = 0) -> Tuple[List[str], List[int], dict]:
        """Generate random background sequence."""
        
        bases = ['A', 'T', 'G', 'C']
        sequence = [random.choice(bases) for _ in range(length)]
        targets = [0] * length  # All INTERGENIC
        
        decoy_atgs = 0
        
        if include_decoy_atgs:
            # Ensure at least 1 ATG in this background region
            min_atgs = 1
            max_atgs = max(1, length // 100)  # ~1 ATG per 100bp
            num_atgs = random.randint(min_atgs, max_atgs)
            
            # Place decoy ATGs
            for _ in range(num_atgs):
                attempts = 0
                while attempts < 50:
                    pos = random.randint(0, length - 3)
                    
                    # Skip if too close to start (for second background)
                    if exclude_near_start > 0 and pos < exclude_near_start:
                        attempts += 1
                        continue
                    
                    # Place ATG but keep as INTERGENIC (decoy)
                    sequence[pos] = 'A'
                    sequence[pos + 1] = 'T'
                    sequence[pos + 2] = 'G'
                    # targets remain [0, 0, 0] - these are decoy ATGs
                    
                    decoy_atgs += 1
                    break
                    
                attempts += 1
        
        stats = {'decoy_atgs': decoy_atgs}
        return sequence, targets, stats
    
    def __len__(self) -> int:
        return self.num_contigs
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sequence = self.sequences[idx]
        targets = self.targets[idx]
        
        # Encode DNA sequence to integers
        dna_vocab = {'A': 0, 'T': 1, 'G': 2, 'C': 3, 'N': 4}
        encoded_seq = np.array([dna_vocab.get(base, 4) for base in sequence])
        
        return torch.tensor(encoded_seq, dtype=torch.long), torch.tensor(targets, dtype=torch.long)

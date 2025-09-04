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

from utils.sequences import (
    KOZAK_SEQUENCES, UTR5_REAL_SEQUENCES, IRES_SEQUENCES,
    mutate_sequence
)
from utils.constants import DNAEmbed

class UTRStartDataset(Dataset):
    """
    Dataset for testing UTR-START context learning.
    
    Creates full-length contigs with UTR-START layouts, then provides
    sliding windows for training. Each training sample is a window
    of fixed length that slides across the full contigs.
    """
    
    def __init__(self, num_contigs: int = 1000, layouts_per_contig: int = 1, 
                 background_length: int = 500, window_size: int = 1100, window_stride: int = 1100,
                 max_seq_length: int = None, one_window_per_contig: bool = False):
        """
        Args:
            num_contigs: Number of contigs to generate (default 1000)
            layouts_per_contig: Number of UTR-START layouts per contig (default 1)
            background_length: Length of background regions (default 500bp)
            window_size: Size of sliding windows for training (default 1100bp)
            window_stride: Stride between windows (default 1100bp - no overlap)
        """
        self.num_contigs = num_contigs
        self.layouts_per_contig = layouts_per_contig
        self.background_length = background_length
        self.window_size = window_size
        self.window_stride = window_stride
        self.one_window_per_contig = one_window_per_contig
        # Ensure samples fit within model length: use (max_seq_length - 1) as strict cap for window
        self.max_window_len = None
        if max_seq_length is not None and max_seq_length > 0:
            self.max_window_len = max_seq_length - 1
            # Align window to cap
            self.window_size = min(self.window_size, self.max_window_len)
        
        # Collect all UTR5 sequences
        self.utr5_sequences = KOZAK_SEQUENCES + UTR5_REAL_SEQUENCES + IRES_SEQUENCES
        
        # Generate all contigs (full length) and calculate windows
        self.contigs = []  # Store full-length contigs
        self.contig_targets = []  # Store full-length targets
        self.windows = []  # Store (contig_idx, start_pos) for each window
        
        print(f"Generating {num_contigs} contigs with UTR-START layouts...")
        print(f"  {layouts_per_contig} layouts per contig")
        print(f"  Background regions: {background_length}bp each")
        print(f"  Window size: {self.window_size}bp, stride: {self.window_stride}bp")
        
        layout_counts = []
        start_counts = []
        decoy_atg_counts = []
        total_windows = 0
        
        for contig_idx in range(num_contigs):
            seq, targets, stats = self._generate_contig()
            self.contigs.append(seq)
            self.contig_targets.append(targets)
            
            layout_counts.append(stats['layouts'])
            start_counts.append(stats['real_starts'])
            decoy_atg_counts.append(stats['decoy_atgs'])
            
            # Calculate windows for this contig
            contig_length = len(seq)

            if self.one_window_per_contig:
                # Exactly one window per contig
                self.windows.append((contig_idx, 0))
                total_windows += 1
            else:
                # Sliding windows as usual
                if contig_length >= self.window_size:
                    for start_pos in range(0, contig_length - self.window_size + 1, self.window_stride):
                        self.windows.append((contig_idx, start_pos))
                        total_windows += 1
                else:
                    self.windows.append((contig_idx, 0))
                    total_windows += 1
            
            if (contig_idx + 1) % 5 == 0:
                print(f"  Generated {contig_idx + 1}/{num_contigs} contigs")
        
        # Calculate statistics
        total_contig_positions = sum(len(seq) for seq in self.contigs)
        total_starts = sum(np.sum(targets == 2) for targets in self.contig_targets)
        total_utr5 = sum(np.sum(targets == 1) for targets in self.contig_targets)
        total_intergenic = total_contig_positions - total_starts - total_utr5
        
        print(f"Dataset statistics:")
        print(f"  Total contigs: {num_contigs}")
        print(f"  Contig positions: {total_contig_positions}")
        print(f"  Training windows: {total_windows}")
        print(f"  START positions: {total_starts} ({total_starts/total_contig_positions:.3f})")
        print(f"  UTR5 positions: {total_utr5} ({total_utr5/total_contig_positions:.3f})")
        print(f"  INTERGENIC positions: {total_intergenic} ({total_intergenic/total_contig_positions:.3f})")
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
        
        # No trimming - keep full contig length for sliding windows
        
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
        
        # Determine dynamic background sizes to fit within max window length if configured
        dynamic_bg = None
        
        # 2. UTR5 sequence
        utr5_seq = random.choice(self.utr5_sequences)
        utr5_seq = mutate_sequence(utr5_seq, mutation_prob=0.1)
        
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
        
        # If we have a max window cap and one-window mode or single layout, size backgrounds to fit
        if self.max_window_len is not None and (self.one_window_per_contig or self.layouts_per_contig == 1):
            remaining = max(0, self.max_window_len - (len(utr5_seq) + 3))
            bg1_len = remaining // 2
            bg2_len = remaining - bg1_len
            dynamic_bg = (bg1_len, bg2_len)
        
        # 1. First background region
        bg1_len = dynamic_bg[0] if dynamic_bg else self.background_length
        bg1_seq, bg1_targets, bg1_stats = self._generate_background(
            bg1_len, include_decoy_atgs=True
        )

        # 3. Second background region (avoid ATGs too close to start)
        bg2_len = dynamic_bg[1] if dynamic_bg else self.background_length
        bg2_seq, bg2_targets, bg2_stats = self._generate_background(
            bg2_len, include_decoy_atgs=True, exclude_near_start=30
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
        return len(self.windows)  # Number of sliding windows, not contigs
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Get window information
        contig_idx, start_pos = self.windows[idx]
        
        # Extract window from full contig
        full_sequence = self.contigs[contig_idx]
        full_targets = self.contig_targets[contig_idx]
        
        # Get window slice
        end_pos = start_pos + self.window_size
        window_sequence = full_sequence[start_pos:end_pos]
        window_targets = full_targets[start_pos:end_pos]
        
        # Pad if window is shorter than expected (for simplified contigs)
        if len(window_sequence) < self.window_size:
            padding_needed = self.window_size - len(window_sequence)
            # Pad with random bases and INTERGENIC labels
            bases = ['A', 'T', 'G', 'C']
            padding_seq = ''.join([bases[i % 4] for i in range(padding_needed)])
            padding_targets = [0] * padding_needed  # INTERGENIC
            
            window_sequence += padding_seq
            window_targets = list(window_targets) + padding_targets
        
        # Encode DNA sequence to integers
        dna_vocab = {'A': DNAEmbed.A, 'T': DNAEmbed.T, 'G': DNAEmbed.G, 'C': DNAEmbed.C, 'N': DNAEmbed.N}
        encoded_seq = np.array([dna_vocab.get(base, 4) for base in window_sequence])
        
        return torch.tensor(encoded_seq, dtype=torch.long), torch.tensor(window_targets, dtype=torch.long)

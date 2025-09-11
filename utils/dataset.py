#!/usr/bin/env python3

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
from utils.constants import DNAEmbed, GenePredictionClass as P


class SequenceSegmentGeneratorBase(object):

    def generate(self, last_segment_sequence) -> Tuple[str, List[int]]:
        """Should return a new sequence and targets """
        pass


class GenomicSyntheticTestingDataset(Dataset):
    """
    Dataset to make synthetic testing data to minic genomic sequences.
    """

    def __init__(self,
                 max_sequence_length: int,
                 num_contigs: int,
                 layouts_per_contig: int,
                 layouts: List[SequenceSegmentGeneratorBase]):

        self.num_contigs = num_contigs
        self.layouts_per_contig = layouts_per_contig
        # Support multiple layout variants: accept either a single list of segment
        # generators or a list of lists (each inner list is a layout variant).
        if len(layouts) > 0 and isinstance(layouts[0], list):
            # type: ignore[unreachable]
            self.layout_variants: List[List[SequenceSegmentGeneratorBase]] = layouts  # type: ignore[assignment]
        else:
            # Single layout provided
            self.layout_variants = [layouts]  # type: ignore[list-item]

        self.max_sequence_length = max_sequence_length
        self.contigs = []  # Store full-length contigs
        self.contig_targets = []  # Store full-length targets

        for contig_idx in range(num_contigs):
            # Round-robin select layout variant
            current_layout = self.layout_variants[contig_idx % len(self.layout_variants)]
            seq, targets = self._generate_contig(current_layout)
            self.contigs.append(seq)
            self.contig_targets.append(targets)
            assert len(seq) <= max_sequence_length, f"Each contig cannot be larger than {max_sequence_length}"

        # Calculate statistics
        total_contig_positions = sum(len(seq) for seq in self.contigs)
        print(f"total contigs: {num_contigs}")

        # Shuffle contigs to avoid ordering bias across layout variants
        order = list(range(len(self.contigs)))
        random.shuffle(order)
        self.contigs = [self.contigs[i] for i in order]
        self.contig_targets = [self.contig_targets[i] for i in order]

    def _generate_contig(self, layout: List[SequenceSegmentGeneratorBase]) -> Tuple[str, np.ndarray]:
        full_sequence = []
        full_targets = []

        for layout_idx in range(self.layouts_per_contig):
            layout_seq, layout_targets = self._generate_layout(layout)
            full_sequence.append(layout_seq)
            full_targets.extend(layout_targets)

        sequence_str = ''.join(full_sequence)
        targets_array = np.array(full_targets, dtype=np.int64)
        return sequence_str, targets_array

    def _generate_layout(self, layout: List[SequenceSegmentGeneratorBase]) -> Tuple[str, List[int]]:
        sequence = []
        targets = []        

        last_sequence = None
        for segment_generator in layout:
            seq, tar = segment_generator.generate(last_sequence)
            sequence.append(seq)
            targets.extend(tar)
            last_sequence = seq

        return "".join(sequence), targets

    def __len__(self) -> int:
        return len(self.contigs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sequence = self.contigs[idx]
        targets = list(self.contig_targets[idx])

        # Pad to fixed length for batching
        if len(sequence) < self.max_sequence_length:
            pad_len = self.max_sequence_length - len(sequence)
            sequence = sequence + ('N' * pad_len)
            targets = targets + [P.INTERGENIC] * pad_len

        # Encode DNA sequence to integers
        dna_vocab = {'A': DNAEmbed.A, 'T': DNAEmbed.T, 'G': DNAEmbed.G, 'C': DNAEmbed.C, 'N': DNAEmbed.N}
        encoded_seq = np.array([dna_vocab.get(base, 4) for base in sequence])
        return torch.tensor(encoded_seq, dtype=torch.long), torch.tensor(np.array(targets, dtype=np.int64), dtype=torch.long)


class RandomBasesGenerator(SequenceSegmentGeneratorBase):

    def __init__(self, length, target: int = P.INTERGENIC, decoy: str = None, max_decoy: int = None, random_min_length: int = None, avoid: str = None):
        self.length = length
        self.target = target
        self.decoy = decoy
        self.max_decoy = max_decoy
        self.random_min_length = random_min_length
        self.avoid = avoid

    def generate(self, _) -> Tuple[str, List[int]]:
        bases = ['A', 'T', 'G', 'C']
        length = self.length
        if self.random_min_length is not None:
            length = random.randint(self.random_min_length, self.length)
        sequence = [random.choice(bases) for _ in range(length)]

        if self.max_decoy:
            assert self.decoy and len(self.decoy)
            min_decoy = 1
            max_decoy = max(1, self.max_decoy)
            num_decoys = random.randint(min_decoy, max_decoy)

            for _ in range(num_decoys):
                pos = random.randint(0, length - len(self.decoy))
                for j in range(0, len(self.decoy)):
                    sequence[pos+j] = self.decoy[j]

        sequence = "".join(sequence)
        if self.avoid:
            while self.avoid in sequence:
                sequence = sequence.replace(self.avoid, "")

        return sequence, [self.target] * len(sequence)
 

class RandomChoiceGenerator(SequenceSegmentGeneratorBase):

    def __init__(self, choices: List[str], target, mutation_prob: float = None):
        self.choices = choices
        self.target = target
        self.mutation_prob = mutation_prob

    def generate(self, _) -> Tuple[str, List[int]]:
        sequence = random.choice(self.choices)
        if self.mutation_prob:
            sequence = mutate_sequence(sequence, mutation_prob=self.mutation_prob)
        return sequence, [self.target] * len(sequence)


class RandomUTR5Generator(RandomChoiceGenerator):

    def generate(self, _) -> Tuple[str, List[int]]:
        sequence, targets = super().generate(_)
        if sequence.upper().endswith("ATG"):
            targets[-3] = P.START
            targets[-2] = P.START
            targets[-1] = P.START
        return sequence, targets


class AddATGGenerator(SequenceSegmentGeneratorBase):

    def generate(self, last_segment_sequence) -> Tuple[str, List[int]]:
        # Check if last_segment_sequence ends with ATG (e.g. Kozak sequences do)
        if last_segment_sequence and last_segment_sequence.upper().endswith('ATG'):
            return "", []
        else:
            return "ATG", [P.START] * 3

#!/usr/bin/env python3

from typing import List, Tuple, Dict, Optional, Set
from dataclasses import dataclass
from pathlib import Path
import gzip

import numpy as np
import random

from utils.constants import (
    GenePredictionClass as P,
    DNAEmbed,
    ConventionalStopCodons,
    ConventionalDonorDinucleotides,
    ConventionalAcceptorDinucleotides,
)
from utils.windowing import compute_window_slices


@dataclass
class GeneAnnotation:
    sequence_id: str
    gene_id: str
    strand: str  # '+' or '-'
    exons: List[Tuple[int, int]]  # 0-based half-open [start, end)


def _reverse_complement(seq: str) -> str:
    comp = str.maketrans('ATGCNatgcn', 'TACGNtacgn')
    return seq.translate(comp)[::-1]


def _load_fasta(fasta_path: str) -> Dict[str, str]:
    records: Dict[str, str] = {}
    sid = None
    buf: List[str] = []
    # Support both plain text and gzip-compressed FASTA files
    path_obj = Path(fasta_path)
    is_gz = any(suf == '.gz' for suf in path_obj.suffixes)
    open_fn = gzip.open if is_gz else open
    mode = 'rt' if is_gz else 'r'
    with open_fn(fasta_path, mode) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                if sid is not None:
                    records[sid] = ''.join(buf).upper()
                sid = line[1:].split()[0]
                buf = []
            else:
                buf.append(line)
        if sid is not None:
            records[sid] = ''.join(buf).upper()
    return records


def _parse_tsv_annotations(tsv_path: str) -> List[GeneAnnotation]:
    # Row-per-exon schema: sequence_id, gene_id, gene_start, gene_end, exon_start, exon_end, strand
    genes: Dict[Tuple[str, str, str], List[Tuple[int, int]]] = {}
    with open(tsv_path, 'r') as f:
        header = f.readline().strip().split('\t')
        def _idx(name: str) -> int:
            for i, h in enumerate(header):
                if h.lower() == name.lower():
                    return i
            raise ValueError(f"Missing column '{name}' in TSV header: {header}")
        sid_i = _idx('sequence_id')
        gid_i = _idx('gene_id')
        strand_i = _idx('strand')
        xs_i = _idx('exon_start')
        xe_i = _idx('exon_end')
        for line in f:
            parts = line.rstrip('\n').split('\t')
            if len(parts) <= max(sid_i, gid_i, strand_i, xs_i, xe_i):
                continue
            sid = parts[sid_i]
            gid = parts[gid_i]
            strand = parts[strand_i]
            # Convert 1-based inclusive to 0-based half-open
            try:
                start0 = int(parts[xs_i]) - 1
                end0_excl = int(parts[xe_i])  # inclusive->exclusive
            except Exception:
                continue
            key = (sid, gid, strand)
            genes.setdefault(key, []).append((start0, end0_excl))
    anns: List[GeneAnnotation] = []
    for (sid, gid, strand), exs in genes.items():
        exons = sorted(exs, key=lambda t: t[0])
        anns.append(GeneAnnotation(sequence_id=sid, gene_id=gid, strand=strand, exons=exons))
    return anns


def _encode_sequence(seq: str) -> np.ndarray:
    vocab = {'A': DNAEmbed.A, 'T': DNAEmbed.T, 'G': DNAEmbed.G, 'C': DNAEmbed.C, 'N': DNAEmbed.N}
    return np.array([vocab.get(ch, DNAEmbed.N) for ch in seq], dtype=np.int64)


class AnnotatedGenomeDataset:
    def __init__(self, fasta_path: str, annotations_tsv_path: str, window: Optional[int] = None, stride: Optional[int] = None,
                 num_windows: Optional[int] = None, class_weights: Optional[List[float]] = None, seed: int = 17):
        self.fasta_records = _load_fasta(fasta_path)
        self.annotations = _parse_tsv_annotations(annotations_tsv_path)
        self.sequences: List[str] = []
        self.targets: List[np.ndarray] = []
        self.contig_ids: List[str] = []
        # Optional windowing
        self.window: Optional[int] = int(window) if window and int(window) > 0 else None
        self.stride: Optional[int] = int(stride) if stride and int(stride) > 0 else None
        self.windows: List[Tuple[int, int, int]] = []  # (contig_idx, start, end)
        # Sampling/accounting options
        self.num_windows: Optional[int] = int(num_windows) if num_windows is not None else None
        self.class_weights: Optional[List[float]] = list(class_weights) if class_weights is not None else None
        self.seed: int = int(seed)
        # Derived/accounting structures
        self.classset_to_contigs: Dict[frozenset, List[str]] = {}
        self._window_class_sets: List[Set[int]] = []
        self._selected_window_indices: Optional[List[int]] = None
        print("Building/sampling windows for training")
        self._build()

    def __len__(self) -> int:
        if self.window:
            if self._selected_window_indices is not None:
                return len(self._selected_window_indices)
            return len(self.windows)
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        if self.window:
            if self._selected_window_indices is not None:
                real_idx = self._selected_window_indices[idx]
            else:
                real_idx = idx
            contig_idx, start, end = self.windows[real_idx]
            seq = self.sequences[contig_idx][start:end]
            tgt = self.targets[contig_idx][start:end]
            return _encode_sequence(seq), tgt
        else:
            seq = self.sequences[idx]
            tgt = self.targets[idx]
            return _encode_sequence(seq), tgt

    def _build(self):
        for ann in self.annotations:
            if ann.sequence_id not in self.fasta_records:
                continue
            seq = self.fasta_records[ann.sequence_id]
            L = len(seq)
            tgt = np.full(L, P.INTERGENIC, dtype=np.int64)

            # START/STOP based on half-open exons
            if ann.strand == '+':
                first_exon = ann.exons[0]
                last_exon = ann.exons[-1]
                start_pos = first_exon[0]
                stop_pos = last_exon[1] - 3
                start_codon = seq[start_pos:start_pos+3]
                stop_codon = seq[stop_pos:stop_pos+3]
                assert start_codon == 'ATG', f"Expected ATG at + strand START: got {start_codon}"
                assert stop_codon in ConventionalStopCodons, f"Expected STOP at + strand: got {stop_codon}"
                tgt[start_pos:start_pos+3] = P.START
                tgt[stop_pos:stop_pos+3] = P.STOP
                coding_lo = start_pos
                coding_hi_excl = last_exon[1]
            else:
                # minus strand: START at last exon end, STOP at first exon start (reverse complemented)
                first_exon = ann.exons[0]
                last_exon = ann.exons[-1]
                start_pos = last_exon[1] - 3
                stop_pos = first_exon[0]
                start_codon = _reverse_complement(seq[start_pos:last_exon[1]])
                stop_codon = _reverse_complement(seq[stop_pos:stop_pos+3])
                assert start_codon == 'ATG', f"Expected ATG at - strand START: got {start_codon}"
                assert stop_codon in ConventionalStopCodons, f"Expected STOP at - strand: got {stop_codon}"
                tgt[start_pos:last_exon[1]] = P.START
                tgt[stop_pos:stop_pos+3] = P.STOP
                coding_lo = first_exon[0]
                coding_hi_excl = last_exon[1]

            # GENE region between START and STOP default to GENE; refine DSS/ASS
            # Determine coding span (inclusive) by strand
            if ann.strand == '+':
                coding_lo = start_pos
                coding_hi = stop_pos + 2
            else:
                coding_lo = stop_pos
                coding_hi = first_exon[1] - 1
            tgt[coding_lo:coding_hi+1] = np.where(tgt[coding_lo:coding_hi+1] == P.INTERGENIC, P.GENE, tgt[coding_lo:coding_hi+1])

            # Intron labeling using genomic order of exons
            exon_list_sorted = sorted(ann.exons, key=lambda t: t[0])
            for i in range(len(exon_list_sorted) - 1):
                e1 = exon_list_sorted[i]
                e2 = exon_list_sorted[i + 1]
                if ann.strand == '+':
                    donor_pos = e1[1]
                    acceptor_pos = e2[0] - 2
                    if 0 <= donor_pos+1 < L:
                        donor_di = seq[donor_pos:donor_pos+2]
                        assert donor_di in ConventionalDonorDinucleotides, f"Donor not in {ConventionalDonorDinucleotides}: {donor_di}"
                        tgt[donor_pos:donor_pos+2] = P.DSS
                    if 0 <= acceptor_pos+1 < L:
                        acceptor_di = seq[acceptor_pos:acceptor_pos+2]
                        assert acceptor_di in ConventionalAcceptorDinucleotides, f"Acceptor not in {ConventionalAcceptorDinucleotides}: {acceptor_di}"
                        tgt[acceptor_pos:acceptor_pos+2] = P.ASS
                else:
                    donor_pos = e2[0] - 2
                    acceptor_pos = e1[1]
                    if 0 <= donor_pos and donor_pos+1 < L:
                        donor_di = _reverse_complement(seq[donor_pos:donor_pos+2])
                        assert donor_di in ConventionalDonorDinucleotides, f"Donor(-) not in {ConventionalDonorDinucleotides}: {donor_di}"
                        tgt[donor_pos:donor_pos+2] = P.DSS
                    if 0 <= acceptor_pos and acceptor_pos+1 < L:
                        acceptor_di = _reverse_complement(seq[acceptor_pos:acceptor_pos+2])
                        assert acceptor_di in ConventionalAcceptorDinucleotides, f"Acceptor(-) not in {ConventionalAcceptorDinucleotides}: {acceptor_di}"
                        tgt[acceptor_pos:acceptor_pos+2] = P.ASS

            self.sequences.append(seq)
            self.targets.append(tgt)
            self.contig_ids.append(ann.sequence_id)

        # If windowing is enabled, precompute windows over each contig
        if self.window:
            win = int(self.window)
            st = int(self.stride) if self.stride else max(1, win // 2)
            for contig_idx, seq in enumerate(self.sequences):
                L = len(seq)
                slices = compute_window_slices(L, window=win, stride=st)
                for s, e in slices:
                    self.windows.append((contig_idx, s, e))

            # Build accounting for classes per window and classset->contigs map
            self._compute_window_accounting_and_sampling()

    def _compute_window_accounting_and_sampling(self) -> None:
        # Classes to consider for balancing: exclude any with weight == 1.0 if provided
        weights = self.class_weights
        exclude_weight_one: Set[int] = set()
        if weights is not None:
            for i, w in enumerate(weights):
                if w == 1.0:
                    exclude_weight_one.add(i)

        rng = random.Random(self.seed)

        # Compute class set for each window
        self._window_class_sets = []
        class_to_windows: Dict[int, List[int]] = {}
        for w_idx, (contig_idx, s, e) in enumerate(self.windows):
            tgt_slice = self.targets[contig_idx][s:e]
            present = set(int(c) for c in np.unique(tgt_slice))
            present_considered = {c for c in present if c not in exclude_weight_one}
            self._window_class_sets.append(present_considered)
            for c in present_considered:
                class_to_windows.setdefault(c, []).append(w_idx)

        # Build classset->contigs mapping (order-insensitive)
        classset_to_contigs: Dict[frozenset, Set[str]] = {}
        for w_idx, cls_set in enumerate(self._window_class_sets):
            key = frozenset(cls_set)
            contig_idx, _, _ = self.windows[w_idx]
            contig_id = self.contig_ids[contig_idx]
            classset_to_contigs.setdefault(key, set()).add(contig_id)
        # Convert sets to sorted lists for determinism
        self.classset_to_contigs = {k: sorted(list(v)) for k, v in classset_to_contigs.items()}

        # If no sampling requested, we're done
        if self.num_windows is None:
            self._selected_window_indices = None
            return

        target_num = max(0, int(self.num_windows))
        total_available = len(self.windows)
        if target_num >= total_available:
            self._selected_window_indices = list(range(total_available))
            return

        # Determine classes to balance
        classes_to_balance: List[int]
        if weights is not None:
            classes_to_balance = [i for i, w in enumerate(weights) if w != 1.0]
        else:
            # All classes observed across windows
            observed: Set[int] = set()
            for s in self._window_class_sets:
                observed.update(s)
            classes_to_balance = sorted(list(observed))

        # Greedy round-robin selection to balance per-class counts
        selected: Set[int] = set()
        per_class_count: Dict[int, int] = {c: 0 for c in classes_to_balance}

        def deficit(cls: int) -> int:
            # Desired average count grows with selection; use current min target
            return min(per_class_count.values()) - per_class_count[cls]

        # Pre-shuffle candidate lists deterministically
        for c in classes_to_balance:
            rng.shuffle(class_to_windows.get(c, []))

        while len(selected) < target_num:
            # Choose the class with the smallest count so far (tie-broken deterministically)
            cls_order = sorted(classes_to_balance, key=lambda c: (per_class_count[c], c))
            picked_window = None
            for cls in cls_order:
                candidates = [w for w in class_to_windows.get(cls, []) if w not in selected]
                if not candidates:
                    continue
                # Prefer candidate window that improves balance across included classes
                def score(wi: int) -> Tuple[int, int, int]:
                    included = self._window_class_sets[wi]
                    # Higher score for covering more underrepresented classes
                    underrep = sum(1 for c in included if c in per_class_count and per_class_count[c] == min(per_class_count.values()))
                    # Fewer already well-represented classes
                    overrep = sum(1 for c in included if c in per_class_count and per_class_count[c] > min(per_class_count.values()))
                    # Tie-breaker: prefer smaller index for determinism after RNG shuffle
                    return (underrep, -overrep, -wi)
                candidates.sort(key=score, reverse=True)
                picked_window = candidates[0]
                break

            if picked_window is None:
                break

            selected.add(picked_window)
            for c in self._window_class_sets[picked_window]:
                if c in per_class_count:
                    per_class_count[c] += 1

        # If still short (e.g., no informative windows), fill with remaining windows deterministically
        if len(selected) < target_num:
            remaining = [i for i in range(total_available) if i not in selected]
            # Deterministic order
            selected_list = list(selected)
            remaining.sort()
            needed = target_num - len(selected_list)
            selected_list.extend(remaining[:needed])
            self._selected_window_indices = selected_list
        else:
            self._selected_window_indices = sorted(list(selected))

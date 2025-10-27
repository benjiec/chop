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
    StandardDonorDinucleotides,
    DinoDonorDinucleotides,
    ConventionalAcceptorDinucleotides,
)
from utils.windowing import compute_window_slices
from torch.utils.data import Subset
from utils.sequences import reverse_complement
from utils.stream import NumericalStream, load_fasta


def build_class_windows(
    windows: List[Tuple[int, int, int]],
    targets: List[np.ndarray],
    classes_to_balance: List[int],
    exclude_margin_bps: Optional[int],
    class_weights: Optional[List[float]] = None,
) -> Dict[int, List[int]]:
    """Build mapping from class index to list of window indices.

    For each window, iterate target tokens within the center region excluding
    edges defined by ``exclude_margin_bps``. Consider only classes in
    ``classes_to_balance``; assign the window to the present class with the
    highest weight (ties broken by smaller class id). Windows with no relevant
    classes present are skipped.
    """
    class_windows: Dict[int, List[int]] = {}
    if not windows:
        return class_windows

    classes_set = set(int(c) for c in classes_to_balance)
    for w_idx, (contig_idx, s, e) in enumerate(windows):
        tgt = targets[contig_idx]
        Lw = e - s
        if Lw <= 0:
            continue
        margin = int(exclude_margin_bps) if exclude_margin_bps is not None else 0
        inner_start = s + margin
        inner_end = e - margin
        if inner_start >= inner_end:
            # No central region to evaluate
            continue

        # Track highest-weight class present in the center region
        top_class: Optional[int] = None
        top_weight: float = float('-inf')
        for pos in range(inner_start, inner_end):
            cls = int(tgt[pos])
            if cls not in classes_set:
                continue
            w = 1.0
            if class_weights is not None:
                cw = class_weights[cls]
                w = 1.0 if cw is None else float(cw)
            if (w > top_weight) or (w == top_weight and (top_class is None or cls < top_class)):
                top_weight = w
                top_class = cls

        if top_class is None:
            # No relevant classes present in the center
            continue

        assigned_cls = top_class
        class_windows.setdefault(assigned_cls, []).append(w_idx)

    return class_windows


# Removed old balanced_select_from_class_windows helper; balancing with recycling
# is handled directly inside _sample_windows.


@dataclass
class GeneAnnotation:
    sequence_id: str
    gene_id: str
    strand: str  # '+' or '-'
    exons: List[Tuple[int, int]]  # 0-based half-open [start, end)


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


def build_targets_for_annotation(
    seq: str,
    ann: GeneAnnotation,
    gene_class: int,
    allow_nonconforming_start: bool = False,
    allow_nonconforming_stop: bool = False,
    allow_nonconforming_ass: bool = False,
    allow_nonconforming_dss: bool = False,
    failure_counts: Optional[Dict[str, int]] = None,
) -> Optional[np.ndarray]:
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
        start_ok = (start_codon == 'ATG')
        stop_ok = (stop_codon in ConventionalStopCodons)
        if not start_ok:
            if failure_counts is not None:
                failure_counts['start'] = failure_counts.get('start', 0) + 1
            if not allow_nonconforming_start:
                return None
        if not stop_ok:
            if failure_counts is not None:
                failure_counts['stop'] = failure_counts.get('stop', 0) + 1
            if not allow_nonconforming_stop:
                return None
        tgt[start_pos:start_pos+3] = P.START
        tgt[stop_pos:stop_pos+3] = P.STOP
    else:
        # minus strand: START at last exon end, STOP at first exon start (reverse complemented)
        first_exon = ann.exons[0]
        last_exon = ann.exons[-1]
        start_pos = last_exon[1] - 3
        stop_pos = first_exon[0]
        start_codon = reverse_complement(seq[start_pos:last_exon[1]])
        stop_codon = reverse_complement(seq[stop_pos:stop_pos+3])
        start_ok = (start_codon == 'ATG')
        stop_ok = (stop_codon in ConventionalStopCodons)
        if not start_ok:
            if failure_counts is not None:
                failure_counts['start'] = failure_counts.get('start', 0) + 1
            if not allow_nonconforming_start:
                return None
        if not stop_ok:
            if failure_counts is not None:
                failure_counts['stop'] = failure_counts.get('stop', 0) + 1
            if not allow_nonconforming_stop:
                return None
        tgt[start_pos:last_exon[1]] = P.START
        tgt[stop_pos:stop_pos+3] = P.STOP

    # GENE region between START and STOP default to GENE; refine DSS/ASS
    # Determine coding span (inclusive) by strand
    if ann.strand == '+':
        coding_lo = start_pos
        coding_hi = stop_pos + 2
    else:
        coding_lo = stop_pos
        coding_hi = first_exon[1] - 1
    tgt[coding_lo:coding_hi+1] = np.where(
        tgt[coding_lo:coding_hi+1] == P.INTERGENIC,
        gene_class,
        tgt[coding_lo:coding_hi+1]
    )

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
                donor_ok = (donor_di in StandardDonorDinucleotides.union(DinoDonorDinucleotides))
                if not donor_ok:
                    if failure_counts is not None:
                        failure_counts['dss'] = failure_counts.get('dss', 0) + 1
                    if not allow_nonconforming_dss:
                        return None
                tgt[donor_pos:donor_pos+2] = P.DSS
            if 0 <= acceptor_pos+1 < L:
                acceptor_di = seq[acceptor_pos:acceptor_pos+2]
                acceptor_ok = (acceptor_di in ConventionalAcceptorDinucleotides)
                if not acceptor_ok:
                    if failure_counts is not None:
                        failure_counts['ass'] = failure_counts.get('ass', 0) + 1
                    if not allow_nonconforming_ass:
                        return None
                tgt[acceptor_pos:acceptor_pos+2] = P.ASS
        else:
            donor_pos = e2[0] - 2
            acceptor_pos = e1[1]
            if 0 <= donor_pos and donor_pos+1 < L:
                donor_di = reverse_complement(seq[donor_pos:donor_pos+2])
                donor_ok = (donor_di in StandardDonorDinucleotides.union(DinoDonorDinucleotides))
                if not donor_ok:
                    if failure_counts is not None:
                        failure_counts['dss'] = failure_counts.get('dss', 0) + 1
                    if not allow_nonconforming_dss:
                        return None
                tgt[donor_pos:donor_pos+2] = P.DSS
            if 0 <= acceptor_pos and acceptor_pos+1 < L:
                acceptor_di = reverse_complement(seq[acceptor_pos:acceptor_pos+2])
                acceptor_ok = (acceptor_di in ConventionalAcceptorDinucleotides)
                if not acceptor_ok:
                    if failure_counts is not None:
                        failure_counts['ass'] = failure_counts.get('ass', 0) + 1
                    if not allow_nonconforming_ass:
                        return None
                tgt[acceptor_pos:acceptor_pos+2] = P.ASS

    return tgt


def add_random_n_prefix(
    seq: str,
    tgt: np.ndarray,
    enabled: bool,
    rng: Optional[object] = None,
    min_len: int = 100,
    max_len: int = 400,
) -> Tuple[str, np.ndarray, int]:
    pad_len = 0
    if enabled:
        r = rng if rng is not None else random
        pad_len = r.randint(int(min_len), int(max_len))
    if pad_len > 0:
        seq = ('N' * pad_len) + seq
        pad_tgt = np.full(pad_len, P.INTERGENIC, dtype=np.int64)
        tgt = np.concatenate([pad_tgt, tgt], axis=0)
    return seq, tgt, pad_len


class AnnotatedGenomeDataset:
    def __init__(self, fasta_path: str, annotations_tsv_path: str,
                 num_contigs: Optional[int] = None, window: Optional[int] = None, stride: Optional[int] = None,
                 num_windows: Optional[int] = None, class_weights: Optional[List[float]] = None,
                 window_boost_classes: Optional[List[int]] = None,
                 exclude_margin_bps: Optional[int] = 200,
                 gene_class: int = P.INTERGENIC,
                 random_prefix_ns: bool = True,
                 allow_nonconforming_start: bool = False,
                 allow_nonconforming_stop: bool = False,
                 allow_nonconforming_ass: bool = False,
                 allow_nonconforming_dss: bool = False,
                 aux_stream_path: Optional[str] = None,
                 aux_normalize: bool = True,
                 ):
        self.fasta_records = load_fasta(fasta_path)
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
        # Derived/accounting structures
        self._selected_window_indices: Optional[List[int]] = None
        self._window_boost_classes = window_boost_classes
        self._exclude_margin_bps = int(exclude_margin_bps) if exclude_margin_bps is not None else None
        # Class to use for coding region between START and STOP (default: INTERGENIC)
        self.gene_class: int = int(gene_class)
        self._random_prefix_ns: bool = bool(random_prefix_ns)
        self.allow_nonconforming_start = bool(allow_nonconforming_start)
        self.allow_nonconforming_stop = bool(allow_nonconforming_stop)
        self.allow_nonconforming_ass = bool(allow_nonconforming_ass)
        self.allow_nonconforming_dss = bool(allow_nonconforming_dss)
        self.motif_fail_counts: Dict[str, int] = {"start": 0, "stop": 0, "ass": 0, "dss": 0}
        # Aux
        self.aux_stream_path: Optional[str] = str(aux_stream_path) if aux_stream_path else None
        self.aux_normalize: bool = bool(aux_normalize)
        self.aux_by_contig: List[Optional[np.ndarray]] = []
        self._num_stream: Optional[NumericalStream] = None
        self._has_any_aux: bool = False

        # Load aux raw first
        if self.aux_stream_path:
            ns = NumericalStream(self.aux_stream_path)
            # Early validation: aux lengths must match original FASTA lengths
            for sid in ns.sequence_ids:
                orig = len(self.fasta_records.get(sid, ""))
                if int(orig) <= 0:
                    raise ValueError(f"aux stream sequence_id not found in FASTA: {sid}")
                if int(ns.get(sid).shape[0]) != int(orig):
                    raise ValueError(f"aux stream length mismatch for {sid}: aux={int(ns.get(sid).shape[0])} vs original_fasta={int(orig)}")
            if self.aux_normalize:
                print("Normalizing aux stream")
                ns.normalize()
            self._num_stream = ns
            self._has_any_aux = True

        self._build(num_contigs)

    def __len__(self) -> int:
        if self.window:
            return len(self._selected_window_indices)
        return len(self.sequences)

    def __getitem__(self, idx: int):
        if self.window:
            real_idx = self._selected_window_indices[idx]
            contig_idx, start, end = self.windows[real_idx]
            seq = self.sequences[contig_idx][start:end]
            tgt = self.targets[contig_idx][start:end]
            aux: Optional[np.ndarray] = None
            a_full = self.aux_by_contig[contig_idx]
            if a_full is not None:
                a = a_full[start:end, :]
                aux = a.copy()
            # Pad to fixed window length for batching
            win_len = int(self.window)
            cur_len = end - start
            if cur_len < win_len:
                pad = win_len - cur_len
                if pad > 0:
                    seq = seq + ('N' * pad)
                    tgt = np.concatenate([tgt, np.full((pad,), P.INTERGENIC, dtype=tgt.dtype)])
                    if aux is not None:
                        C = int(aux.shape[1])
                        aux = np.concatenate([aux, np.zeros((pad, C), dtype=aux.dtype)], axis=0)
            if self._has_any_aux:
                return _encode_sequence(seq), tgt, aux
            else:
                return _encode_sequence(seq), tgt
        else:
            seq = self.sequences[idx]
            tgt = self.targets[idx]
            if self._has_any_aux:
                aux = self.aux_by_contig[idx]
                return _encode_sequence(seq), tgt, aux
            else:
                return _encode_sequence(seq), tgt

    def _build(self, num_contigs: Optional[int] = 0):

        by_contig_aux: List[Optional[np.ndarray]] = []
        for ann in self.annotations:
            if ann.sequence_id not in self.fasta_records:
                continue
            seq = self.fasta_records[ann.sequence_id]

            tgt = build_targets_for_annotation(
                seq=seq,
                ann=ann,
                gene_class=self.gene_class,
                allow_nonconforming_start=self.allow_nonconforming_start,
                allow_nonconforming_stop=self.allow_nonconforming_stop,
                allow_nonconforming_ass=self.allow_nonconforming_ass,
                allow_nonconforming_dss=self.allow_nonconforming_dss,
                failure_counts=self.motif_fail_counts,
            )

            if tgt is None:
                continue

            seq, tgt, pad_len = add_random_n_prefix(
                seq=seq,
                tgt=tgt,
                enabled=self._random_prefix_ns,
            )

            if self._num_stream is not None:
                # Validate aux length equals original FASTA length when present; synthesize zeros if missing
                raw_len = len(self.fasta_records.get(ann.sequence_id, ""))
                try:
                    aux_raw = self._num_stream.get(ann.sequence_id)
                    if int(aux_raw.shape[0]) != int(raw_len):
                        raise ValueError(f"aux stream length mismatch for {ann.sequence_id}: aux={int(aux_raw.shape[0])} vs original_fasta={int(raw_len)}")
                    arr = self._num_stream.pad(ann.sequence_id, pad_len)
                except KeyError:
                    # Missing contig in stream: create zeros with same channels and apply left prefix padding
                    C = int(len(self._num_stream.channels)) if self._num_stream.channels is not None else 0
                    base = np.zeros((raw_len, C), dtype=np.float32)
                    if pad_len > 0:
                        pad = np.zeros((pad_len, C), dtype=np.float32)
                        arr = np.concatenate([pad, base], axis=0)
                    else:
                        arr = base
                by_contig_aux.append(arr.astype(np.float32, copy=False))
            else:
                by_contig_aux.append(None)

            self.sequences.append(seq)
            self.targets.append(tgt)
            self.contig_ids.append(ann.sequence_id)

        if num_contigs and num_contigs > 0:
            self.sequences = self.sequences[0:num_contigs]
            self.targets = self.targets[0:num_contigs]
            self.contig_ids = self.contig_ids[0:num_contigs]
            by_contig_aux = by_contig_aux[0:num_contigs]

        self.aux_by_contig = by_contig_aux

        print(len(self.sequences), "sequences")
        # If windowing is enabled, precompute windows over each contig
        if self.window:
            win = int(self.window)
            st = int(self.stride) if self.stride else max(1, win // 2)
            for contig_idx, seq in enumerate(self.sequences):
                L = len(seq)
                slices = compute_window_slices(L, window=win, stride=st)
                # Exclude windows that don't contain any targets with weight > 1
                weights = self.class_weights
                for s, e in slices:
                    if weights is not None:
                        tgt_seg = self.targets[contig_idx][s:e]
                        # any class index with weight > 1 present?
                        has_weighted = False
                        for cls_idx in np.unique(tgt_seg):
                            w = float(weights[int(cls_idx)])
                            if w > 1.0:
                                has_weighted = True
                                break
                        if not has_weighted:
                            continue
                    self.windows.append((contig_idx, s, e))
            print(len(self.windows), "windows")

            # Build accounting for classes per window and classset->contigs map, then sample
            self._sample_windows()

    def _sample_windows(self) -> None:
        # Classes to consider for balancing: exclude any with weight == 1.0 if provided
        weights = self.class_weights
        exclude_weight_one: Set[int] = set()
        if weights is not None:
            for i, w in enumerate(weights):
                if w == 1.0:
                    exclude_weight_one.add(i)

        rng = random

        # If no sampling requested, select all windows
        if self.num_windows is None and self._window_boost_classes is None:
            print("not sampling windows, not boosting windows")
            self._selected_window_indices = list(range(len(self.windows)))
            return

        # Determine classes to balance
        if weights is not None:
            classes_to_balance = [i for i, w in enumerate(weights) if w != 1.0]
        else:
            # If no explicit weights, consider indices based on available weights length when present; otherwise default to [0]
            classes_to_balance = list(range(len(weights))) if weights else [0]
        if self._window_boost_classes is not None:
            classes_to_balance = [i for i in classes_to_balance if i in self._window_boost_classes]

        # Build class->windows assignment using center-only counting and max-weight rule
        class_windows = build_class_windows(
            windows=self.windows,
            targets=self.targets,
            classes_to_balance=classes_to_balance,
            exclude_margin_bps=self._exclude_margin_bps,
            class_weights=self.class_weights,
        )

        # If specific classes are requested, restrict candidates to allowed classes only
        if self._window_boost_classes is not None and self.num_windows is None:
            print("not sampling windows, boosting windows")
            self._selected_window_indices = list(range(len(self.windows)))
            allowed = set(self._window_boost_classes)
            for c, lst in class_windows.items():
                if c in allowed:
                    print("boost", c, "by", len(lst))
                    self._selected_window_indices.extend(lst)
            return

        target_num = max(0, int(self.num_windows))
        total_available = len(self.windows)
        if target_num >= total_available:
            self._selected_window_indices = list(range(total_available))
            return

        # Determine per-class target and perform recycling-based balancing via list replication
        nonempty = {c: list(lst) for c, lst in class_windows.items() if lst}
        if not nonempty:
            self._selected_window_indices = []
            return

        num_classes = len(nonempty)
        per_class_target = max(1, target_num // num_classes)

        selected_list: List[int] = []
        for c in sorted(nonempty.keys()):
            lst = list(nonempty[c])
            if len(lst) == 0:
                continue
            if len(lst) < per_class_target:
                from math import ceil
                k = int(ceil(per_class_target / float(len(lst))))
                print("taking", per_class_target, "for class", c, "after expanding by", k, "from", len(lst))
                expanded = (lst * k)[:per_class_target]
                selected_list.extend(expanded)
            else:
                print("taking", per_class_target, "for class", c)
                selected_list.extend(lst[:per_class_target])

        # If total is short due to rounding, fill from union
        if len(selected_list) < target_num:
            union_pool: List[int] = []
            for c in sorted(nonempty.keys()):
                union_pool.extend(nonempty[c])
            need = target_num - len(selected_list)
            selected_list.extend(union_pool[:need])

        # Final shuffle for batch-level distribution
        rng.shuffle(selected_list)
        self._selected_window_indices = selected_list[:target_num]

    def split(self, train_size: int, val_size: int):
        total = len(self)
        if train_size + val_size != total:
            # scale proportionally if sizes sum to ratio
            ratio = (train_size + val_size)
            if ratio > 0:
                train_size = int(round(total * (train_size / ratio)))
                val_size = total - train_size
            else:
                train_size = total
                val_size = 0
        # If windowing is enabled, split by contigs to avoid overlapping windows across splits
        if self.window:
            # Always split using positions into the selected list
            contig_to_positions: Dict[int, List[int]] = {}
            for pos, w_idx in enumerate(self._selected_window_indices):
                contig_idx, _, _ = self.windows[w_idx]
                contig_to_positions.setdefault(contig_idx, []).append(pos)
            # Sort contigs by group size (largest first) for stable, group-wise assignment
            contig_ids = sorted(contig_to_positions.keys(), key=lambda cid: len(contig_to_positions[cid]), reverse=True)
            train_positions: List[int] = []
            val_positions: List[int] = []
            for cid in contig_ids:
                group = contig_to_positions[cid]
                # Greedy assignment: push the next whole contig to the split that is currently lowest and can be filled
                # Avoid trimming; accept slight size mismatch to keep contigs disjoint
                train_gap = max(0, train_size - len(train_positions))
                val_gap = max(0, val_size - len(val_positions))
                if len(val_positions) < len(train_positions) and val_gap > 0:
                    val_positions.extend(group)
                else:
                    train_positions.extend(group)
            return Subset(self, train_positions), Subset(self, val_positions)
        else:
            # No windowing; split by sequence index deterministically
            idxs = list(range(len(self.sequences)))
            random.shuffle(idxs)
            train_idx = idxs[:train_size]
            val_idx = idxs[train_size:train_size + val_size]
            return Subset(self, train_idx), Subset(self, val_idx)



#!/usr/bin/env python3

from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from pathlib import Path

import numpy as np

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
    with open(fasta_path, 'r') as f:
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
    def __init__(self, fasta_path: str, annotations_tsv_path: str, window: Optional[int] = None, stride: Optional[int] = None):
        self.fasta_records = _load_fasta(fasta_path)
        self.annotations = _parse_tsv_annotations(annotations_tsv_path)
        self.sequences: List[str] = []
        self.targets: List[np.ndarray] = []
        # Optional windowing
        self.window: Optional[int] = int(window) if window and int(window) > 0 else None
        self.stride: Optional[int] = int(stride) if stride and int(stride) > 0 else None
        self.windows: List[Tuple[int, int, int]] = []  # (contig_idx, start, end)
        self._build()

    def __len__(self) -> int:
        if self.window:
            return len(self.windows)
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        if self.window:
            contig_idx, start, end = self.windows[idx]
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

        # If windowing is enabled, precompute windows over each contig
        if self.window:
            win = int(self.window)
            st = int(self.stride) if self.stride else max(1, win // 2)
            for contig_idx, seq in enumerate(self.sequences):
                L = len(seq)
                slices = compute_window_slices(L, window=win, stride=st)
                for s, e in slices:
                    self.windows.append((contig_idx, s, e))

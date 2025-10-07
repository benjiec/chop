#!/usr/bin/env python3

from typing import Dict, List, Tuple, Optional, Set

from dataclasses import dataclass


@dataclass
class _Candidate:
    strand: str
    exons: List[Tuple[int, int]]  # 0-based half-open
    boundary_score: float
    start_rank: int


def _idx(header: List[str], name: str) -> int:
    lname = name.lower()
    for i, h in enumerate(header):
        if h.lower() == lname:
            return i
    raise ValueError(f"Missing column '{name}' in TSV header: {header}")


def _parse_decoded_tsv(decoded_tsv: str) -> Dict[str, Dict[int, List[_Candidate]]]:
    """Parse decoder TSV into mapping: sequence_id -> start_pos(1-based) -> list of candidates.

    A candidate aggregates rows for the same (sequence_id, gene_start, gene_id) and stores
    - strand
    - exons as 0-based half-open
    - boundary_score
    - start_rank
    """
    by_seq: Dict[str, Dict[int, Dict[str, _Candidate]]] = {}
    with open(decoded_tsv, 'r') as f:
        header = f.readline().strip().split('\t')
        sid_i = _idx(header, 'sequence_id')
        gid_i = _idx(header, 'gene_id')
        gstart_i = _idx(header, 'gene_start')
        xs_i = _idx(header, 'exon_start')
        xe_i = _idx(header, 'exon_end')
        strand_i = _idx(header, 'strand')
        bnd_i = _idx(header, 'boundary_score')
        srank_i = _idx(header, 'start_rank')

        for line in f:
            parts = line.rstrip('\n').split('\t')
            if len(parts) <= max(sid_i, gid_i, gstart_i, xs_i, xe_i, strand_i, bnd_i, srank_i):
                continue
            sid = parts[sid_i]
            gid = parts[gid_i]
            try:
                start1 = int(parts[gstart_i])
                exon_s1 = int(parts[xs_i])
                exon_e1 = int(parts[xe_i])
                # convert to 0-based half-open
                exon_s0 = exon_s1 - 1
                exon_e0 = exon_e1
                strand = parts[strand_i]
                boundary = float(parts[bnd_i]) if parts[bnd_i] != '' else 0.0
                start_rank = int(parts[srank_i])
            except Exception:
                continue

            seq_map = by_seq.setdefault(sid, {})
            start_map = seq_map.setdefault(start1, {})
            cand = start_map.get(gid)
            if cand is None:
                cand = _Candidate(strand=strand, exons=[], boundary_score=boundary, start_rank=start_rank)
                start_map[gid] = cand
            cand.exons.append((exon_s0, exon_e0))
            # keep first boundary/start_rank encountered; values should be consistent per gene_id

    # sort exons within each candidate and convert mapping to list
    out: Dict[str, Dict[int, List[_Candidate]]] = {}
    for sid, starts in by_seq.items():
        out_starts: Dict[int, List[_Candidate]] = {}
        for start1, gid_map in starts.items():
            lst = list(gid_map.values())
            for c in lst:
                c.exons.sort(key=lambda t: t[0])
            out_starts[start1] = lst
        out[sid] = out_starts
    return out


def _parse_expected(expected_tsv: str):
    # Reuse existing annotation loader to avoid duplication
    from utils.genome import _parse_tsv_annotations
    anns = _parse_tsv_annotations(expected_tsv)
    by_seq: Dict[str, List[Tuple[str, Tuple[Tuple[int, int], ...]]]] = {}
    for ann in anns:
        # ann.exons already 0-based half-open and sorted in loader
        key = (ann.strand, tuple(ann.exons))
        by_seq.setdefault(ann.sequence_id, []).append(key)
    return by_seq


def _parse_expected_starts(expected_tsv: str) -> Dict[str, Set[Tuple[str, int]]]:
    """Parse expected TSV to collect strand-aware gene START positions (1-based) per sequence_id."""
    starts_by_seq: Dict[str, Set[Tuple[str, int]]] = {}
    with open(expected_tsv, 'r') as f:
        header = f.readline().strip().split('\t')
        sid_i = _idx(header, 'sequence_id')
        gstart_i = _idx(header, 'gene_start')
        strand_i = _idx(header, 'strand')
        for line in f:
            parts = line.rstrip('\n').split('\t')
            if len(parts) <= max(sid_i, gstart_i, strand_i):
                continue
            sid = parts[sid_i]
            strand = parts[strand_i]
            try:
                start1 = int(parts[gstart_i])
            except Exception:
                continue
            starts_by_seq.setdefault(sid, set()).add((strand, start1))
    return starts_by_seq


def _pick_top_k_starts(starts: Dict[int, List[_Candidate]], topk_starts: int) -> List[int]:
    """Rank starts by boundary_score of best start_rank==1 candidate (fallback to best boundary).
    Return the list of selected start positions (1-based). Exactly topk_starts, or fewer if not enough starts.
    """
    scored: List[Tuple[float, int]] = []
    for start1, cands in starts.items():
        # prefer rank==1; fallback to min rank; then fallback to max boundary
        best = None
        best_rank = None
        for c in cands:
            if c.start_rank == 1:
                best = c
                best_rank = 1
                break
        if best is None and cands:
            # choose candidate with smallest start_rank
            best = min(cands, key=lambda x: (x.start_rank, -x.boundary_score))
            best_rank = best.start_rank
        score = best.boundary_score if best is not None else float('-inf')
        scored.append((score, start1))
    scored.sort(key=lambda t: t[0], reverse=True)
    selected = [s for _, s in scored[:max(0, int(topk_starts))]]
    return selected


def _compute_counts(tp: int, fp: int, fn: int) -> Dict[str, float]:
    sensitivity = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    return {
        'tp': int(tp),
        'fp': int(fp),
        'fn': int(fn),
        'sensitivity': float(sensitivity),
        'precision': float(precision),
    }


def evaluate_decoding(
    decoded_tsv: str,
    expected_tsv: str,
    topk_starts: int,
    top_start_rank_only: bool = False,
    per_sequence: bool = False,
) -> Dict[str, object]:
    """Evaluate exon-level and gene-level metrics comparing decoded TSV vs expected TSV.

    - Strand-aware exact matching for both exons and genes.
    - Starts are ranked per sequence by boundary_score of start_rank==1 candidate (fallback described above).
    - If top_start_rank_only is True, only include start_rank==1 candidate per selected start.
      Otherwise include all candidates under each selected start.
    """
    decoded = _parse_decoded_tsv(decoded_tsv)
    expected = _parse_expected(expected_tsv)
    expected_starts = _parse_expected_starts(expected_tsv)

    total_tp_ex = total_fp_ex = total_fn_ex = 0
    total_tp_g = total_fp_g = total_fn_g = 0
    total_tp_st = total_fp_st = total_fn_st = 0
    per_seq_out: Dict[str, Dict[str, Dict[str, float]]] = {}

    for sid, starts in decoded.items():
        exp_genes_list = expected.get(sid, [])
        exp_gene_set: Set[Tuple[str, Tuple[Tuple[int, int], ...]]] = set(exp_genes_list)
        exp_exon_set: Set[Tuple[str, int, int]] = set()
        for strand, exons in exp_genes_list:
            for s0, e0 in exons:
                exp_exon_set.add((strand, s0, e0))

        selected_starts = _pick_top_k_starts(starts, topk_starts)

        pred_gene_set: Set[Tuple[str, Tuple[Tuple[int, int], ...]]] = set()
        pred_exon_set: Set[Tuple[str, int, int]] = set()
        pred_start_set: Set[Tuple[str, int]] = set()

        for start1 in selected_starts:
            cands = starts.get(start1, [])
            if top_start_rank_only:
                # include only start_rank==1; if multiple flagged as 1, include all such
                chosen = [c for c in cands if c.start_rank == 1]
                if not chosen and cands:
                    # fall back to best-rank candidate
                    best = min(cands, key=lambda x: (x.start_rank, -x.boundary_score))
                    chosen = [best]
            else:
                chosen = list(cands)

            for c in chosen:
                gene_key = (c.strand, tuple(c.exons))
                pred_gene_set.add(gene_key)
                for s0, e0 in c.exons:
                    pred_exon_set.add((c.strand, s0, e0))
                # Record strand-aware start coordinate (1-based)
                pred_start_set.add((c.strand, start1))

        # exon-level
        tp_ex = len(pred_exon_set & exp_exon_set)
        fp_ex = len(pred_exon_set - exp_exon_set)
        fn_ex = len(exp_exon_set - pred_exon_set)

        # gene-level
        tp_g = len(pred_gene_set & exp_gene_set)
        fp_g = len(pred_gene_set - exp_gene_set)
        fn_g = len(exp_gene_set - pred_gene_set)

        # start-level (strand-aware)
        exp_start_set = set(expected_starts.get(sid, set()))
        tp_st = len(pred_start_set & exp_start_set)
        fp_st = len(pred_start_set - exp_start_set)
        fn_st = len(exp_start_set - pred_start_set)

        total_tp_ex += tp_ex
        total_fp_ex += fp_ex
        total_fn_ex += fn_ex
        total_tp_g += tp_g
        total_fp_g += fp_g
        total_fn_g += fn_g
        # accumulate starts
        total_tp_st += tp_st
        total_fp_st += fp_st
        total_fn_st += fn_st

        if per_sequence:
            per_seq_out[sid] = {
                'exon': _compute_counts(tp_ex, fp_ex, fn_ex),
                'gene': _compute_counts(tp_g, fp_g, fn_g),
                'start': _compute_counts(tp_st, fp_st, fn_st),
            }

    result = {
        'exon': _compute_counts(total_tp_ex, total_fp_ex, total_fn_ex),
        'gene': _compute_counts(total_tp_g, total_fp_g, total_fn_g),
        'start': _compute_counts(total_tp_st, total_fp_st, total_fn_st),
    }
    if per_sequence:
        result['per_sequence'] = per_seq_out
    return result



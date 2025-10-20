from typing import Dict, Tuple, List, Optional
import csv
import numpy as np
import torch

from gene_decoder import PredictedSequence
from utils.constants import GenePredictionClass as P
from utils.events import compute_event_spans_vectorized
from gene_decoder.flanking_stats import _build_motifs_map, _extract_flank


def _class_name_to_index(class_order: List[str], class_name: str) -> Optional[int]:
    try:
        return int(class_order.index(str(class_name)))
    except ValueError:
        return None


def load_flank_counts_csv(path: str) -> Tuple[Dict[str, Tuple[int, float]], Dict[str, Tuple[int, float]]]:
    """Load flank stats CSV produced by scripts/analyze-flanking-motifs.py.

    Returns two dicts (dss_map, ass_map) mapping motif -> (t, p_over_t).
    """
    dss: Dict[str, Tuple[int, float]] = {}
    ass: Dict[str, Tuple[int, float]] = {}
    with open(path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader, None)
        # Expect: site,motif,t,p,n,p_over_t
        for row in reader:
            if not row or len(row) < 6:
                continue
            site = row[0].strip().upper()
            motif = row[1].strip().upper()
            try:
                t_val = int(float(row[2]))
            except Exception:
                continue
            try:
                p_over_t = float(row[5])
            except Exception:
                continue
            if site == 'DSS':
                dss[motif] = (t_val, p_over_t)
            elif site == 'ASS':
                ass[motif] = (t_val, p_over_t)
    return dss, ass


def _apply_span_mean_override(probs: np.ndarray, cls_col: int, s: int, e: int, target_mean: float) -> None:
    s0 = int(max(0, s))
    e0 = int(min(probs.shape[0], e))
    if e0 <= s0 or cls_col is None:
        return
    # For now, set both positions to the target mean to achieve exact mean
    new_val = float(max(1e-8, min(1.0 - 1e-8, target_mean)))
    probs[s0:e0, int(cls_col)] = new_val


def determine_replacement_prob(t_val: int, stats_prob: float, existing_prob: float, mode: str) -> float:
    """Compute replacement probability for a span.

    For now, always use stats_prob regardless of mode; placeholder for future modes.
    """
    if mode == "override":
        return float(stats_prob)
    else:
        if t_val > 50 and existing_prob < stats_prob and stats_prob > 0.50:
            print("boosting", existing_prob, stats_prob)
            return float((existing_prob+stats_prob*3)/4)
        return float(existing_prob)


def augment_items_from_flanks(
    items: List[PredictedSequence],
    flank_counts_csv: str,
    *,
    flank: int,
    dss_motifs_mode: str = 'standard',
    mode: str = 'override',
) -> List[PredictedSequence]:
    """Override/augment ASS/DSS probabilities using flank stats CSV.

    mode:
      - 'override': set span values to the stats probability
      - 'augment': currently same as override (placeholder for future)
    """
    dss_map, ass_map = load_flank_counts_csv(flank_counts_csv)
    out: List[PredictedSequence] = []
    motifs_by_class = _build_motifs_map(dss_motifs_mode)

    for r in items:
        seq = r.sequence
        probs = r.probabilities
        class_order = r.class_order
        if seq is None or probs is None or class_order is None:
            out.append(r)
            continue
        L = int(len(seq))
        if probs.shape[0] != L:
            out.append(r)
            continue

        probs_new = np.array(probs, dtype=np.float32, copy=True)
        dss_col = _class_name_to_index(class_order, 'DSS')
        ass_col = _class_name_to_index(class_order, 'ASS')

        # Compute spans once
        tokens_t = torch.tensor([{'A':0,'T':1,'G':2,'C':3}.get(ch,4) for ch in seq], dtype=torch.long)
        spans_map = compute_event_spans_vectorized(tokens_t, motifs_by_class)

        # DSS
        if dss_col is not None:
            for (s, e) in spans_map.get(int(P.DSS), []):
                motif_str = _extract_flank(seq, s, e, int(flank))
                if motif_str is None:
                    continue
                stats = dss_map.get(motif_str)
                if stats is None:
                    continue
                t_val, prob_mean = stats
                s0 = int(max(0, s))
                e0 = int(min(L, e))
                if e0 <= s0:
                    continue
                span_vals = probs_new[s0:e0, int(dss_col)].astype(float)
                finite = np.isfinite(span_vals)
                existing_mean = float(np.mean(span_vals[finite])) if finite.any() else 0.0
                replacement = determine_replacement_prob(int(t_val), float(prob_mean), existing_mean, mode)
                _apply_span_mean_override(probs_new, dss_col, s, e, replacement)

        # ASS
        if ass_col is not None:
            for (s, e) in spans_map.get(int(P.ASS), []):
                motif_str = _extract_flank(seq, s, e, int(flank))
                if motif_str is None:
                    continue
                stats = ass_map.get(motif_str)
                if stats is None:
                    continue
                t_val, prob_mean = stats
                s0 = int(max(0, s))
                e0 = int(min(L, e))
                if e0 <= s0:
                    continue
                span_vals = probs_new[s0:e0, int(ass_col)].astype(float)
                finite = np.isfinite(span_vals)
                existing_mean = float(np.mean(span_vals[finite])) if finite.any() else 0.0
                replacement = determine_replacement_prob(int(t_val), float(prob_mean), existing_mean, mode)
                _apply_span_mean_override(probs_new, ass_col, s, e, replacement)

        out.append(PredictedSequence(
            sequence_index=r.sequence_index,
            sequence=seq,
            probabilities=probs_new,
            class_order=class_order,
            sequence_id=r.sequence_id,
        ))

    return out



#!/usr/bin/env python3

from typing import List, Dict, Optional

import numpy as np

from utils.constants import DNAEmbed


def convert_tokens_to_sequence(tokens) -> str:
    """Convert token indices back to DNA sequence using DNAEmbed.idx_to_bp mapping."""
    idx_to_nucleotide = DNAEmbed.idx_to_bp
    # tokens may be numpy array or list; ensure int conversion per element
    return ''.join([idx_to_nucleotide.get(int(token), 'N') for token in tokens])


def _discover_motif_len_for_class(results_data: List[Dict], class_idx: int, max_len: int = 5) -> Optional[int]:
    lengths: Dict[int, int] = {}
    for result in results_data:
        tgt = result.get('targets')
        if tgt is None:
            continue
        L = len(tgt)
        i = 0
        while i < L:
            if int(tgt[i]) == int(class_idx):
                j = i
                while j < L and int(tgt[j]) == int(class_idx):
                    j += 1
                run_len = min(max_len, j - i)
                lengths[run_len] = lengths.get(run_len, 0) + 1
                i = j
            else:
                i += 1
    if not lengths:
        return None
    candidate = None
    best_count = -1
    for k in (2, 3):
        if lengths.get(k, 0) > best_count:
            best_count = lengths.get(k, 0)
            candidate = k
    return candidate if best_count > 0 else None


def _collect_motifs_from_targets(results_data: List[Dict], class_idx: int, motif_len: int) -> set:
    motifs = set()
    for result in results_data:
        seq = convert_tokens_to_sequence(result['sequence_tokens'])
        tgt = result.get('targets')
        if tgt is None:
            continue
        L = min(len(seq), len(tgt))
        for pos in range(0, max(0, L - motif_len + 1)):
            segment = tgt[pos:pos+motif_len]
            if len(segment) == motif_len and np.all(segment == class_idx):
                motifs.add(seq[pos:pos+motif_len])
    return motifs


def calculate_generic_metrics(results_data: List[Dict], class_weights: Optional[List[float]] = None, min_weight: float = 1.0) -> Dict[int, Dict[str, float]]:
    """Compute per-class TP/FP/FN/TN and derived metrics over results_data.

    Behavior overview
    -----------------
    - Class filtering: If ``class_weights`` is provided, only classes with weight > ``min_weight``
      are evaluated. Otherwise, all classes present in targets are considered.
    - Motif-aware counting (length 2 or 3): For classes that appear as contiguous runs of length
      2 or 3 in targets, we discover a predominant run length per class and collect the set of
      concrete sequence motifs (e.g., {"ATG"} for START) observed at those runs. We then slide a
      window of that motif length across each sequence and evaluate only windows whose underlying
      sequence substring is in the target-derived motif set. For each such window, we mark:
        * target_is_cls = any target token within the window equals the class
        * pred_is_cls = any predicted token within the window equals the class
      We increment TP/FP/FN/TN at the window level accordingly. Probabilities are not used.
    - Per-position counting (all other classes): If no 2/3-length motif is discovered for a class,
      fall back to token-level counts using exact class equality per position.

    Examples (motif-aware START with motif set {"ATG"})
    ---------------------------------------------------
    - CTCA, predicted START at the central T → windows "CTC" and "TCA" are not in {"ATG"},
      so both windows are ignored (no FP).
    - ATCA with targets labeling A/T as START but window substring is "ATC" → not in {"ATG"},
      so ignored (no TP/FP/FN from this window).
    - ATGA with predicted START only on T within "ATG" → window "ATG" is evaluated; since
      both target_is_cls and pred_is_cls are True (any-in-window), it counts as TP.
    - Predicted-only on a non-motif substring (e.g., START on "TTG" when {"ATG"}) → ignored (no FP).

    DSS/ASS (2-mer) behave the same way with their discovered motif set and a 2-length window.
    """
    has_targets = any(result.get('targets') is not None for result in results_data)
    if not has_targets:
        return {}

    classes_in_targets = set()
    for result in results_data:
        tgt = result.get('targets')
        if tgt is None:
            continue
        for val in np.unique(tgt):
            classes_in_targets.add(int(val))
    if class_weights is not None and len(class_weights) > 0:
        weighted_classes = {i for i, w in enumerate(class_weights) if w is not None and float(w) > float(min_weight)}
        candidate_classes = [c for c in sorted(classes_in_targets) if c in weighted_classes]
    else:
        candidate_classes = sorted(classes_in_targets)

    metrics_by_class: Dict[int, Dict[str, float]] = {}

    motif_len_map: Dict[int, Optional[int]] = {c: _discover_motif_len_for_class(results_data, c) for c in candidate_classes}
    motif_sets: Dict[int, set] = {}
    for c, mlen in motif_len_map.items():
        if mlen in (2, 3):
            motif_sets[c] = _collect_motifs_from_targets(results_data, c, mlen)

    for class_idx in candidate_classes:
        mlen = motif_len_map.get(class_idx)
        tp = fp = fn = tn = 0
        if mlen in (2, 3) and motif_sets.get(class_idx):
            motifs = motif_sets[class_idx]
            for result in results_data:
                seq = convert_tokens_to_sequence(result['sequence_tokens'])
                tgt = result['targets']
                pred = result['predictions']
                L = min(len(seq), len(tgt), len(pred))
                for pos in range(0, max(0, L - mlen + 1)):
                    if seq[pos:pos+mlen] not in motifs:
                        continue
                    target_is_cls = bool((tgt[pos:pos+mlen] == class_idx).any())
                    pred_is_cls = bool((pred[pos:pos+mlen] == class_idx).any())
                    if pred_is_cls and target_is_cls:
                        tp += 1
                    elif pred_is_cls and not target_is_cls:
                        fp += 1
                    elif (not pred_is_cls) and target_is_cls:
                        fn += 1
                    else:
                        tn += 1
        else:
            for result in results_data:
                tgt = result['targets']
                pred = result['predictions']
                L = min(len(tgt), len(pred))
                target_mask = (tgt[:L] == class_idx)
                pred_mask = (pred[:L] == class_idx)
                tp += int(np.logical_and(target_mask, pred_mask).sum())
                fp += int(np.logical_and(~target_mask, pred_mask).sum())
                fn += int(np.logical_and(target_mask, ~pred_mask).sum())
                tn += int(np.logical_and(~target_mask, ~pred_mask).sum())

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        metrics_by_class[class_idx] = {
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'tn': tn,
            'sensitivity': sensitivity,
            'precision': precision,
            'specificity': specificity,
        }

    return metrics_by_class



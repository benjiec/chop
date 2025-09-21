#!/usr/bin/env python3

from typing import List, Dict, Optional, Sequence

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


def calculate_generic_metrics_and_predictions(results_data: List[Dict], class_weights: Optional[List[float]] = None, min_weight: float = 1.0,
                                              valid_masks: Optional[Sequence[Optional[Sequence[bool]]]] = None):
    """Compute per-class metrics and also return motif-window events for visualization.

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
    
    Returns
    -------
    (metrics_by_class, window_events)
        - metrics_by_class: dict[class_index] -> metrics dict
        - window_events: list of dict with keys:
            {'sequence_index', 'class_index', 'class_label', 'start', 'end', 'classification'}
          Included only when target_is_cls or pred_is_cls (no TN windows).
    """
    has_targets = any(result.get('targets') is not None for result in results_data)
    if not has_targets:
        return {}, []

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
    window_events: List[Dict] = []

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
            for idx, result in enumerate(results_data):
                seq = convert_tokens_to_sequence(result['sequence_tokens'])
                tgt = result['targets']
                pred = result['predictions']
                L = min(len(seq), len(tgt), len(pred))
                seq_idx = result.get('sequence_index')
                # Optional validity mask for this sequence
                vm_np = None
                if valid_masks is not None and idx < len(valid_masks) and valid_masks[idx] is not None:
                    vm_list = list(valid_masks[idx])
                    if len(vm_list) >= L:
                        vm_np = np.array(vm_list[:L], dtype=bool)
                    else:
                        vm_np = np.array(vm_list + [False] * (L - len(vm_list)), dtype=bool)
                for pos in range(0, max(0, L - mlen + 1)):
                    if vm_np is not None and not vm_np[pos:pos+mlen].all():
                        continue
                    if seq[pos:pos+mlen] not in motifs:
                        continue
                    target_is_cls = bool((tgt[pos:pos+mlen] == class_idx).any())
                    pred_is_cls = bool((pred[pos:pos+mlen] == class_idx).any())
                    if pred_is_cls and target_is_cls:
                        tp += 1
                        window_events.append({
                            'sequence_index': seq_idx,
                            'class_index': int(class_idx),
                            'class_label': str(class_idx),
                            'start': pos,
                            'end': pos + mlen - 1,
                            'classification': 'TP',
                        })
                    elif pred_is_cls and not target_is_cls:
                        fp += 1
                        window_events.append({
                            'sequence_index': seq_idx,
                            'class_index': int(class_idx),
                            'class_label': str(class_idx),
                            'start': pos,
                            'end': pos + mlen - 1,
                            'classification': 'FP',
                        })
                    elif (not pred_is_cls) and target_is_cls:
                        fn += 1
                        window_events.append({
                            'sequence_index': seq_idx,
                            'class_index': int(class_idx),
                            'class_label': str(class_idx),
                            'start': pos,
                            'end': pos + mlen - 1,
                            'classification': 'FN',
                        })
                    else:
                        tn += 1
        else:
            for idx, result in enumerate(results_data):
                tgt = result['targets']
                pred = result['predictions']
                L = min(len(tgt), len(pred))
                target_mask = (tgt[:L] == class_idx)
                pred_mask = (pred[:L] == class_idx)
                # Apply optional validity mask
                if valid_masks is not None and idx < len(valid_masks) and valid_masks[idx] is not None:
                    vm_list = list(valid_masks[idx])
                    if len(vm_list) >= L:
                        vm_np = np.array(vm_list[:L], dtype=bool)
                    else:
                        vm_np = np.array(vm_list + [False] * (L - len(vm_list)), dtype=bool)
                    target_mask = np.logical_and(target_mask, vm_np)
                    pred_mask = np.logical_and(pred_mask, vm_np)
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

    # Fill human-readable labels for events now that metrics_by_class is assembled
    from utils.constants import GenePredictionClass as P
    for ev in window_events:
        idx = int(ev['class_index'])
        ev['class_label'] = P.idx_to_cls.get(idx, str(idx))
    return metrics_by_class, window_events


def calculate_generic_metrics(results_data: List[Dict], class_weights: Optional[List[float]] = None, min_weight: float = 1.0,
                              valid_masks: Optional[Sequence[Optional[Sequence[bool]]]] = None) -> Dict[int, Dict[str, float]]:
    """Back-compat wrapper: return only metrics, discarding window events."""
    metrics_by_class, _events = calculate_generic_metrics_and_predictions(results_data, class_weights=class_weights, min_weight=min_weight, valid_masks=valid_masks)
    return metrics_by_class

# Back-compat alias (window events == motif-span predictions)
calculate_generic_metrics_with_windows = calculate_generic_metrics_and_predictions


def compute_brier_scores(results_data: List[Dict],
                         class_weights: Optional[Sequence[Optional[float]]] = None,
                         min_weight: float = 0.0,
                         valid_masks: Optional[Sequence[Optional[Sequence[bool]]]] = None,
                         event_only: bool = True) -> Dict[str, object]:
    """Compute multi-class Brier score (overall) and per-class Brier from results_data.

    Expected result entry keys:
      - 'targets': np.ndarray of shape (L,)
      - 'probabilities': np.ndarray of shape (L, C) (softmax probs). If missing but 'logits' is present, will softmax.
      - Optional 'predictions': np.ndarray of shape (L,) used when event_only=True to include predicted events.

    The overall multi-class Brier is mean over positions of sum_k (p_k - y_k)^2 without dividing by C.
    Per-class Brier is mean over positions of (p_c - y_c)^2 for each class c.
    If class_weights is provided, only classes with weight > min_weight are included in the overall and per-class outputs.
    If event_only is True, only positions where either the target or the predicted class is in the allowed classes are included.
    """
    total_sq_error_sum = 0.0
    total_token_count = 0

    per_class_sq_error_sum: Dict[int, float] = {}
    per_class_token_count: Dict[int, int] = {}
    allowed_classes: Optional[set] = None
    if class_weights is not None and len(class_weights) > 0:
        try:
            allowed_classes = {i for i, w in enumerate(class_weights) if (w is not None) and (float(w) > float(min_weight))}
        except Exception:
            allowed_classes = None

    for idx, result in enumerate(results_data):
        probs = result.get('probabilities')
        logits = result.get('logits')
        if probs is None and logits is not None:
            # Softmax logits to probabilities if provided as logits
            import numpy as _np
            x = logits - _np.max(logits, axis=-1, keepdims=True)
            ex = _np.exp(x)
            probs = ex / _np.clip(_np.sum(ex, axis=-1, keepdims=True), 1e-12, None)
        if probs is None:
            # Cannot compute Brier for this entry
            continue
        tgt = result.get('targets')
        if tgt is None:
            continue
        L = min(int(len(tgt)), int(probs.shape[0]))
        if L <= 0:
            continue

        # Optional validity mask per sequence
        if valid_masks is not None and idx < len(valid_masks) and valid_masks[idx] is not None:
            vm_list = list(valid_masks[idx])
            if len(vm_list) >= L:
                vm_np = np.array(vm_list[:L], dtype=bool)
            else:
                vm_np = np.array(vm_list + [False] * (L - len(vm_list)), dtype=bool)
        else:
            vm_np = np.ones(L, dtype=bool)

        # Slice
        probsL = probs[:L]
        tgtL = np.array(tgt[:L], dtype=int)

        # Determine class mask
        C = int(probsL.shape[1])
        if allowed_classes is not None:
            class_mask = np.array([c in allowed_classes for c in range(C)], dtype=bool)
        else:
            class_mask = np.ones(C, dtype=bool)
        if not class_mask.any():
            continue

        # Event-only position mask
        include_positions = np.ones(L, dtype=bool)
        if event_only:
            # Positions where target in allowed or prediction in allowed
            tgt_allowed = class_mask[tgtL]
            pred = result.get('predictions')
            pred_allowed = np.zeros(L, dtype=bool)
            if pred is not None and len(pred) >= L:
                predL = np.array(pred[:L], dtype=int)
                pred_allowed = class_mask[predL]
            include_positions = np.logical_or(tgt_allowed, pred_allowed)
        # Apply validity mask
        include_positions = np.logical_and(include_positions, vm_np)
        pos_idx = np.where(include_positions)[0]
        if pos_idx.size == 0:
            continue

        probsM = probsL[pos_idx]
        tgtM = tgtL[pos_idx]

        # Build one-hot labels for included positions
        one_hot = np.zeros((tgtM.shape[0], C), dtype=np.float32)
        one_hot[np.arange(tgtM.shape[0]), tgtM] = 1.0

        # Overall multi-class Brier per token: sum over allowed classes only
        sq_err = (probsM - one_hot) ** 2
        sq_err = sq_err[:, class_mask]
        total_sq_error_sum += float(np.sum(sq_err))
        total_token_count += int(sq_err.shape[0])

        # Per-class Brier (binary for each allowed class): (p_c - y_c)^2
        for c in range(C):
            if not class_mask[c]:
                continue
            se_c = (probsM[:, c] - one_hot[:, c]) ** 2
            per_class_sq_error_sum[c] = per_class_sq_error_sum.get(c, 0.0) + float(np.sum(se_c))
            per_class_token_count[c] = per_class_token_count.get(c, 0) + int(se_c.shape[0])

    overall = (total_sq_error_sum / total_token_count) if total_token_count > 0 else 0.0

    brier_by_class: Dict[int, float] = {}
    for c, s in per_class_sq_error_sum.items():
        n = per_class_token_count.get(c, 0)
        brier_by_class[int(c)] = (s / n) if n > 0 else 0.0

    return {
        'brier': float(overall),
        'brier_by_class': brier_by_class,
    }



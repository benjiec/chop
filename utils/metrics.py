#!/usr/bin/env python3

from typing import List, Dict, Optional, Sequence, Iterable, Union, Set, Tuple, Callable
from dataclasses import dataclass

import numpy as np

from utils.constants import DNAEmbed
from utils.constants import GenePredictionClass as P
from utils.events import normalize_event_motifs_map
from utils.events import group_motifs_by_length
from utils.events import convert_tokens_to_sequence
from utils.events import build_event_masks, build_center_mask
from utils.events import compute_event_spans_vectorized
import torch


@dataclass(frozen=True)
class SequenceResult:
    sequence_index: Optional[int]
    sequence_tokens: np.ndarray
    targets: Optional[np.ndarray]
    sequence_id: Optional[str] = None
    predictions: Optional[np.ndarray] = None
    probabilities: Optional[np.ndarray] = None
    attentions: Optional[Dict[str, np.ndarray]] = None

    @staticmethod
    def from_batch(
        sequence_tokens_batch: torch.Tensor,
        targets_batch: Optional[torch.Tensor],
        logits_batch: torch.Tensor,
        sequence_index_start: int = 0,
        prob_activation: str = 'softmax',
        sequence_ids: Optional[Sequence[Optional[str]]] = None,
        event_motifs_by_class: Optional[Dict[int, Set[str]]] = None,
        event_margin_bp: Optional[int] = None,
        mask_non_event_probs: bool = True,
    ) -> List["SequenceResult"]:
        """Build a list of SequenceResult from batched tensors/arrays.

        Accepts PyTorch tensors for inputs. Computes predictions and probabilities
        for the full batch and returns SequenceResult objects (fields stored as numpy arrays).
        """
        with torch.no_grad():
            predictions_t = logits_batch.argmax(dim=-1)
            is_sigmoid = (str(prob_activation).lower() == 'sigmoid')
            if is_sigmoid:
                probabilities_t = torch.sigmoid(logits_batch)
            else:
                probabilities_t = torch.softmax(logits_batch, dim=-1)

        B = int(logits_batch.size(0))
        results: List[SequenceResult] = []
        for b in range(B):
            probs_b = probabilities_t[b]
            # Optional per-class NaN masking outside event spans when using sigmoid
            if mask_non_event_probs and (str(prob_activation).lower() == 'sigmoid') and (event_motifs_by_class is not None):
                tokens_b = sequence_tokens_batch[b:b+1]
                L = int(tokens_b.size(1))
                masks = build_event_masks(tokens_b, event_motifs_by_class)
                center = build_center_mask(1, L, int(event_margin_bp) if event_margin_bp is not None else 0, device=tokens_b.device)[0]
                for cls_id, mask in masks.items():
                    mask1 = (mask[0] & center)  # (L,)
                    inv = ~mask1
                    # Set outside-event positions to NaN for this class
                    probs_b[inv, int(cls_id)] = float('nan')
            results.append(SequenceResult(
                sequence_index=sequence_index_start + b,
                sequence_tokens=sequence_tokens_batch[b].detach().cpu().numpy(),
                targets=(targets_batch[b].detach().cpu().numpy() if targets_batch is not None else None),
                sequence_id=(sequence_ids[b] if (sequence_ids is not None and b < len(sequence_ids)) else None),
                predictions=predictions_t[b].detach().cpu().numpy().astype(np.int64),
                probabilities=probs_b.detach().cpu().numpy().astype(np.float32),
            ))
        return results

    def to_dict(self) -> Dict[str, object]:
        return {
            'sequence_index': self.sequence_index,
            'sequence_tokens': self.sequence_tokens,
            'targets': self.targets,
            'predictions': self.predictions,
            'probabilities': self.probabilities,
        }

def _build_valid_mask_for(seq_len: int, mask_like: Optional[Sequence[bool]]) -> np.ndarray:
    if mask_like is None:
        return np.ones(seq_len, dtype=bool)
    vm_list = list(mask_like)
    if len(vm_list) >= seq_len:
        return np.array(vm_list[:seq_len], dtype=bool)
    return np.array(vm_list + [False] * (seq_len - len(vm_list)), dtype=bool)


def event_based_generic_metrics_factory(event_motifs_by_class: Dict[Union[int, str], Iterable[str]]):
    """Build event-based generic metrics functions (metrics-only and metrics+events).

    The returned functions have the same signatures as existing calculate_generic_metrics
    and calculate_generic_metrics_and_predictions.
    """
    normalized = normalize_event_motifs_map(event_motifs_by_class)
    motifs_by_len = group_motifs_by_length(normalized)

    def calculate_generic_metrics_and_predictions(results_data: List[SequenceResult], min_weight: float = 1.0,
                                                  valid_masks: Optional[Sequence[Optional[Sequence[bool]]]] = None):
        has_targets = any(result.targets is not None for result in results_data)
        if not has_targets:
            return {}, []

        # Event-based: evaluate exactly the classes configured in the motifs map
        candidate_classes = sorted(list(motifs_by_len.keys()))

        metrics_by_class: Dict[int, Dict[str, float]] = {}
        window_events: List[Dict] = []

        for class_idx in candidate_classes:
            tp = fp = fn = tn = 0
            lens_map = motifs_by_len.get(int(class_idx), {})
            if not lens_map:
                continue
            for ridx, result in enumerate(results_data):
                seq = convert_tokens_to_sequence(result.sequence_tokens)
                tgt = result.targets
                pred = result.predictions
                L = min(len(seq), len(tgt), len(pred))
                vm = _build_valid_mask_for(L, valid_masks[ridx] if (valid_masks is not None and ridx < len(valid_masks)) else None)
                seq_idx = result.sequence_index

                for k, motifs in lens_map.items():
                    if k <= 0 or k > L:
                        continue
                    for pos in range(0, L - k + 1):
                        if not vm[pos:pos+k].all():
                            continue
                        if seq[pos:pos+k].upper() not in motifs:
                            continue
                        target_is_cls = bool((tgt[pos:pos+k] == class_idx).any())
                        pred_is_cls = bool((pred[pos:pos+k] == class_idx).any())
                        if pred_is_cls and target_is_cls:
                            tp += 1
                            window_events.append({
                                'sequence_index': seq_idx,
                                'class_index': int(class_idx),
                                'class_label': str(class_idx),
                                'start': pos,
                                'end': pos + k - 1,
                                'classification': 'TP',
                            })
                        elif pred_is_cls and not target_is_cls:
                            fp += 1
                            window_events.append({
                                'sequence_index': seq_idx,
                                'class_index': int(class_idx),
                                'class_label': str(class_idx),
                                'start': pos,
                                'end': pos + k - 1,
                                'classification': 'FP',
                            })
                        elif (not pred_is_cls) and target_is_cls:
                            fn += 1
                            window_events.append({
                                'sequence_index': seq_idx,
                                'class_index': int(class_idx),
                                'class_label': str(class_idx),
                                'start': pos,
                                'end': pos + k - 1,
                                'classification': 'FN',
                            })
                        else:
                            tn += 1

            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            metrics_by_class[int(class_idx)] = {
                'tp': tp,
                'fp': fp,
                'fn': fn,
                'tn': tn,
                'sensitivity': sensitivity,
                'precision': precision,
                'specificity': specificity,
            }

        for ev in window_events:
            idx = int(ev['class_index'])
            ev['class_label'] = P.idx_to_cls.get(idx, str(idx))
        return metrics_by_class, window_events

    def calculate_generic_metrics(results_data: List[SequenceResult], min_weight: float = 1.0,
                                  valid_masks: Optional[Sequence[Optional[Sequence[bool]]]] = None) -> Dict[int, Dict[str, float]]:
        metrics_by_class, _events = calculate_generic_metrics_and_predictions(results_data, min_weight=min_weight, valid_masks=valid_masks)
        return metrics_by_class

    return calculate_generic_metrics, calculate_generic_metrics_and_predictions


def event_based_brier_factory(event_motifs_by_class: Dict[Union[int, str], Iterable[str]]):
    """Build an event-driven Brier score function.

    Returned function signature: compute_brier_scores(results_data, valid_masks=None, event_only=True)
    Semantics:
      - Compute per-class Brier only on that class's motif spans
      - Overall Brier = unweighted mean of per-class Brier across classes with at least one included token
    """
    normalized = normalize_event_motifs_map(event_motifs_by_class)
    motifs_by_len = group_motifs_by_length(normalized)

    def compute_brier_scores(results_data: List[SequenceResult],
                             valid_masks: Optional[Sequence[Optional[Sequence[bool]]]] = None,
                             event_only: bool = True) -> Dict[str, object]:
        # Evaluate exactly the classes configured in the motifs map
        classes = sorted(list(motifs_by_len.keys()))

        per_class_sq_error_sum: Dict[int, float] = {}
        per_class_token_count: Dict[int, int] = {}

        for ridx, result in enumerate(results_data):
            probs = result.probabilities
            if probs is None:
                continue
            tgt = result.targets
            if tgt is None:
                continue
            L = min(int(len(tgt)), int(probs.shape[0]))
            if L <= 0:
                continue
            vm = _build_valid_mask_for(L, valid_masks[ridx] if (valid_masks is not None and ridx < len(valid_masks)) else None)
            tgtL = np.array(tgt[:L], dtype=int)
            probsL = probs[:L]
            C = int(probsL.shape[1])

            # Precompute one-hot labels
            one_hot = np.zeros((L, C), dtype=np.float32)
            one_hot[np.arange(L), tgtL] = 1.0

            seq_str = convert_tokens_to_sequence(result.sequence_tokens)[:L]

            for cls in classes:
                lens_map = motifs_by_len.get(int(cls), {})
                if not lens_map:
                    continue
                included_positions = np.zeros(L, dtype=bool)
                for k, motifs in lens_map.items():
                    if k <= 0 or k > L:
                        continue
                    for pos in range(0, L - k + 1):
                        if seq_str[pos:pos+k].upper() in motifs:
                            # include only the positions within the span that are valid per mask
                            seg_mask = vm[pos:pos+k]
                            if not seg_mask.any():
                                continue
                            included_positions[pos:pos+k] = np.logical_or(included_positions[pos:pos+k], seg_mask)
                idx = np.where(included_positions)[0]
                if idx.size == 0:
                    continue
                pcls = probsL[idx, int(cls)]
                valid = np.isfinite(pcls)
                if not valid.any():
                    continue
                se_c = (pcls[valid] - one_hot[idx[valid], int(cls)]) ** 2
                per_class_sq_error_sum[int(cls)] = per_class_sq_error_sum.get(int(cls), 0.0) + float(np.sum(se_c))
                per_class_token_count[int(cls)] = per_class_token_count.get(int(cls), 0) + int(se_c.shape[0])

        # Per-class Brier
        brier_by_class: Dict[int, float] = {}
        for c, s in per_class_sq_error_sum.items():
            n = per_class_token_count.get(c, 0)
            brier_by_class[int(c)] = (s / n) if n > 0 else 0.0

        # Overall: unweighted mean across classes with tokens
        vals = [v for v in brier_by_class.values()]
        overall = float(sum(vals) / len(vals)) if len(vals) > 0 else 0.0
        return {'brier': overall, 'brier_by_class': brier_by_class}

    return compute_brier_scores



def compute_event_span_mean_probability_metrics(
    results_data: List[SequenceResult],
    event_motifs_by_class: Dict[Union[int, str], Iterable[str]],
) -> Dict[int, Dict[str, Dict[str, float]]]:
    """Aggregate per-span mean probabilities into TP/TN Beta fits per class.

    Returns a nested dict keyed by class id, then category 'tp'/'tn', each with:
    { 'n': int, 'mean': float, 'std': float, 'beta_alpha': float, 'beta_beta': float }
    """
    normalized = normalize_event_motifs_map(event_motifs_by_class)
    classes = sorted(normalized.keys())

    # Collect samples per class and category
    samples_tp: Dict[int, list] = {int(c): [] for c in classes}
    samples_tn: Dict[int, list] = {int(c): [] for c in classes}

    for r in results_data:
        probs = r.probabilities
        labels = r.targets
        if probs is None or labels is None:
            continue
        tokens_t = torch.as_tensor(r.sequence_tokens, dtype=torch.long)
        spans_map = compute_event_spans_vectorized(tokens_t, normalized)
        L = int(probs.shape[0])
        for cls_idx in classes:
            for (s, e_excl) in spans_map.get(int(cls_idx), []):
                s0, e0 = int(s), int(e_excl - 1)
                if e0 <= s0:
                    continue
                s0 = max(0, s0)
                e0 = min(L - 1, e0)
                window = probs[s0:e0+1, int(cls_idx)].astype(float)
                finite_mask = np.isfinite(window)
                if not finite_mask.any():
                    continue
                mean_p = float(np.mean(window[finite_mask]))
                mean_p = float(np.clip(mean_p, 1e-8, 1.0 - 1e-8))
                is_pos = bool(np.any(np.asarray(labels[s0:e0+1], dtype=int) == int(cls_idx)))
                if is_pos:
                    samples_tp[int(cls_idx)].append(mean_p)
                else:
                    samples_tn[int(cls_idx)].append(mean_p)

    def _fit_beta_moments(samples: list) -> Tuple[int, float, float, float, float]:
        if not samples:
            return 0, 0.0, 0.0, 0.0, 0.0
        arr = np.asarray(samples, dtype=float)
        arr = np.clip(arr, 1e-8, 1.0 - 1e-8)
        m = float(np.mean(arr))
        v = float(np.var(arr))
        n = int(arr.size)
        if v <= 1e-12 or m <= 1e-8 or m >= 1.0 - 1e-8:
            k = 100.0
            alpha = m * k
            beta = (1.0 - m) * k
            s = float(np.std(arr))
            return n, m, s, float(alpha), float(beta)
        k = m * (1.0 - m) / v - 1.0
        if k <= 0.0:
            k = 100.0
        alpha = m * k
        beta = (1.0 - m) * k
        s = float(np.sqrt(v))
        return n, m, s, float(alpha), float(beta)

    def _median_iqr(samples: list) -> Tuple[float, float]:
        if not samples:
            return 0.0, 0.0
        arr = np.asarray(samples, dtype=float)
        arr = np.clip(arr, 1e-8, 1.0 - 1e-8)
        med = float(np.median(arr))
        q1 = float(np.percentile(arr, 25))
        q3 = float(np.percentile(arr, 75))
        iqr = max(0.0, q3 - q1)
        return med, iqr

    out: Dict[int, Dict[str, Dict[str, float]]] = {}
    for cls in classes:
        n_tp, m_tp, s_tp, a_tp, b_tp = _fit_beta_moments(samples_tp[int(cls)])
        n_tn, m_tn, s_tn, a_tn, b_tn = _fit_beta_moments(samples_tn[int(cls)])
        med_tp, iqr_tp = _median_iqr(samples_tp[int(cls)])
        med_tn, iqr_tn = _median_iqr(samples_tn[int(cls)])
        out[int(cls)] = {
            'tp': {
                'n': float(n_tp),
                'mean': m_tp,
                'std': s_tp,
                'beta_alpha': a_tp,
                'beta_beta': b_tp,
                'median': med_tp,
                'iqr': iqr_tp,
            },
            'tn': {
                'n': float(n_tn),
                'mean': m_tn,
                'std': s_tn,
                'beta_alpha': a_tn,
                'beta_beta': b_tn,
                'median': med_tn,
                'iqr': iqr_tn,
            },
        }
    return out


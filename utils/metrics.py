#!/usr/bin/env python3

from typing import List, Dict, Optional, Sequence, Iterable, Union, Set, Tuple, Callable
from dataclasses import dataclass

import numpy as np

from utils.constants import DNAEmbed
from utils.constants import GenePredictionClass as P
from utils.events import normalize_event_motifs_map
from utils.events import group_motifs_by_length
from utils.events import convert_tokens_to_sequence
import torch


@dataclass(frozen=True)
class SequenceResult:
    sequence_index: Optional[int]
    sequence_tokens: np.ndarray
    targets: Optional[np.ndarray]
    predictions: Optional[np.ndarray] = None
    probabilities: Optional[np.ndarray] = None

    @staticmethod
    def from_batch(
        sequence_tokens_batch: torch.Tensor,
        targets_batch: Optional[torch.Tensor],
        logits_batch: torch.Tensor,
        sequence_index_start: int = 0,
    ) -> List["SequenceResult"]:
        """Build a list of SequenceResult from batched tensors/arrays.

        Accepts PyTorch tensors for inputs. Computes predictions and probabilities
        for the full batch and returns SequenceResult objects (fields stored as numpy arrays).
        """
        with torch.no_grad():
            predictions_t = logits_batch.argmax(dim=-1)
            probabilities_t = torch.softmax(logits_batch, dim=-1)

        B = int(logits_batch.size(0))
        results: List[SequenceResult] = []
        for b in range(B):
            results.append(SequenceResult(
                sequence_index=sequence_index_start + b,
                sequence_tokens=sequence_tokens_batch[b].detach().cpu().numpy(),
                targets=(targets_batch[b].detach().cpu().numpy() if targets_batch is not None else None),
                predictions=predictions_t[b].detach().cpu().numpy().astype(np.int64),
                probabilities=probabilities_t[b].detach().cpu().numpy().astype(np.float32),
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
                se_c = (probsL[idx, int(cls)] - one_hot[idx, int(cls)]) ** 2
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



#!/usr/bin/env python3

import torch
import torch.nn.functional as F
from typing import Iterable, Set, Dict, Any, Optional, Sequence, Union

from utils.constants import (
    DNAEmbed,
    GenePredictionClass as P,
    ConventionalStopCodons,
    ConventionalAcceptorDinucleotides,
)


def _normalize_class_weight_mapping(
    class_weights: Optional[Union[Sequence[Optional[float]], Dict[Union[int, str], Optional[float]]]],
) -> Optional[torch.Tensor]:
    """
    Convert various class weight specifications into a tensor of shape (num_classes,).

    Accepts one of:
      - Sequence[float | None] indexed by class id
      - Dict[int | str, float | None] where int is class id or str is class name in GenePredictionClass
    Returns None if class_weights is None.
    """
    if class_weights is None:
        return None
    # Build mapping id -> weight
    id_to_weight: Dict[int, float] = {}
    if isinstance(class_weights, dict):
        for k, v in class_weights.items():
            if v is None:
                continue
            if isinstance(k, int):
                id_to_weight[int(k)] = float(v)
            else:
                # assume class name string
                for cid, cname in P.idx_to_cls.items():
                    if str(k).upper() == str(cname).upper():
                        id_to_weight[int(cid)] = float(v)
                        break
    else:
        # sequence by id
        for cid, w in enumerate(class_weights):
            if w is None:
                continue
            id_to_weight[int(cid)] = float(w)

    # Build dense vector
    num_classes = len(P.idx_to_cls)
    vec = torch.ones(num_classes, dtype=torch.float32)
    for cid, w in id_to_weight.items():
        if 0 <= int(cid) < num_classes:
            vec[int(cid)] = float(w)
    return vec


def event_based_ce_loss_factory(
    dss_motifs: Iterable[str],
    class_weights: Optional[Union[Sequence[Optional[float]], Dict[Union[int, str], Optional[float]]]] = None,
):
    """
    Build an event-based CE loss that only computes cross-entropy on motif event positions.

    Events considered:
    - START: 'ATG' triplet starting at position i
    - STOP: any conventional stop codon triplet at position i
    - DSS: any 2-mer in dss_motifs at position i
    - ASS: any 2-mer in ConventionalAcceptorDinucleotides at position i

    Args:
        dss_motifs: iterable of donor splice site 2-mers to consider (e.g., {'GT'} or {'GT','GC','GA'})

    Returns:
        A function (sequences, targets, logits, components_out) -> loss_tensor
    """
    dss_set: Set[str] = set(m.upper() for m in dss_motifs)

    def event_based_ce_loss(sequences, targets, logits, components_out: Dict[str, Any]):
        # sequences: (B, L) int tokens
        # targets: (B, L) long class ids
        # logits: (B, L, C)
        device = logits.device
        B, L = sequences.shape

        # Convert token ids to base chars for motif scanning
        idx_to_bp = DNAEmbed.idx_to_bp

        # Build event mask of shape (B, L)
        event_mask = torch.zeros((B, L), dtype=torch.bool, device=device)

        # Precompute motif groups by length for dynamic scanning
        from collections import defaultdict
        stop_by_len: Dict[int, Set[str]] = defaultdict(set)
        for m in ConventionalStopCodons:
            stop_by_len[len(m)].add(m.upper())
        acc_by_len: Dict[int, Set[str]] = defaultdict(set)
        for m in ConventionalAcceptorDinucleotides:
            acc_by_len[len(m)].add(m.upper())
        dss_by_len: Dict[int, Set[str]] = defaultdict(set)
        for m in dss_set:
            dss_by_len[len(m)].add(m.upper())

        for b in range(B):
            # Convert to list of chars and string for substring checks
            bases = [idx_to_bp[int(sequences[b, i].item())] for i in range(L)]
            seq_str = ''.join(bases)

            # START 'ATG' fixed 3bp
            k = 3
            for i in range(0, L - k + 1):
                if seq_str[i:i+k] == 'ATG':
                    event_mask[b, i:i+k] = True

            # Dynamic STOP motifs by length
            for k, motifs in stop_by_len.items():
                if k <= 0 or k > L:
                    continue
                for i in range(0, L - k + 1):
                    if seq_str[i:i+k] in motifs:
                        event_mask[b, i:i+k] = True

            # Dynamic DSS motifs by length
            for k, motifs in dss_by_len.items():
                if k <= 0 or k > L:
                    continue
                for i in range(0, L - k + 1):
                    if seq_str[i:i+k] in motifs:
                        event_mask[b, i:i+k] = True

            # Dynamic ASS motifs by length
            for k, motifs in acc_by_len.items():
                if k <= 0 or k > L:
                    continue
                for i in range(0, L - k + 1):
                    if seq_str[i:i+k] in motifs:
                        event_mask[b, i:i+k] = True

        if not event_mask.any():
            # no events; return connected zero
            return logits.sum() * 0.0

        # Compute CE over all tokens, then mask to event positions
        C = logits.size(-1)
        logits_flat = logits.view(-1, C)
        targets_flat = targets.view(-1)
        ce_vec = F.cross_entropy(logits_flat, targets_flat, reduction='none')  # (B*L,)
        mask_flat = event_mask.view(-1).to(ce_vec.dtype)

        # Optional per-class weights applied only on masked tokens
        cw_vec = _normalize_class_weight_mapping(class_weights)
        if cw_vec is not None:
            cw_vec = cw_vec.to(device=logits.device, dtype=ce_vec.dtype)
            token_w = cw_vec[targets_flat]
        else:
            token_w = torch.ones_like(ce_vec)

        w = token_w * mask_flat
        denom = torch.clamp(w.sum(), min=1.0)
        loss = (ce_vec * w).sum() / denom

        if components_out is not None:
            components_out['event_count'] = int(event_mask.sum().detach().cpu().item())
            components_out['event_fraction'] = float((event_mask.sum().detach().cpu() / (B * L)).item())

        return loss

    return event_based_ce_loss


def event_based_bce_loss_factory(
    dss_motifs: Iterable[str],
    pos_weights: Optional[Union[Sequence[Optional[float]], Dict[Union[int, str], Optional[float]]]] = None,
    neg_weights: Optional[Union[Sequence[Optional[float]], Dict[Union[int, str], Optional[float]]]] = None,
):
    """
    Build an event-based BCE loss that computes binary cross-entropy for each event class
    (START, STOP, DSS, ASS) over only that class's motif spans. For each class c:
      - y_c = 1 if target == c else 0
      - p_c = softmax(logits)[..., c]
      - loss_c per token = -[y_c log p_c + (1-y_c) log(1-p_c)]
      - per-token weight = y_c * pos_weight[c] + (1-y_c) * neg_weight[c]
    The final loss is the weighted mean over all included tokens across the four classes.
    """
    dss_set: Set[str] = set(m.upper() for m in dss_motifs)

    # Prepare per-class positive/negative weight vectors
    pos_vec = _normalize_class_weight_mapping(pos_weights)
    neg_vec = _normalize_class_weight_mapping(neg_weights)

    def event_based_bce_loss(sequences, targets, logits, components_out: Dict[str, Any]):
        device = logits.device
        B, L = sequences.shape
        C = logits.size(-1)

        # Convert token ids to base chars for motif scanning
        idx_to_bp = DNAEmbed.idx_to_bp

        # Build class-specific event masks of shape (B, L)
        event_masks: Dict[int, torch.Tensor] = {
            P.START: torch.zeros((B, L), dtype=torch.bool, device=device),
            P.STOP: torch.zeros((B, L), dtype=torch.bool, device=device),
            P.DSS: torch.zeros((B, L), dtype=torch.bool, device=device),
            P.ASS: torch.zeros((B, L), dtype=torch.bool, device=device),
        }

        from collections import defaultdict
        stop_by_len: Dict[int, Set[str]] = defaultdict(set)
        for m in ConventionalStopCodons:
            stop_by_len[len(m)].add(m.upper())
        acc_by_len: Dict[int, Set[str]] = defaultdict(set)
        for m in ConventionalAcceptorDinucleotides:
            acc_by_len[len(m)].add(m.upper())
        dss_by_len: Dict[int, Set[str]] = defaultdict(set)
        for m in dss_set:
            dss_by_len[len(m)].add(m.upper())

        for b in range(B):
            bases = [idx_to_bp[int(sequences[b, i].item())] for i in range(L)]
            seq_str = ''.join(bases)

            # START spans (ATG triplets)
            k = 3
            for i in range(0, L - k + 1):
                if seq_str[i:i+k] == 'ATG':
                    event_masks[P.START][b, i:i+k] = True

            # STOP spans
            for k, motifs in stop_by_len.items():
                if k <= 0 or k > L:
                    continue
                for i in range(0, L - k + 1):
                    if seq_str[i:i+k] in motifs:
                        event_masks[P.STOP][b, i:i+k] = True

            # DSS spans
            for k, motifs in dss_by_len.items():
                if k <= 0 or k > L:
                    continue
                for i in range(0, L - k + 1):
                    if seq_str[i:i+k] in motifs:
                        event_masks[P.DSS][b, i:i+k] = True

            # ASS spans
            for k, motifs in acc_by_len.items():
                if k <= 0 or k > L:
                    continue
                for i in range(0, L - k + 1):
                    if seq_str[i:i+k] in motifs:
                        event_masks[P.ASS][b, i:i+k] = True

        # If no class has any events, return connected zero
        any_events = any(mask.any().item() for mask in event_masks.values())
        if not any_events:
            return logits.sum() * 0.0

        # Probabilities
        probs = torch.softmax(logits, dim=-1)  # (B, L, C)

        total_weighted_loss = logits.sum() * 0.0  # connected zero
        total_weight = logits.new_tensor(0.0)

        # Prepare per-class pos/neg weight vectors with defaults 1.0
        default_pos = torch.ones(C, dtype=probs.dtype, device=probs.device)
        default_neg = torch.ones(C, dtype=probs.dtype, device=probs.device)
        if pos_vec is not None:
            default_pos[: len(pos_vec)] = pos_vec.to(device=probs.device, dtype=probs.dtype)
        if neg_vec is not None:
            default_neg[: len(neg_vec)] = neg_vec.to(device=probs.device, dtype=probs.dtype)

        # Compute BCE per event class over its event mask
        for cls in (P.START, P.STOP, P.DSS, P.ASS):
            mask = event_masks[cls]
            if not mask.any():
                continue
            p_c = probs[:, :, int(cls)]  # (B, L)
            y_c = (targets == int(cls)).to(dtype=probs.dtype)  # (B, L)
            # Per-token weights
            pos_w = default_pos[int(cls)]
            neg_w = default_neg[int(cls)]
            token_w = torch.where(y_c > 0.5, pos_w, neg_w)  # (B, L)

            # BCE
            eps = 1e-12
            bce = -(y_c * torch.log(torch.clamp(p_c, min=eps)) + (1.0 - y_c) * torch.log(torch.clamp(1.0 - p_c, min=eps)))
            w = token_w * mask.to(token_w.dtype)
            denom = torch.clamp(w.sum(), min=1.0)
            total_weighted_loss = total_weighted_loss + (bce * w).sum() / denom
            total_weight = total_weight + 1.0

        # Average over classes with events
        loss = total_weighted_loss / torch.clamp(total_weight, min=1.0)

        if components_out is not None:
            components_out['bce_event_token_counts'] = {int(k): int(v.sum().detach().cpu().item()) for k, v in event_masks.items()}

        return loss

    return event_based_bce_loss



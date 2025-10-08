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


def adjusted_ce_entropy_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    loss_window_margin_bp: int = 0,
    class_weights: Optional[Union[Sequence[Optional[float]], Dict[Union[int, str], Optional[float]]]] = None,
    entropy_lambda: float = 0.0,
    fp_beta: float = 0.0,
    components_out: Optional[Dict[str, Any]] = None,
) -> torch.Tensor:
    """
    Cross-entropy over included tokens with optional edge masking, entropy regularization,
    and false-positive penalty. Mirrors GenePredictorModule._compute_adjusted_loss behavior.

    Args:
        logits: (B, L, C) raw logits
        targets: (B, L) class indices
        loss_window_margin_bp: number of base-pairs to exclude on each window edge
        class_weights: per-class weights; accepts sequence by id or dict keyed by id or class name
        entropy_lambda: coefficient for entropy regularization (subtracts lambda * H)
        fp_beta: coefficient for one-vs-rest BCE false-positive penalty on weighted classes
        components_out: optional dict to receive component values
    """
    batch_size, seq_length, num_classes = logits.shape
    logits_flat = logits.view(-1, num_classes)
    targets_flat = targets.view(-1)

    # Edge mask (exclude window edges)
    margin = int(max(0, min(seq_length // 2, int(loss_window_margin_bp))))
    if margin > 0 and seq_length > 2 * margin:
        center = torch.ones(seq_length, dtype=torch.float32, device=logits.device)
        center[:margin] = 0.0
        center[-margin:] = 0.0
        include_mask = center.unsqueeze(0).expand(batch_size, -1).contiguous().view(-1) > 0
    else:
        include_mask = torch.ones_like(targets_flat, dtype=torch.bool)

    if not include_mask.any():
        # Return a zero loss connected to the graph to allow backward()
        return logits_flat.sum() * 0.0

    # Shared primitives (compute once)
    log_probs = F.log_softmax(logits_flat, dim=-1)
    probs = torch.exp(log_probs)

    # Per-token CE (unreduced) via negative log-likelihood of true class
    idx = torch.arange(logits_flat.size(0), device=logits.device)
    ce_vec = -log_probs[idx, targets_flat]

    # Entropy per-token
    ent_vec = -(probs * torch.log(torch.clamp(probs, min=1e-12))).sum(dim=-1)

    # Build unified per-token weights (mask × class weights if provided)
    cw_vec = _normalize_class_weight_mapping(class_weights)
    if cw_vec is not None:
        cw = cw_vec.to(device=logits.device, dtype=logits_flat.dtype)
        token_weights_full = cw[targets_flat]
    else:
        token_weights_full = torch.ones_like(ce_vec)

    w = include_mask.to(token_weights_full.dtype) * token_weights_full
    denom = torch.clamp(w.sum(), min=1e-12)

    # Weighted means for CE, entropy, and combined objective
    ce_mean = (ce_vec * w).sum() / denom
    ent_mean = (ent_vec * w).sum() / denom
    lambda_h = float(entropy_lambda)
    ce_entropy_loss = ((ce_vec - lambda_h * ent_vec) * w).sum() / denom

    # Optional FP penalty (one-vs-rest BCE over weighted classes) on included tokens
    beta = float(fp_beta)
    fp_penalty = 0
    if beta > 0.0 and cw_vec is not None:
        cw_full = cw_vec.to(device=logits.device)
        weighted_classes_mask = cw_full > 1.0
        if weighted_classes_mask.any():
            probs_full = probs[include_mask]  # (N_incl, C)
            targets_incl = targets_flat[include_mask]
            # clamp for numerical stability
            probs_full = torch.clamp(probs_full, min=1e-6, max=1.0 - 1e-6)
            # Select only weighted classes columns
            class_indices = torch.nonzero(weighted_classes_mask, as_tuple=False).view(-1)
            p_sel = probs_full[:, class_indices]  # (N_incl, C_w)
            # Build y for selected classes
            y_sel = (targets_incl.view(-1, 1) == class_indices.view(1, -1)).to(p_sel.dtype)
            bce = -(y_sel * torch.log(p_sel) + (1.0 - y_sel) * torch.log(1.0 - p_sel))
            # Keep FP penalty unweighted: mean over tokens and selected classes
            fp_penalty = bce.mean()
    else:
        beta = 0
        fp_penalty = 0

    total_loss = ce_entropy_loss + beta * fp_penalty

    # Optionally emit loss components (overall and per-class CE aggregates)
    if components_out is not None:
        # Per-class CE weighted sums and denominators (included tokens only)
        ce_weighted_sum_by_class: Dict[int, float] = {}
        weight_sum_by_class: Dict[int, float] = {}
        total_weighted_ce_sum = float(((ce_vec * w).sum()).detach().cpu().item())
        C = int(logits.shape[-1])
        for k in range(C):
            cls_mask = include_mask & (targets_flat == k)
            if cls_mask.any():
                num_k = float(((ce_vec[cls_mask] * token_weights_full[cls_mask]).sum()).detach().cpu().item())
                den_k = float((token_weights_full[cls_mask].sum()).detach().cpu().item())
                ce_weighted_sum_by_class[int(k)] = num_k
                weight_sum_by_class[int(k)] = den_k

        components_out['total'] = float(total_loss.detach().cpu().item())
        components_out['ce'] = float(ce_mean.detach().cpu().item())
        components_out['entropy'] = float(ent_mean.detach().cpu().item())
        components_out['fp_penalty'] = float((fp_penalty if isinstance(fp_penalty, torch.Tensor) else torch.tensor(fp_penalty)).detach().cpu().item())
        components_out['ce_weighted_sum_by_class'] = ce_weighted_sum_by_class
        components_out['weight_sum_by_class'] = weight_sum_by_class
        components_out['total_weighted_ce_sum'] = total_weighted_ce_sum

    return total_loss


def build_event_motifs(dss_motifs: Iterable[str]) -> Dict[int, Set[str]]:
    """
    Build the default event-motifs mapping used by event-based losses.

    Returns a dict keyed by class id (GenePredictionClass) with a set of uppercase motifs.
    Includes defaults for START, STOP, ASS, and uses the provided DSS motifs.
    """
    return {
        int(P.START): {'ATG'},
        int(P.STOP): set(m.upper() for m in ConventionalStopCodons),
        int(P.DSS): set(m.upper() for m in dss_motifs),
        int(P.ASS): set(m.upper() for m in ConventionalAcceptorDinucleotides),
    }


def _normalize_event_motifs_map(event_motifs_by_class: Dict[Union[int, str], Iterable[str]]) -> Dict[int, Set[str]]:
    normalized: Dict[int, Set[str]] = {}
    for key, motifs in event_motifs_by_class.items():
        if isinstance(key, int):
            cls_id = int(key)
        else:
            found = None
            for cid, cname in P.idx_to_cls.items():
                if str(key).upper() == str(cname).upper():
                    found = int(cid)
                    break
            if found is None:
                continue
            cls_id = found
        normalized[cls_id] = set(m.upper() for m in motifs)
    return normalized


def event_based_ce_loss_factory(
    event_motifs_by_class: Dict[Union[int, str], Iterable[str]],
    *,
    class_weights: Optional[Union[Sequence[Optional[float]], Dict[Union[int, str], Optional[float]]]] = None,
    loss_window_margin_bp: Optional[int] = None,
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
    normalized_motifs = _normalize_event_motifs_map(event_motifs_by_class)

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

        # Precompute motif groups by length for all classes
        from collections import defaultdict
        by_len_by_class: Dict[int, Dict[int, Set[str]]] = {}
        for cls_id, motif_set in normalized_motifs.items():
            lens: Dict[int, Set[str]] = defaultdict(set)
            for m in motif_set:
                lens[len(m)].add(m.upper())
            by_len_by_class[int(cls_id)] = lens

        for b in range(B):
            # Convert to list of chars and string for substring checks
            bases = [idx_to_bp[int(sequences[b, i].item())] for i in range(L)]
            seq_str = ''.join(bases)

            # Generic scan across all configured classes/motifs
            for cls_id, lens in by_len_by_class.items():
                for k, motifs in lens.items():
                    if k <= 0 or k > L:
                        continue
                    for i in range(0, L - k + 1):
                        if seq_str[i:i+k] in motifs:
                            event_mask[b, i:i+k] = True

        # Optional edge masking (exclude window edges)
        if loss_window_margin_bp is not None and int(loss_window_margin_bp) > 0:
            m = int(loss_window_margin_bp)
            m = max(0, min(L // 2, m))
            if m > 0 and L > 2 * m:
                center = torch.ones(L, dtype=torch.bool, device=device)
                center[:m] = False
                center[-m:] = False
                event_mask = event_mask & center.unsqueeze(0).expand(B, -1)

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
    event_motifs_by_class: Dict[Union[int, str], Iterable[str]],
    *,
    pos_weights: Optional[Union[Sequence[Optional[float]], Dict[Union[int, str], Optional[float]]]] = None,
    neg_weights: Optional[Union[Sequence[Optional[float]], Dict[Union[int, str], Optional[float]]]] = None,
    loss_window_margin_bp: Optional[int] = None,
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
    normalized_motifs = _normalize_event_motifs_map(event_motifs_by_class)

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
        event_masks: Dict[int, torch.Tensor] = {int(k): torch.zeros((B, L), dtype=torch.bool, device=device) for k in normalized_motifs.keys()}

        from collections import defaultdict
        by_len_by_class: Dict[int, Dict[int, Set[str]]] = {}
        for cls_id, motif_set in normalized_motifs.items():
            lens: Dict[int, Set[str]] = defaultdict(set)
            for m in motif_set:
                lens[len(m)].add(m.upper())
            by_len_by_class[int(cls_id)] = lens

        for b in range(B):
            bases = [idx_to_bp[int(sequences[b, i].item())] for i in range(L)]
            seq_str = ''.join(bases)

            # Generic spans for all configured classes
            for cls_id, lens in by_len_by_class.items():
                for k, motifs in lens.items():
                    if k <= 0 or k > L:
                        continue
                    for i in range(0, L - k + 1):
                        if seq_str[i:i+k] in motifs:
                            event_masks[int(cls_id)][b, i:i+k] = True

        # Optional edge masking: build center mask once
        center_mask = None
        if loss_window_margin_bp is not None and int(loss_window_margin_bp) > 0:
            m = int(loss_window_margin_bp)
            m = max(0, min(L // 2, m))
            if m > 0 and L > 2 * m:
                cm = torch.ones((B, L), dtype=torch.bool, device=device)
                cm[:, :m] = False
                cm[:, -m:] = False
                center_mask = cm

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
        for cls in event_masks.keys():
            mask = event_masks[cls]
            if center_mask is not None:
                mask = mask & center_mask
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



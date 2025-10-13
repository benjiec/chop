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
from utils.events import (
    normalize_event_motifs_map,
    group_motifs_by_length,
    build_event_masks,
    build_center_mask,
    normalize_class_weight_mapping,
)


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
    cw_vec = normalize_class_weight_mapping(class_weights)
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
    normalized_motifs = normalize_event_motifs_map(event_motifs_by_class)

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
        per_class_masks = build_event_masks(sequences, normalized_motifs)
        for m in per_class_masks.values():
            event_mask |= m

        # Optional edge masking (exclude window edges)
        if loss_window_margin_bp is not None and int(loss_window_margin_bp) > 0:
            center = build_center_mask(B, L, int(loss_window_margin_bp), device=device)
            event_mask = event_mask & center

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
        cw_vec = normalize_class_weight_mapping(class_weights)
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
    normalized_motifs = normalize_event_motifs_map(event_motifs_by_class)

    # Prepare per-class positive/negative weight vectors
    pos_vec = normalize_class_weight_mapping(pos_weights)
    neg_vec = normalize_class_weight_mapping(neg_weights)

    def event_based_bce_loss(sequences, targets, logits, components_out: Dict[str, Any]):
        device = logits.device
        B, L = sequences.shape
        C = logits.size(-1)

        # Convert token ids to base chars for motif scanning
        idx_to_bp = DNAEmbed.idx_to_bp

        # Build class-specific event masks of shape (B, L)
        event_masks: Dict[int, torch.Tensor] = build_event_masks(sequences, normalized_motifs)

        # Optional edge masking: build center mask once
        center_mask = None
        if loss_window_margin_bp is not None and int(loss_window_margin_bp) > 0:
            center_mask = build_center_mask(B, L, int(loss_window_margin_bp), device=device)

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


def _event_masked_bce_with_logits(
    *,
    sequences: torch.Tensor,
    targets: torch.Tensor,
    per_head_event_masks: Dict[int, torch.Tensor],
    logits_like: torch.Tensor,  # shape (B, L, H)
    pos_vec: Optional[torch.Tensor],
    neg_vec: Optional[torch.Tensor],
    center_mask: Optional[torch.Tensor] = None,
    alpha_weights: Optional[torch.Tensor] = None,
    components_out: Optional[Dict[str, Any]] = None,
) -> torch.Tensor:
    """Compute event-masked BCEWithLogits for per-head logits in logits_like.

    - logits_like[..., h] is the raw logit for event head h
    - per_head_event_masks[h] selects positions to include for head h
    - pos_vec/neg_vec are per-head positive/negative weights (length H)
    - alpha_weights weights each head term in the final average
    """
    device = logits_like.device
    dtype = logits_like.dtype
    B, L, H = logits_like.shape

    # Defaults
    if pos_vec is None:
        pos_vec = torch.ones(H, device=device, dtype=dtype)
    if neg_vec is None:
        neg_vec = torch.ones(H, device=device, dtype=dtype)
    if alpha_weights is None:
        alpha_weights = torch.ones(H, device=device, dtype=dtype)

    total = logits_like.sum() * 0.0
    total_alpha = torch.zeros((), device=device, dtype=dtype)

    for h in range(H):
        mask = per_head_event_masks.get(h)
        if mask is None:
            continue
        m = mask
        if center_mask is not None:
            m = m & center_mask
        if not m.any():
            continue
        z = logits_like[:, :, h]
        y = (targets == int(h)).to(dtype=dtype)
        token_w = torch.where(y > 0.5, pos_vec[h], neg_vec[h])
        # Masked mean BCEWithLogits
        bce = torch.nn.functional.binary_cross_entropy_with_logits(z, y, reduction='none')
        w = token_w * m.to(dtype)
        denom = torch.clamp(w.sum(), min=1.0)
        head_loss = (bce * w).sum() / denom
        # Record per-head loss components if requested
        if components_out is not None:
            key = f"loss_head_{int(h)}"
            key_w = f"loss_head_{int(h)}_weighted"
            components_out[key] = float(head_loss.detach().cpu().item())
            components_out[key_w] = float((alpha_weights[h] * head_loss).detach().cpu().item())
        total = total + alpha_weights[h] * head_loss
        total_alpha = total_alpha + alpha_weights[h]

    loss_total = total / torch.clamp(total_alpha, min=1.0)
    if components_out is not None:
        components_out['loss_event_heads_total'] = float(loss_total.detach().cpu().item())
    return loss_total


def event_head_bce_loss_factory(
    event_motifs_by_head_idx: Dict[int, Iterable[str]],
    *,
    pos_weights_by_head_idx: Optional[Union[Sequence[Optional[float]], Dict[int, Optional[float]]]] = None,
    neg_weights_by_head_idx: Optional[Union[Sequence[Optional[float]], Dict[int, Optional[float]]]] = None,
    alpha_weights_by_head_idx: Optional[Union[Sequence[Optional[float]], Dict[int, Optional[float]]]] = None,
    loss_window_margin_bp: Optional[int] = None,
):
    """
    Event-head BCEWithLogits loss. Expects event_logits from separate per-class heads.

    Returns a callable (sequences, targets, logits, event_logits, components_out) -> scalar loss.
    """
    # Expect head-indexed motifs; normalize only motifs to uppercase
    normalized_motifs: Dict[int, set[str]] = {}
    for h, motifs in event_motifs_by_head_idx.items():
        normalized_motifs[int(h)] = set(str(mm).upper() for mm in motifs)

    # Note: event-head weights are indexed by head-id 0..H-1, not GenePredictionClass ids.
    # We therefore normalize them inside loss_fn once H is known.

    def loss_fn(sequences, targets, logits, event_logits, components_out: Dict[str, Any]):
        if event_logits is None:
            # Connected zero with same device
            return logits.sum() * 0.0
        device = event_logits.device
        B, L, H = event_logits.shape

        # Build class-specific event masks
        event_masks: Dict[int, torch.Tensor] = build_event_masks(sequences, normalized_motifs)

        # Optional edge masking
        center_mask = None
        if loss_window_margin_bp is not None and int(loss_window_margin_bp) > 0:
            center_mask = build_center_mask(B, L, int(loss_window_margin_bp), device=device)

        # Normalize per-head weight specifications to tensors length H
        def normalize_head_weights(src: Optional[Union[Sequence[Optional[float]], Dict[int, Optional[float]]]], default_value: float = 1.0) -> torch.Tensor:
            vec = torch.full((H,), float(default_value), device=device, dtype=event_logits.dtype)
            if src is None:
                return vec
            if isinstance(src, dict):
                for k, v in src.items():
                    if v is None:
                        continue
                    try:
                        idx = int(k)
                    except Exception:
                        continue
                    if 0 <= idx < H:
                        vec[idx] = float(v)
            else:
                # Sequence-like
                for i, w in enumerate(list(src)[:H]):
                    if w is None:
                        continue
                    vec[i] = float(w)
            return vec

        pos_v = normalize_head_weights(pos_weights_by_head_idx, 1.0)
        neg_v = normalize_head_weights(neg_weights_by_head_idx, 1.0)
        alp_v = normalize_head_weights(alpha_weights_by_head_idx, 1.0)

        loss = _event_masked_bce_with_logits(
            sequences=sequences,
            targets=targets,
            per_head_event_masks=event_masks,
            logits_like=event_logits,
            pos_vec=pos_v,
            neg_vec=neg_v,
            center_mask=center_mask,
            alpha_weights=alp_v,
            components_out=components_out,
        )

        if components_out is not None:
            components_out['event_count_total'] = int(sum(int(m.sum().detach().cpu().item()) for m in event_masks.values()))
        return loss

    return loss_fn

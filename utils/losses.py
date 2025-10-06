#!/usr/bin/env python3

import torch
import torch.nn.functional as F
from typing import Iterable, Set, Dict, Any

from utils.constants import (
    DNAEmbed,
    GenePredictionClass as P,
    ConventionalStopCodons,
    ConventionalAcceptorDinucleotides,
)


def event_based_ce_loss_factory(dss_motifs: Iterable[str]):
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
        denom = torch.clamp(mask_flat.sum(), min=1.0)
        loss = (ce_vec * mask_flat).sum() / denom

        if components_out is not None:
            components_out['event_count'] = int(event_mask.sum().detach().cpu().item())
            components_out['event_fraction'] = float((event_mask.sum().detach().cpu() / (B * L)).item())

        return loss

    return event_based_ce_loss



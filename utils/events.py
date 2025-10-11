#!/usr/bin/env python3

from typing import Dict, Iterable, Optional, Sequence, Set, Union

import torch

from utils.constants import DNAEmbed
from utils.constants import GenePredictionClass as P
from utils.constants import ConventionalStopCodons, ConventionalAcceptorDinucleotides


def normalize_event_motifs_map(event_motifs_by_class: Dict[Union[int, str], Iterable[str]]) -> Dict[int, Set[str]]:
    """Normalize a mapping of class -> motifs to use integer class ids and uppercase motifs.

    Accepts class keys as int ids or class-name strings matching GenePredictionClass.
    Motifs are uppercased and collected into sets.
    """
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
        normalized[int(cls_id)] = set(str(m).upper() for m in motifs)
    return normalized


def group_motifs_by_length(normalized_motifs: Dict[int, Set[str]]) -> Dict[int, Dict[int, Set[str]]]:
    """Group normalized motifs by length for each class.

    Returns: dict[class_id] -> dict[length] -> set[motif]
    """
    from collections import defaultdict
    grouped: Dict[int, Dict[int, Set[str]]] = {}
    for cls_id, motif_set in normalized_motifs.items():
        lens: Dict[int, Set[str]] = defaultdict(set)
        for m in motif_set:
            lens[len(m)].add(str(m).upper())
        grouped[int(cls_id)] = lens
    return grouped


def convert_tokens_to_sequence(tokens) -> str:
    """Convert token indices to DNA bases using DNAEmbed.idx_to_bp mapping."""
    idx_to_nucleotide = DNAEmbed.idx_to_bp
    return ''.join([idx_to_nucleotide.get(int(token), 'N') for token in tokens])


def build_event_masks(
    sequences: torch.Tensor,
    normalized_motifs: Dict[int, Set[str]],
) -> Dict[int, torch.Tensor]:
    """Build per-class event masks for a batch of sequences.

    Args:
        sequences: (B, L) token ids
        normalized_motifs: dict[class_id] -> set of uppercase motif strings

    Returns:
        dict[class_id] -> (B, L) boolean mask with motif spans marked True
    """
    B, L = sequences.shape
    masks: Dict[int, torch.Tensor] = {int(k): torch.zeros((B, L), dtype=torch.bool, device=sequences.device) for k in normalized_motifs.keys()}
    by_len = group_motifs_by_length(normalized_motifs)

    # Convert tokens to strings per batch
    idx_to_bp = DNAEmbed.idx_to_bp
    for b in range(B):
        bases = [idx_to_bp.get(int(sequences[b, i].item()), 'N') for i in range(L)]
        seq_str = ''.join(bases)
        for cls_id, lens in by_len.items():
            for k, motif_set in lens.items():
                if k <= 0 or k > L:
                    continue
                for i in range(0, L - k + 1):
                    if seq_str[i:i+k] in motif_set:
                        masks[int(cls_id)][b, i:i+k] = True
    return masks


def build_center_mask(B: int, L: int, margin_bp: int | None, device: torch.device | None = None) -> torch.Tensor:
    """Build a (B, L) boolean mask that excludes margin_bp tokens on both ends.

    If margin_bp is None or <= 0 or the sequence is too short, the mask is all True.
    """
    if margin_bp is None or int(margin_bp) <= 0 or L <= 2 * int(margin_bp):
        return torch.ones((B, L), dtype=torch.bool, device=device)
    m = int(max(0, min(L // 2, int(margin_bp))))
    center = torch.ones((L,), dtype=torch.bool, device=device)
    center[:m] = False
    center[-m:] = False
    return center.unsqueeze(0).expand(B, -1)



def build_event_motifs(dss_motifs: Iterable[str]) -> Dict[int, Set[str]]:
    """Build the default event-motifs mapping for START, STOP, DSS, ASS.

    Returns a dict keyed by integer class id with a set of uppercase motifs.
    """

    assert type(dss_motifs) != type("")

    return {
        int(P.START): {'ATG'},
        int(P.STOP): set(m.upper() for m in ConventionalStopCodons),
        int(P.DSS): set(m.upper() for m in dss_motifs),
        int(P.ASS): set(m.upper() for m in ConventionalAcceptorDinucleotides),
    }


def normalize_class_weight_mapping(
    class_weights: Optional[Union[Sequence[Optional[float]], Dict[Union[int, str], Optional[float]]]],
) -> Optional[torch.Tensor]:
    """Normalize class-weight specifications into a dense tensor of shape (num_classes,).

    Accepts:
      - Sequence[float | None] indexed by class id
      - Dict[int | str, float | None] where int is class id or str is class name in GenePredictionClass
    Returns None if class_weights is None.
    """
    if class_weights is None:
        return None

    id_to_weight: Dict[int, float] = {}
    if isinstance(class_weights, dict):
        for k, v in class_weights.items():
            if v is None:
                continue
            if isinstance(k, int):
                id_to_weight[int(k)] = float(v)
            else:
                # Map class-name string to id
                for cid, cname in P.idx_to_cls.items():
                    if str(k).upper() == str(cname).upper():
                        id_to_weight[int(cid)] = float(v)
                        break
    else:
        for cid, w in enumerate(class_weights):
            if w is None:
                continue
            id_to_weight[int(cid)] = float(w)

    num_classes = len(P.idx_to_cls)
    vec = torch.ones(num_classes, dtype=torch.float32)
    for cid, w in id_to_weight.items():
        if 0 <= int(cid) < num_classes:
            vec[int(cid)] = float(w)
    return vec


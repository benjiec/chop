#!/usr/bin/env python3

from typing import Dict, Iterable, Optional, Sequence, Set, Union, Tuple, List
import numpy as np
import torch

import torch

from utils.constants import DNAEmbed
from utils.constants import GenePredictionClass as P
from utils.constants import ConventionalStopCodons, ConventionalAcceptorDinucleotides
from functools import lru_cache


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

    Delegates to the vectorized implementation for performance.
    """
    return build_event_masks_vectorized(sequences, normalized_motifs)


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


def build_event_window_logits(
    seq_window_tokens: torch.Tensor,
    event_logits_window: torch.Tensor,
    event_motifs_by_class: Dict[int, Set[str]],
    head_class_ids: Sequence[int],
    num_classes: int,
    margin_bp: Optional[int] = 0,
) -> np.ndarray:
    """Build a (win_len, num_classes) array of per-class LOGITS for a single window.

    For each head h mapped to class c, copy the event head logits into column c at
    positions included by the event mask (and center mask). Other positions remain 0.
    """
    assert isinstance(seq_window_tokens, torch.Tensor) and seq_window_tokens.dim() == 2 and int(seq_window_tokens.size(0)) == 1
    assert isinstance(event_logits_window, torch.Tensor) and event_logits_window.dim() == 3 and int(event_logits_window.size(0)) == 1
    B, L, H = int(event_logits_window.shape[0]), int(event_logits_window.shape[1]), int(event_logits_window.shape[2])
    device = seq_window_tokens.device

    # Build class masks once
    per_class_masks_t = build_event_masks(seq_window_tokens, {int(c): set(str(m).upper() for m in motifs)
                                                             for c, motifs in event_motifs_by_class.items()})
    center_mask_t = build_center_mask(1, L, int(margin_bp) if margin_bp is not None else 0, device=device)
    center_mask_1d = center_mask_t[0]

    wl = np.zeros((L, int(num_classes)), dtype=np.float32)
    ev_np = event_logits_window[0].detach().cpu().numpy()

    for h in range(H):
        cls_id = int(head_class_ids[h]) if h < len(head_class_ids) else -1
        if cls_id < 0 or cls_id >= int(num_classes):
            continue
        mc_t = per_class_masks_t.get(int(cls_id))
        if mc_t is None:
            continue
        mask = (mc_t[0] & center_mask_1d).detach().cpu().numpy()
        if not mask.any():
            continue
        wl[mask, cls_id] = ev_np[mask, h].astype(np.float32)

    return wl


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


@lru_cache(maxsize=64)
def _cached_motif_tensors(key: Tuple[Tuple[int, Tuple[str, ...]], ...]) -> Dict[int, Tuple[torch.Tensor, Dict[int, List[int]]]]:
    """Cache encoded motif tensors grouped by length.

    Returns dict[length] -> (motifs_tensor(M,k), class_to_indices)
    motifs_tensor contains ints 0..3 for A/T/G/C per motif row.
    class_to_indices maps class_id -> list of motif row indices for that class.
    """
    # Reconstruct mapping from key
    motifs_by_class: Dict[int, Set[str]] = {int(cls): set(mots) for (cls, mots) in key}
    by_len: Dict[int, Dict[int, Set[str]]] = {}
    # Group motifs by length for each class
    from collections import defaultdict
    tmp_group: Dict[int, Dict[int, Set[str]]] = {}
    for cls_id, motif_set in motifs_by_class.items():
        lens: Dict[int, Set[str]] = defaultdict(set)
        for m in motif_set:
            lens[len(m)].add(str(m).upper())
        tmp_group[int(cls_id)] = lens

    # Build consolidated per-length arrays
    out: Dict[int, Tuple[torch.Tensor, Dict[int, List[int]]]] = {}
    for k in sorted({l for lens in tmp_group.values() for l in lens.keys()}):
        rows: List[List[int]] = []
        owners: List[int] = []
        for cls_id, lens in tmp_group.items():
            ms = sorted(lens.get(int(k), set()))
            for mot in ms:
                row: List[int] = []
                for ch in mot:
                    if ch == 'A':
                        row.append(int(DNAEmbed.A))
                    elif ch == 'T':
                        row.append(int(DNAEmbed.T))
                    elif ch == 'G':
                        row.append(int(DNAEmbed.G))
                    elif ch == 'C':
                        row.append(int(DNAEmbed.C))
                    else:
                        # Non-ATGC in motifs is unsupported; encode impossible value 5
                        row.append(5)
                rows.append(row)
                owners.append(int(cls_id))
        if rows:
            mt = torch.tensor(rows, dtype=torch.long)
            class_to_indices: Dict[int, List[int]] = {}
            for idx, cid in enumerate(owners):
                class_to_indices.setdefault(int(cid), []).append(int(idx))
            out[int(k)] = (mt, class_to_indices)
    return out


def _motif_cache_key(normalized_motifs: Dict[int, Set[str]]) -> Tuple[Tuple[int, Tuple[str, ...]], ...]:
    items = []
    for cls_id in sorted(normalized_motifs.keys()):
        ms = tuple(sorted(str(m).upper() for m in normalized_motifs[int(cls_id)]))
        items.append((int(cls_id), ms))
    return tuple(items)


def build_event_masks_vectorized(
    sequences: torch.Tensor,
    normalized_motifs: Dict[int, Set[str]],
) -> Dict[int, torch.Tensor]:
    """Vectorized build of per-class event masks for a batch of sequences.

    Args:
        sequences: (B, L) token ids
        normalized_motifs: dict[class_id] -> set of uppercase motif strings

    Returns:
        dict[class_id] -> (B, L) boolean mask with motif spans marked True
    """
    assert isinstance(sequences, torch.Tensor) and sequences.dim() == 2
    B, L = int(sequences.size(0)), int(sequences.size(1))
    device = sequences.device

    # Initialize output masks per class on device
    masks: Dict[int, torch.Tensor] = {int(k): torch.zeros((B, L), dtype=torch.bool, device=device) for k in normalized_motifs.keys()}

    if L <= 0 or B <= 0 or not normalized_motifs:
        return masks

    key = _motif_cache_key(normalized_motifs)
    by_len_cached = _cached_motif_tensors(key)

    # For each motif length k, match all motifs at once using unfold and broadcasting
    for k, (motifs_k_cpu, class_to_idx) in by_len_cached.items():
        if int(k) <= 0 or int(k) > L:
            continue
        # Windows: (B, L-k+1, k)
        win = sequences.unfold(dimension=1, size=int(k), step=1)  # (B, L-k+1, k)
        # Move motif tensor to device; shape (M, k)
        motifs_k = motifs_k_cpu.to(device=device, dtype=torch.long)
        if motifs_k.numel() == 0:
            continue
        # Compare windows to all motifs: (B, L-k+1, 1, k) == (1, 1, M, k)
        eq = (win.unsqueeze(2) == motifs_k.view(1, 1, motifs_k.shape[0], int(k)))
        # Reduce over k to get start matches per motif: (B, L-k+1, M)
        starts_per_motif = eq.all(dim=-1)

        # For each class, OR the motifs and expand starts to spans of length k
        for cls_id, idxs in class_to_idx.items():
            if not idxs:
                continue
            # (B, L-k+1)
            starts = starts_per_motif.index_select(dim=2, index=torch.tensor(idxs, device=device)).any(dim=2)
            if not starts.any():
                continue
            # Expand to spans: for offset 0..k-1, shift and OR
            span = masks[int(cls_id)]
            # For each offset, map starts[:, s] -> positions s+offset
            for off in range(int(k)):
                # span[:, off:off + (L-k+1)] |= starts
                s_len = L - int(k) + 1
                span_slice = span[:, off:off + s_len]
                span_slice |= starts
            masks[int(cls_id)] = span

    return masks


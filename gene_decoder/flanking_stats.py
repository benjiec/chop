from typing import Dict, Tuple, List, Optional, Set, Literal
from dataclasses import dataclass

import numpy as np
import torch

from utils.constants import (
    GenePredictionClass as P,
    StandardDonorDinucleotides,
    DinoDonorDinucleotides,
    ConventionalAcceptorDinucleotides,
)
from utils.genome import AnnotatedGenomeDataset
from utils.events import compute_event_spans_vectorized
from utils.metrics import convert_tokens_to_sequence
from gene_decoder import PredictedSequence


SiteKind = Literal['ASS', 'DSS', 'both']


@dataclass
class FlankingCounts:
    total: int = 0
    positives: int = 0
    negatives: int = 0

    def add(self, is_positive: bool) -> None:
        self.total += 1
        if is_positive:
            self.positives += 1
        else:
            self.negatives += 1


def _labels_match_span(labels: np.ndarray, pos: int, cls_idx: int, span_len: int) -> bool:
    s = int(pos)
    e = min(labels.shape[0], s + int(span_len))
    if e <= s:
        return False
    window = labels[s:e]
    return bool(np.all(window.astype(int) == int(cls_idx)))


def _build_motifs_map(dss_motifs_mode: str) -> Dict[int, Set[str]]:
    dss_set: Set[str] = set(StandardDonorDinucleotides)
    if dss_motifs_mode == 'dino':
        dss_set = dss_set.union(DinoDonorDinucleotides)
    return {
        int(P.DSS): set(m.upper() for m in dss_set),
        int(P.ASS): set(m.upper() for m in ConventionalAcceptorDinucleotides),
    }


def _extract_flank(sequence: str, start: int, end_excl: int, flank: int) -> Optional[str]:
    L = len(sequence)
    s = int(start) - int(flank)
    e = int(end_excl) + int(flank)
    if s < 0 or e > L or e <= s:
        return None
    return sequence[s:e]


def compute_flanking_motif_stats(
    fna_fn: str,
    tsv_fn: str,
    *,
    flank: int = 3,
    site: SiteKind = 'both',
    dss_motifs_mode: str = 'standard',
    num_contigs: int = 0,
) -> Tuple[Dict[str, FlankingCounts], Dict[str, FlankingCounts]]:
    """Compute flanking motif statistics around DSS/ASS sites.

    Returns a pair of dicts (dss_counts, ass_counts) mapping flanking 8-mer/10-mer strings
    to FlankingCounts. If `site` filters to one kind, the other dict will be empty.
    """

    if int(flank) < 0:
        raise ValueError("flank must be >= 0")

    dataset = AnnotatedGenomeDataset(
        fna_fn,
        tsv_fn,
        window=None,
        num_contigs=num_contigs,
        random_prefix_ns=False,
    )

    motifs_by_class = _build_motifs_map(dss_motifs_mode)

    want_dss = (site in ('DSS', 'both'))
    want_ass = (site in ('ASS', 'both'))

    dss_counts: Dict[str, FlankingCounts] = {}
    ass_counts: Dict[str, FlankingCounts] = {}

    for seq_idx in range(len(dataset)):
        item = dataset[seq_idx]
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            seq_tokens, labels = item[0], item[1]
        else:
            seq_tokens, labels = item
        sequence = convert_tokens_to_sequence(seq_tokens)
        L = len(sequence)
        tokens_t = torch.tensor(seq_tokens, dtype=torch.long).unsqueeze(0)[0]
        spans_map = compute_event_spans_vectorized(tokens_t, motifs_by_class)

        if want_dss:
            for (s, e) in spans_map.get(int(P.DSS), []):
                flank_str = _extract_flank(sequence, s, e, flank)
                if flank_str is None:
                    continue
                is_tp = _labels_match_span(np.asarray(labels), s, int(P.DSS), e - s)
                c = dss_counts.get(flank_str)
                if c is None:
                    c = FlankingCounts()
                    dss_counts[flank_str] = c
                c.add(is_tp)

        if want_ass:
            for (s, e) in spans_map.get(int(P.ASS), []):
                flank_str = _extract_flank(sequence, s, e, flank)
                if flank_str is None:
                    continue
                is_tp = _labels_match_span(np.asarray(labels), s, int(P.ASS), e - s)
                c = ass_counts.get(flank_str)
                if c is None:
                    c = FlankingCounts()
                    ass_counts[flank_str] = c
                c.add(is_tp)

    return dss_counts, ass_counts


def format_counts_as_csv(site_label: str, counts: Dict[str, FlankingCounts], include_header: bool = False) -> List[str]:
    """Return CSV lines for the given counts mapping with a leading site column.

    Columns: site,motif,t,p,n,p_over_t
    Sorted by p_over_t (mean) desc, then t desc, then motif asc. Rounds mean for
    stable ordering consistent with printed precision.
    """
    lines: List[str] = []
    if include_header:
        lines.append("site,motif,t,p,n,p_over_t")
    def _sort_key(kv):
        c = kv[1]
        t = int(c.total)
        p = int(c.positives)
        ratio = (float(p) / float(t)) if t > 0 else 0.0
        rkey = round(ratio, 12)
        return (-p, -rkey, -t, str(kv[0]))
    items = sorted(counts.items(), key=_sort_key)
    for motif, c in items:
        t = int(c.total)
        p = int(c.positives)
        n = int(c.negatives)
        ratio = (float(p) / float(t)) if t > 0 else 0.0
        lines.append(f"{site_label},{motif},{t},{p},{n},{ratio:.6f}")
    return lines


def _encode_sequence_to_tokens(sequence: str) -> torch.Tensor:
    from utils.constants import DNAEmbed
    vocab = {'A': int(DNAEmbed.A), 'T': int(DNAEmbed.T), 'G': int(DNAEmbed.G), 'C': int(DNAEmbed.C)}
    toks = [vocab.get(ch, int(DNAEmbed.N)) for ch in sequence]
    return torch.tensor(toks, dtype=torch.long)


def _class_name_to_index(class_order: List[str], class_name: str) -> Optional[int]:
    try:
        return int(class_order.index(str(class_name)))
    except ValueError:
        return None


@dataclass
class FlankingProbStats:
    total: int
    mean: float
    std: float


def compute_flanking_prob_stats_from_items(
    items: List[PredictedSequence],
    *,
    flank: int = 3,
    site: SiteKind = 'both',
    dss_motifs_mode: str = 'standard',
    num_sequences: int = 0,
) -> Tuple[Dict[str, FlankingProbStats], Dict[str, FlankingProbStats]]:
    """Aggregate per-event probabilities by flanking motif from PredictedSequence items.

    For each ASS/DSS span found in the sequence, compute the mean probability for the
    corresponding class over the span (length 2). Group these values by the flanking
    motif [flank bp upstream][2bp motif][flank bp downstream] and compute count, mean, std.
    """
    if int(flank) < 0:
        raise ValueError("flank must be >= 0")

    motifs_by_class = _build_motifs_map(dss_motifs_mode)
    want_dss = (site in ('DSS', 'both'))
    want_ass = (site in ('ASS', 'both'))

    dss_values: Dict[str, List[float]] = {}
    ass_values: Dict[str, List[float]] = {}

    seq_iter = items
    if num_sequences and int(num_sequences) > 0:
        seq_iter = items[:int(num_sequences)]

    for r in seq_iter:
        seq = r.sequence
        probs = r.probabilities
        class_order = r.class_order
        if probs is None or seq is None or class_order is None:
            continue
        L = len(seq)
        if L == 0 or probs.shape[0] != L:
            continue

        dss_col = _class_name_to_index(class_order, 'DSS')
        ass_col = _class_name_to_index(class_order, 'ASS')

        tokens_t = _encode_sequence_to_tokens(seq)
        spans_map = compute_event_spans_vectorized(tokens_t, motifs_by_class)

        if want_dss and dss_col is not None:
            for (s, e) in spans_map.get(int(P.DSS), []):
                flank_str = _extract_flank(seq, s, e, flank)
                if flank_str is None:
                    continue
                s0 = max(0, int(s))
                e0 = min(L, int(e))
                if e0 <= s0:
                    continue
                vals = probs[s0:e0, int(dss_col)].astype(float)
                if vals.size == 0:
                    continue
                finite = np.isfinite(vals)
                if not finite.any():
                    continue
                m = float(np.mean(vals[finite]))
                dss_values.setdefault(flank_str, []).append(m)

        if want_ass and ass_col is not None:
            for (s, e) in spans_map.get(int(P.ASS), []):
                flank_str = _extract_flank(seq, s, e, flank)
                if flank_str is None:
                    continue
                s0 = max(0, int(s))
                e0 = min(L, int(e))
                if e0 <= s0:
                    continue
                vals = probs[s0:e0, int(ass_col)].astype(float)
                if vals.size == 0:
                    continue
                finite = np.isfinite(vals)
                if not finite.any():
                    continue
                m = float(np.mean(vals[finite]))
                ass_values.setdefault(flank_str, []).append(m)

    def _to_stats(src: Dict[str, List[float]]) -> Dict[str, FlankingProbStats]:
        out: Dict[str, FlankingProbStats] = {}
        for motif, arr in src.items():
            a = np.asarray(arr, dtype=float)
            if a.size == 0:
                continue
            out[motif] = FlankingProbStats(total=int(a.size), mean=float(a.mean()), std=float(a.std(ddof=0)))
        return out

    return _to_stats(dss_values), _to_stats(ass_values)


def format_prob_stats_as_csv(site_label: str, stats: Dict[str, FlankingProbStats], include_header: bool = False) -> List[str]:
    lines: List[str] = []
    if include_header:
        lines.append("site,motif,t,mean,std")
    # Sort by mean (rounded for stability) desc, then t desc, then motif asc
    def _prob_sort_key(kv):
        s = kv[1]
        # drop NaNs by treating as -inf; though upstream we avoid NaNs
        m = float(s.mean)
        if not np.isfinite(m):
            m = float('-inf')
        rkey = round(m, 12)
        return (-rkey, -int(s.total), str(kv[0]))
    items = sorted(stats.items(), key=_prob_sort_key)
    for motif, s in items:
        lines.append(f"{site_label},{motif},{int(s.total)},{s.mean:.6f},{s.std:.6f}")
    return lines



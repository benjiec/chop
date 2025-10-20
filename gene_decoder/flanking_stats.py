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
        seq_tokens, labels = dataset[seq_idx]
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
    Sorted by p desc, then t desc, then motif asc.
    """
    lines: List[str] = []
    if include_header:
        lines.append("site,motif,t,p,n,p_over_t")
    items = sorted(counts.items(), key=lambda kv: (-int(kv[1].positives), -int(kv[1].total), str(kv[0])))
    for motif, c in items:
        t = int(c.total)
        p = int(c.positives)
        n = int(c.negatives)
        ratio = (float(p) / float(t)) if t > 0 else 0.0
        lines.append(f"{site_label},{motif},{t},{p},{n},{ratio:.6f}")
    return lines



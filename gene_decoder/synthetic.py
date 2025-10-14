import numpy as np
from typing import Dict, Tuple, List, Optional, Set
from dataclasses import dataclass

from gene_decoder import PredictedSequence
from utils.constants import (
    GenePredictionClass as P,
    ConventionalStopCodons,
    StandardDonorDinucleotides,
    DinoDonorDinucleotides,
    ConventionalAcceptorDinucleotides,
)
from utils.genome import AnnotatedGenomeDataset
from utils.events import compute_event_spans_vectorized
import torch
from utils.metrics import convert_tokens_to_sequence


@dataclass
class MeanStdParams:
    mean: float
    std: float


def _event_span_len(cls_idx: int) -> int:
    if int(cls_idx) in (int(P.START), int(P.STOP)):
        return 3
    if int(cls_idx) in (int(P.DSS), int(P.ASS)):
        return 2
    return 1


def _build_motifs_map(dss_motifs: Set[str]) -> Dict[int, Set[str]]:
    return {
        int(P.START): {'ATG'},
        int(P.STOP): set(m.upper() for m in ConventionalStopCodons),
        int(P.DSS): set(m.upper() for m in dss_motifs),
        int(P.ASS): set(m.upper() for m in ConventionalAcceptorDinucleotides),
    }


def _labels_match_span(labels: np.ndarray, pos: int, cls_idx: int) -> bool:
    span = _event_span_len(cls_idx)
    s = pos
    e = min(labels.shape[0], pos + span)
    if e <= s:
        return False
    # labels may be torch tensor upstream; ensure np array of ints
    window = labels[s:e]
    return bool(np.all(window.astype(int) == int(cls_idx)))


def _sample_beta(alpha: float, beta: float, size: Optional[Tuple[int, ...]] = None) -> np.ndarray:
    # Guard against degenerate params
    a = max(float(alpha), 1e-6)
    b = max(float(beta), 1e-6)
    return np.random.beta(a, b, size=size)


def _sample_normal_clipped(mean: float, std: float) -> float:
    m = float(mean)
    s = float(max(std, 0.0))
    if s <= 0.0:
        return float(np.clip(m, 1e-6, 1.0 - 1e-6))
    v = float(np.random.normal(loc=m, scale=s))
    return float(np.clip(v, 1e-6, 1.0 - 1e-6))


def build_synthetic_decoder_inputs(
    fna_fn: str,
    tsv_fn: str,
    start_tp: MeanStdParams,
    start_tn: MeanStdParams,
    stop_tp: MeanStdParams,
    stop_tn: MeanStdParams,
    dss_tp: MeanStdParams,
    dss_tn: MeanStdParams,
    ass_tp: MeanStdParams,
    ass_tn: MeanStdParams,
    dss_motifs_mode: str = 'standard',
    num_contigs: int = 0,
) -> List[PredictedSequence]:
    """Construct synthetic decoder inputs with event-only probabilities.

    Background probabilities are zero everywhere. Only event motifs receive non-zero
    probabilities for their own class, sampled from Beta distributions for TP/TN.
    """

    dataset = AnnotatedGenomeDataset(fna_fn, tsv_fn, window=None, num_contigs=num_contigs, random_prefix_ns=False)

    dss_motifs: Set[str] = StandardDonorDinucleotides
    if dss_motifs_mode == 'dino':
        dss_motifs = dss_motifs.union(DinoDonorDinucleotides)

    class_order = [
        # preserve index order defined in GenePredictionClass
        name for _, name in sorted(((i, n) for i, n in P.idx_to_cls.items()), key=lambda t: t[0])
    ]
    num_classes = len(class_order)

    items: List[PredictedSequence] = []

    for seq_idx in range(len(dataset)):
        seq_tokens, labels = dataset[seq_idx]
        sequence = convert_tokens_to_sequence(seq_tokens)
        L = len(sequence)
        probs = np.zeros((L, num_classes), dtype=np.float32)

        # Build motif spans using vectorized implementation on tokens
        motifs_by_class = _build_motifs_map(dss_motifs)
        spans_map = compute_event_spans_vectorized(torch.tensor(seq_tokens, dtype=torch.long).unsqueeze(0)[0], motifs_by_class)

        # For each class, sample per-event value and fill span
        for cls_idx in (int(P.START), int(P.STOP), int(P.DSS), int(P.ASS)):
            spans = spans_map.get(cls_idx, [])
            for (s, e) in spans:
                is_tp = _labels_match_span(np.asarray(labels), s, cls_idx)
                if cls_idx == int(P.START):
                    mp = start_tp if is_tp else start_tn
                elif cls_idx == int(P.STOP):
                    mp = stop_tp if is_tp else stop_tn
                elif cls_idx == int(P.DSS):
                    mp = dss_tp if is_tp else dss_tn
                else:
                    mp = ass_tp if is_tp else ass_tn
                val = _sample_normal_clipped(float(mp.mean), float(mp.std))
                val = float(np.clip(val, 1e-6, 1.0 - 1e-6))
                s0 = int(max(0, s))
                e0 = int(min(L, e))
                if e0 > s0:
                    probs[s0:e0, int(cls_idx)] = val

        items.append(PredictedSequence(
            sequence_index=seq_idx,
            sequence=sequence,
            probabilities=probs,
            class_order=class_order,
            sequence_id=getattr(dataset, 'contig_ids', [None])[seq_idx] if hasattr(dataset, 'contig_ids') else None,
        ))

    return items



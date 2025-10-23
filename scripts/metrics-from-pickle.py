#!/usr/bin/env python3

import argparse
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from gene_decoder import PredictedSequence
from utils.constants import (
    GenePredictionClass,
    StandardDonorDinucleotides,
    DinoDonorDinucleotides,
    DNAEmbed,
)
from utils.genome import AnnotatedGenomeDataset
from utils.stream import load_fasta
from utils.metrics import SequenceResult
from utils.events import build_event_motifs
from utils.metrics_report import compute_event_metrics, print_event_metrics_report


def _char_to_token_array(seq: str) -> np.ndarray:
    vocab = {'A': DNAEmbed.A, 'T': DNAEmbed.T, 'G': DNAEmbed.G, 'C': DNAEmbed.C, 'N': DNAEmbed.N}
    return np.array([vocab.get(ch.upper(), DNAEmbed.N) for ch in seq], dtype=np.int64)


def _normalize_seq(s: str) -> str:
    allowed = {'A', 'T', 'G', 'C'}
    s = str(s).upper()
    return ''.join(ch if ch in allowed else 'N' for ch in s)


def _align_probabilities_to_canonical(
    probs: np.ndarray,
    class_order: List[str],
    canonical_map: Dict[int, str],
) -> np.ndarray:
    L = int(probs.shape[0])
    C = len(canonical_map)
    out = np.zeros((L, C), dtype=np.float32)
    name_to_idx: Dict[str, int] = {v: k for k, v in canonical_map.items()}
    for j, name in enumerate(class_order):
        if name in name_to_idx:
            ci = int(name_to_idx[name])
            out[:, ci] = probs[:, j].astype(np.float32)
    return out


def _build_sequence_results(
    items: List[PredictedSequence],
    dataset: AnnotatedGenomeDataset,
) -> List[SequenceResult]:
    results: List[SequenceResult] = []
    canonical = GenePredictionClass.idx_to_cls
    for i, ps in enumerate(items):
        seq_str = ps.sequence
        # Use dataset targets/tokens to guarantee label alignment
        seq_tokens_i, targets_i = dataset[i]
        # Optional consistency check against dataset sequence string
        ds_seq_str = dataset.sequences[i]
        if _normalize_seq(ds_seq_str) != _normalize_seq(seq_str):
            raise AssertionError("Sequence mismatch between pickle and dataset at index %d" % i)

        aligned_probs = _align_probabilities_to_canonical(ps.probabilities, ps.class_order, canonical)
        # Safe argmax with NaNs: treat NaN as -inf so it never wins
        preds = np.argmax(np.nan_to_num(aligned_probs, nan=-np.inf), axis=1).astype(np.int64)

        results.append(SequenceResult(
            sequence_index=i,
            sequence_tokens=seq_tokens_i,
            targets=targets_i,
            sequence_id=(ps.sequence_id if getattr(ps, 'sequence_id', None) else None),
            predictions=preds,
            probabilities=aligned_probs,
        ))
    return results


def _sanity_check_fna_sequences(items: List[PredictedSequence], fna_fn: str) -> None:
    records = load_fasta(fna_fn)
    for idx, ps in enumerate(items):
        sid = getattr(ps, 'sequence_id', None)
        if not sid:
            continue
        if sid not in records:
            raise AssertionError(f"sequence_id '{sid}' not found in FNA records (item index {idx})")
        seq_fna = records[sid]
        if _normalize_seq(seq_fna) != _normalize_seq(ps.sequence):
            raise AssertionError(f"FNA sequence mismatch for sequence_id '{sid}' (item index {idx})")


def main():
    p = argparse.ArgumentParser(description='Compute metrics from decoder pickle of PredictedSequence items')
    p.add_argument('--pickle-path', required=True, help='Path to pickle containing List[PredictedSequence]')
    p.add_argument('--fna-fn', required=True, help='FASTA/FNA file path used for dataset reconstruction')
    p.add_argument('--tsv-fn', required=True, help='TSV annotations path (row-per-exon)')
    p.add_argument('--dss-motifs', choices=['standard', 'dino'], default='standard', help='Donor motif set for event metrics')
    p.add_argument('--num-sequences', type=int, default=0, help='Optional limit of sequences to evaluate')
    args = p.parse_args()

    with open(args.pickle_path, 'rb') as f:
        items: List[PredictedSequence] = pickle.load(f)
    if args.num_sequences:
        items = items[:args.num_sequences]

    # Sanity check against FNA using sequence_id when present
    _sanity_check_fna_sequences(items, args.fna_fn)

    # Rebuild dataset to get targets; disable random prefix to keep exact sequences
    dataset = AnnotatedGenomeDataset(
        args.fna_fn,
        args.tsv_fn,
        window=None,
        random_prefix_ns=False,
    )

    if len(dataset) < len(items):
        raise AssertionError("Dataset smaller than items in pickle; cannot align")

    # Build SequenceResult list
    results = _build_sequence_results(items, dataset)

    # Motifs selection mirrors predictor behavior
    dss = StandardDonorDinucleotides
    if args.dss_motifs == 'dino':
        dss = dss.union(DinoDonorDinucleotides)
    event_motifs_by_class = build_event_motifs(dss)

    # Compute + print using shared helpers
    metrics = compute_event_metrics(results, event_motifs_by_class, min_weight=1.0)
    print_event_metrics_report(metrics)


if __name__ == '__main__':
    main()



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
from utils.genome import AnnotatedGenomeDataset, _load_fasta
from utils.metrics import (
    SequenceResult,
    event_based_generic_metrics_factory,
    event_based_brier_factory,
    compute_event_span_mean_probability_beta_fits,
)
from utils.events import build_event_motifs


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
    records = _load_fasta(fna_fn)
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

    # Compute metrics and events (same API as predictor)
    calc_metrics, calc_metrics_with_windows = event_based_generic_metrics_factory(event_motifs_by_class)
    brier_fn = event_based_brier_factory(event_motifs_by_class)
    generic, events = calc_metrics_with_windows(results, min_weight=1.0)

    # Print Brier
    brier = brier_fn(results, event_only=True)
    print(f"Brier (overall): {brier.get('brier', 0.0):.4f}")
    brier_by_cls = brier.get('brier_by_class', {})
    if brier_by_cls:
        print("\nBrier by class:")
        for cls_idx in sorted(brier_by_cls.keys()):
            name = GenePredictionClass.idx_to_cls.get(int(cls_idx), str(int(cls_idx)))
            print(f"  {name:>10s}: {float(brier_by_cls[cls_idx]):.4f}")

    # Per-class metrics
    if generic:
        print("\nPer-class metrics:")
        for cls_idx in sorted(generic.keys()):
            name = GenePredictionClass.idx_to_cls.get(int(cls_idx), str(cls_idx))
            m = generic[cls_idx]
            print(
                f"  {name:>10s}  TP={m['tp']} FP={m['fp']} FN={m['fn']}  "
                f"Sensitivity={m['sensitivity']:.1%} Precision={m['precision']:.1%} Specificity={m['specificity']:.1%}"
            )

    # Beta fits
    print("\nEvent-only probability Beta fits (decoder span-mean, aggregated):")
    beta_fits = compute_event_span_mean_probability_beta_fits(results, event_motifs_by_class)
    classes = (GenePredictionClass.START, GenePredictionClass.STOP, GenePredictionClass.DSS, GenePredictionClass.ASS)
    for cls_idx in classes:
        cname = GenePredictionClass.idx_to_cls.get(int(cls_idx), str(int(cls_idx)))
        fits = beta_fits.get(int(cls_idx))
        if fits:
            tp = fits['tp']
            tn = fits['tn']
            print(
                f"  {cname:>5s} TP: n={int(tp['n'])} mean={tp['mean']:.4f} std={tp['std']:.4f} "
                f"beta(alpha={tp['beta_alpha']:.2f}, beta={tp['beta_beta']:.2f})"
            )
            print(
                f"  {cname:>5s} TN: n={int(tn['n'])} mean={tn['mean']:.4f} std={tn['std']:.4f} "
                f"beta(alpha={tn['beta_alpha']:.2f}, beta={tn['beta_beta']:.2f})"
            )
        else:
            print(f"  {cname:>5s} TP: n=0 mean=0.0000 std=0.0000 beta(alpha=0.00, beta=0.00)")
            print(f"  {cname:>5s} TN: n=0 mean=0.0000 std=0.0000 beta(alpha=0.00, beta=0.00)")

    if generic and beta_fits and brier_by_cls:
        import math
        print("\nSummary\ncls,sen/pre,brier,tp_m/tp_s-tn_m/tn_s,ssmd")
        for cls_idx in classes:
            cname = GenePredictionClass.idx_to_cls.get(int(cls_idx), str(int(cls_idx)))
            sen = generic[cls_idx]['sensitivity'] * 100
            pre = generic[cls_idx]['precision'] * 100
            b = brier_by_cls[cls_idx]
            tp_m = beta_fits[cls_idx]['tp']['mean'] * 100
            tp_s = beta_fits[cls_idx]['tp']['std'] * 100
            tn_m = beta_fits[cls_idx]['tn']['mean'] * 100
            tn_s = beta_fits[cls_idx]['tn']['std'] * 100
            ssmd = (tp_m - tn_m) / math.sqrt(tp_s * tp_s + tn_s * tn_s) if (tp_s > 0 or tn_s > 0) else 0.0
            print(f"{cname:>5s},{int(sen)}/{int(pre)},{b:.4f},{int(tp_m)}/{int(tp_s)}-{int(tn_m)}/{int(tn_s)},{ssmd:.2f}")


if __name__ == '__main__':
    main()



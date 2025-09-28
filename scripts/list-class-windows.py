#!/usr/bin/env python3

import argparse
from pathlib import Path
from typing import Optional, Dict, List, Iterable, Set

from utils.constants import GenePredictionClass as P
from utils.genome import AnnotatedGenomeDataset, build_class_windows
from gene_predictor.train import create_config


def build_class_weights(use_class_weights: bool,
                        start_weight: float,
                        stop_weight: float,
                        utr_weight: float,
                        dss_weight: float,
                        ass_weight: float) -> Optional[List[float]]:
    if not use_class_weights:
        return None
    weights_map = {
        P.INTERGENIC: 1.0,
        P.UTR5: float(utr_weight),
        P.START: float(start_weight),
        P.GENE: 1.0,
        P.STOP: float(stop_weight),
        P.UTR3: float(utr_weight),
        P.DSS: float(dss_weight),
        P.ASS: float(ass_weight),
    }
    return [weights_map.get(i, 1.0) for i in sorted(P.idx_to_cls.keys())]


def main():
    parser = argparse.ArgumentParser(description="List windows containing a target class after dataset balancing; writes highlighted FASTA")
    parser.add_argument('--fna-fn', type=str, required=True, help='Genome FASTA (can be .gz)')
    parser.add_argument('--tsv-fn', type=str, required=True, help='Annotations TSV (row-per-exon)')
    parser.add_argument('--class', dest='class_name', type=str, default='STOP',
                        help='Target class name (e.g., STOP, START, DSS, ASS)')
    parser.add_argument('--max-seq-length', type=int, default=1000,
                        help='Window size (must match training)')
    parser.add_argument('--num-windows', type=int, default=5000,
                        help='Number of training windows; 0 for all')
    parser.add_argument('--seed', type=int, default=17, help='Deterministic seed for selection')

    # Class-weight args to match training
    parser.add_argument('--disable-class-weights', action='store_true', help='Disable class weights')
    parser.add_argument('--start-weight', type=float, default=8.0, help='Weight for START class')
    parser.add_argument('--stop-weight', type=float, default=10.0, help='Weight for STOP class')
    parser.add_argument('--utr-weight', type=float, default=3.0, help='Weight for UTR5/UTR3')
    parser.add_argument('--dss-weight', type=float, default=8.0, help='Weight for DSS')
    parser.add_argument('--ass-weight', type=float, default=5.0, help='Weight for ASS')

    parser.add_argument('--out-fasta', type=str, default='class_windows.fasta',
                        help='Output FASTA path (ANSI GREEN highlighting in sequence)')

    args = parser.parse_args()

    # Resolve target class index
    name_to_idx = {v.upper(): k for k, v in P.idx_to_cls.items()}
    class_key = args.class_name.strip().upper()
    if class_key not in name_to_idx:
        raise ValueError(f"Unknown class '{args.class_name}'. Known: {sorted(name_to_idx.keys())}")
    target_class = int(name_to_idx[class_key])

    # Class weights mirroring training
    class_weights = build_class_weights(
        use_class_weights=not args.disable_class_weights,
        start_weight=args.start_weight,
        stop_weight=args.stop_weight,
        utr_weight=args.utr_weight,
        dss_weight=args.dss_weight,
        ass_weight=args.ass_weight,
    )

    # Build dataset (same as training)
    stride = args.max_seq_length // 2
    if args.num_windows and int(args.num_windows) > 0:
        dataset = AnnotatedGenomeDataset(
            args.fna_fn,
            args.tsv_fn,
            window=args.max_seq_length,
            stride=stride,
            num_windows=int(args.num_windows),
            class_weights=class_weights,
            random_prefix_ns=True,
        )
    else:
        dataset = AnnotatedGenomeDataset(
            args.fna_fn,
            args.tsv_fn,
            window=args.max_seq_length,
            stride=stride,
            class_weights=class_weights,
            random_prefix_ns=False,
        )

    # Use dataset-selected windows directly; if none, fall back to all
    candidate_indices = list(dataset._selected_window_indices)

    # Restrict to windows assigned to the target class (based on center-only rule)
    class_windows = build_class_windows(
        windows=[dataset.windows[i] for i in candidate_indices],
        targets=dataset.targets,
        classes_to_balance=[target_class],
        exclude_margin_bps=dataset._exclude_margin_bps,
        class_weights=class_weights,
    )
    if target_class in class_windows:
        sampled_indices_with_class: Set[int] = set(candidate_indices[j] for j in class_windows[target_class])
    else:
        sampled_indices_with_class = set()

    # ANSI GREEN highlight for the chosen class
    GREEN = "\x1b[32m"
    RESET = "\x1b[0m"

    def highlight(seq: str, tgt_slice, highlight_cls: int) -> str:
        chars: List[str] = []
        for i, ch in enumerate(seq):
            if int(tgt_slice[i]) == int(highlight_cls):
                chars.append(f"{GREEN}{ch.upper()}{RESET}")
            else:
                chars.append(ch.lower())
        return ''.join(chars)

    # Write FASTA with highlighted sequences for sampler-selected windows
    out_path = Path(args.out_fasta)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w') as f:
        for w_idx in sorted(sampled_indices_with_class):
            contig_idx, s, e = dataset.windows[w_idx]
            contig_id = dataset.contig_ids[contig_idx]
            seq = dataset.sequences[contig_idx][s:e]
            tgt_slice = dataset.targets[contig_idx][s:e]
            colored = highlight(seq, tgt_slice, target_class)
            header = f">{contig_id} {s}:{e} win={w_idx} class={args.class_name.upper()}"
            f.write(header + "\n")
            f.write(colored + "\n")

    print(f"Wrote {out_path}")


if __name__ == '__main__':
    main()



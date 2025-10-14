#!/usr/bin/env python3

import argparse
import pickle
from typing import Tuple

from gene_decoder.synthetic import (
    build_synthetic_decoder_inputs,
    MeanStdParams,
)


def _parse_mean_std(text: str) -> MeanStdParams:
    parts = text.split(',')
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Expected 'mean,std' pair")
    try:
        m = float(parts[0])
        s = float(parts[1])
    except ValueError as e:
        raise argparse.ArgumentTypeError("Mean and std must be numbers") from e
    if not (0.0 <= m <= 1.0):
        raise argparse.ArgumentTypeError("Mean must be in [0,1]")
    if s < 0.0:
        raise argparse.ArgumentTypeError("Std must be non-negative")
    return MeanStdParams(mean=m, std=s)


def main():
    p = argparse.ArgumentParser(description='Synthesize decoder input pickle with event-only probabilities from Beta distributions')
    p.add_argument('--fna-fn', required=True, help='Genome sequence FASTA (can be .gz)')
    p.add_argument('--tsv-fn', required=True, help='Annotations TSV (training format)')
    p.add_argument('--output-pkl', required=True, help='Output pickle path (List[PredictedSequence])')
    p.add_argument('--dss-motifs', required=True, choices=['standard', 'dino'], help='DSS motif set: standard or dino')
    p.add_argument('--num-contigs', type=int, default=0, help='Limit number of contigs (0 = all)')

    # Per-class distribution params (mean,std)
    p.add_argument('--start-tp', type=_parse_mean_std, required=True, help='START TP as mean,std')
    p.add_argument('--start-tn', type=_parse_mean_std, required=True, help='START TN as mean,std')
    p.add_argument('--stop-tp', type=_parse_mean_std, required=True, help='STOP TP as mean,std')
    p.add_argument('--stop-tn', type=_parse_mean_std, required=True, help='STOP TN as mean,std')
    p.add_argument('--dss-tp', type=_parse_mean_std, required=True, help='DSS TP as mean,std')
    p.add_argument('--dss-tn', type=_parse_mean_std, required=True, help='DSS TN as mean,std')
    p.add_argument('--ass-tp', type=_parse_mean_std, required=True, help='ASS TP as mean,std')
    p.add_argument('--ass-tn', type=_parse_mean_std, required=True, help='ASS TN as mean,std')

    args = p.parse_args()

    items = build_synthetic_decoder_inputs(
        fna_fn=args.fna_fn,
        tsv_fn=args.tsv_fn,
        start_tp=args.start_tp,
        start_tn=args.start_tn,
        stop_tp=args.stop_tp,
        stop_tn=args.stop_tn,
        dss_tp=args.dss_tp,
        dss_tn=args.dss_tn,
        ass_tp=args.ass_tp,
        ass_tn=args.ass_tn,
        dss_motifs_mode=args.dss_motifs,
        num_contigs=int(args.num_contigs) if args.num_contigs else 0,
    )

    with open(args.output_pkl, 'wb') as f:
        pickle.dump(items, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"✓ Wrote {args.output_pkl}")


if __name__ == '__main__':
    main()



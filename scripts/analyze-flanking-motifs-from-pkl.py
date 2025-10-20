#!/usr/bin/env python3

import argparse
import pickle

from gene_decoder.flanking_stats import (
    compute_flanking_prob_stats_from_items,
    format_prob_stats_as_csv,
)


def main() -> None:
    p = argparse.ArgumentParser(description="Analyze flanking motif probabilities from decoder input pickle and print CSV stats")
    p.add_argument("--input-pkl", required=True, help="Pickle file containing List[PredictedSequence]")
    p.add_argument("--flank", type=int, default=3, help="Flank length (bp) on each side (>=0)")
    p.add_argument("--site", default="both", choices=["ASS", "DSS", "both"], help="Which site(s) to analyze")
    p.add_argument("--dss-motifs-mode", default="standard", choices=["standard", "dino"], help="DSS motifs set")
    p.add_argument("--num-sequences", type=int, default=0, help="Limit to first N sequences (0 = all)")

    args = p.parse_args()

    with open(args.input_pkl, 'rb') as f:
        items = pickle.load(f)

    dss_stats, ass_stats = compute_flanking_prob_stats_from_items(
        items,
        flank=int(args.flank),
        site=args.site,  # type: ignore[arg-type]
        dss_motifs_mode=args.dss_motifs_mode,
        num_sequences=int(args.num_sequences),
    )

    printed_header = False
    if args.site in ("DSS", "both"):
        for line in format_prob_stats_as_csv("DSS", dss_stats, include_header=not printed_header):
            print(line)
        printed_header = True
    if args.site in ("ASS", "both"):
        for line in format_prob_stats_as_csv("ASS", ass_stats, include_header=not printed_header):
            print(line)


if __name__ == "__main__":
    main()

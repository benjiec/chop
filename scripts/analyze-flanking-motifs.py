#!/usr/bin/env python3

import argparse
from typing import Literal

from gene_decoder.flanking_stats import compute_flanking_motif_stats, format_counts_as_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze flanking motifs around DSS/ASS and print TSV stats")
    parser.add_argument("--fna", required=True, help="Input FASTA/FA file (.fna[.gz])")
    parser.add_argument("--tsv", required=True, help="Annotations TSV path")
    parser.add_argument("--flank", type=int, default=3, help="Flank length (bp) on each side (>=0)")
    parser.add_argument("--site", default="both", choices=["ASS", "DSS", "both"], help="Which site(s) to analyze")
    parser.add_argument("--dss-motifs-mode", default="standard", choices=["standard", "dino"], help="DSS motifs set")
    parser.add_argument("--num-contigs", type=int, default=0, help="Limit to first N contigs (0 = all)")

    args = parser.parse_args()

    dss_counts, ass_counts = compute_flanking_motif_stats(
        fna_fn=args.fna,
        tsv_fn=args.tsv,
        flank=int(args.flank),
        site=args.site,  # type: ignore[arg-type]
        dss_motifs_mode=args.dss_motifs_mode,
        num_contigs=int(args.num_contigs),
    )

    printed_header = False
    if args.site in ("DSS", "both"):
        for line in format_counts_as_csv("DSS", dss_counts, include_header=not printed_header):
            print(line)
        printed_header = True
    if args.site in ("ASS", "both"):
        for line in format_counts_as_csv("ASS", ass_counts, include_header=not printed_header):
            print(line)


if __name__ == "__main__":
    main()



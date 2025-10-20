#!/usr/bin/env python3

import argparse
import pickle

from gene_decoder.augment_from_flanks import augment_items_from_flanks


def main() -> None:
    p = argparse.ArgumentParser(description="Augment/override DSS/ASS probabilities in a decoder input pickle using flanking motif stats CSV")
    p.add_argument("--input-pkl", required=True, help="Input decoder pickle (List[PredictedSequence])")
    p.add_argument("--flank-csv", required=True, help="CSV from analyze-flanking-motifs.py")
    p.add_argument("--output-pkl", required=True, help="Output decoder pickle path")
    p.add_argument("--flank", type=int, required=True, help="Flank length used to build motifs in CSV")
    p.add_argument("--dss-motifs-mode", default="standard", choices=["standard", "dino"], help="DSS motif selection")
    p.add_argument("--mode", default="override", choices=["override", "augment"], help="How to combine stats with predictions")

    args = p.parse_args()

    with open(args.input_pkl, 'rb') as f:
        items = pickle.load(f)

    out_items = augment_items_from_flanks(
        items,
        flank_counts_csv=args.flank_csv,
        flank=int(args.flank),
        dss_motifs_mode=args.dss_motifs_mode,
        mode=args.mode,
    )

    with open(args.output_pkl, 'wb') as f:
        pickle.dump(out_items, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()

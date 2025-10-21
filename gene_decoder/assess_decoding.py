#!/usr/bin/env python3

import argparse

from gene_decoder.evaluate_decoding import evaluate_decoding


def main():
    p = argparse.ArgumentParser(description='Assess decoder predictions against expected annotations.')
    p.add_argument('--decoded-tsv', required=True, help='Decoder output TSV (row-per-exon).')
    p.add_argument('--expected-tsv', required=True, help='Expected annotations TSV (row-per-exon).')
    p.add_argument('--topk-boundaries', type=int, required=True, help='Top-K gene boundaries per sequence_id to include.')
    p.add_argument('--top-gene', action='store_true', help='If set, only return one top gene.')
    p.add_argument('--per-sequence', action='store_true', help='If set, compute per-sequence stats internally (not printed).')
    args = p.parse_args()

    metrics = evaluate_decoding(
        decoded_tsv=args.decoded_tsv,
        expected_tsv=args.expected_tsv,
        topk_boundaries=args.topk_boundaries,
        top_gene_only=bool(args.top_gene),
        per_sequence=bool(args.per_sequence),
    )

    ex = metrics['exon']
    ge = metrics['gene']
    st = metrics['start']
    # Print two lines with labeled counts and metrics; sensitivity/precision to 4 decimals
    print(f"exon TP={ex['tp']} FP={ex['fp']} FN={ex['fn']} Sensitivity={ex['sensitivity']:.4f} Precision={ex['precision']:.4f}")
    print(f"gene TP={ge['tp']} FP={ge['fp']} FN={ge['fn']} Sensitivity={ge['sensitivity']:.4f} Precision={ge['precision']:.4f}")
    print(f"start TP={st['tp']} FP={st['fp']} FN={st['fn']} Sensitivity={st['sensitivity']:.4f} Precision={st['precision']:.4f}")


if __name__ == '__main__':
    main()



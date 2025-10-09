#!/usr/bin/env python3

import argparse
import sys

from utils.merge_split import (
    merge_and_split,
    write_fasta,
    write_tsv,
)


def main():
    parser = argparse.ArgumentParser(
        description="Merge multiple TSV/FNA inputs and split into train/test by sequence_id"
    )
    parser.add_argument('--tsv', nargs='+', required=True, help='Input TSV files (row-per-exon)')
    parser.add_argument('--fna', nargs='+', required=True, help='Input FNA/FASTA files (can be .gz)')
    parser.add_argument('--num-train', type=int, required=True, help='Number of unique sequences for train split')
    parser.add_argument('--num-test', type=int, required=True, help='Number of unique sequences for test split')
    parser.add_argument('--output-train-tsv', required=True, help='Output train TSV path')
    parser.add_argument('--output-train-fna', required=True, help='Output train FNA path')
    parser.add_argument('--output-test-tsv', required=True, help='Output test TSV path')
    parser.add_argument('--output-test-fna', required=True, help='Output test FNA path')

    args = parser.parse_args()

    try:
        sequences, header, train_rows, test_rows, valid_sids = merge_and_split(
            tsv_inputs=args.tsv,
            fasta_inputs=args.fna,
            num_train=int(args.num_train),
            num_test=int(args.num_test),
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Extract selected ids from row lists to write FASTA subsets
    sid_idx = None
    for i, h in enumerate(header):
        if h.lower() == 'sequence_id':
            sid_idx = i
            break
    if sid_idx is None:
        print("Error: header missing sequence_id column", file=sys.stderr)
        sys.exit(1)

    train_ids = []
    seen = set()
    for r in train_rows:
        sid = r[sid_idx]
        if sid not in seen:
            train_ids.append(sid)
            seen.add(sid)

    test_ids = []
    seen = set()
    for r in test_rows:
        sid = r[sid_idx]
        if sid not in seen:
            test_ids.append(sid)
            seen.add(sid)

    # Write outputs
    write_tsv(header, train_rows, args.output_train_tsv)
    write_tsv(header, test_rows, args.output_test_tsv)
    write_fasta(sequences, train_ids, args.output_train_fna)
    write_fasta(sequences, test_ids, args.output_test_fna)

    print(f"Merged sequences available: {len(valid_sids)}")
    print(f"Wrote train: {len(train_ids)} sequences")
    print(f"Wrote test: {len(test_ids)} sequences")


if __name__ == "__main__":
    main()




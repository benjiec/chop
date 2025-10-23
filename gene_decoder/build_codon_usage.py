#!/usr/bin/env python3

import argparse
from typing import Iterator
from utils.genome import _parse_tsv_annotations, GeneAnnotation
from utils.stream import load_fasta
from gene_decoder.codon_usage import build_codon_usage_from_cds


def _iter_cds(records: dict, anns: Iterator[GeneAnnotation]):
    for ann in anns:
        seq = records.get(ann.sequence_id)
        if not seq:
            continue
        # Concatenate exon segments in genomic order
        exons = sorted(ann.exons, key=lambda t: t[0])
        cds = ''.join(seq[s:e] for (s, e) in exons)
        yield cds


def main():
    p = argparse.ArgumentParser(description="Build codon usage JSON from training TSV+FASTA")
    p.add_argument('--fna', required=True, help='FASTA of genomic sequences (can be .gz)')
    p.add_argument('--tsv', required=True, help='Training annotations TSV (row-per-exon)')
    p.add_argument('--alpha', type=float, default=1.0, help='Additive smoothing')
    p.add_argument('--output', required=True, help='Output JSON path for codon->prob')
    args = p.parse_args()

    records = load_fasta(args.fna)
    anns = _parse_tsv_annotations(args.tsv)
    codon_model = build_codon_usage_from_cds(_iter_cds(records, anns), alpha=args.alpha)
    codon_model.to_json(args.output)
    print(f"✓ Codon usage written to {args.output}")


if __name__ == '__main__':
    main()



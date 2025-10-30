#!/usr/bin/env python3

import argparse
import csv
from pathlib import Path

import numpy as np

from utils.genome import AnnotatedGenomeDataset
from utils.stream import NumericalStream, load_fasta
from utils.constants import (
    GenePredictionClass as P,
    ConventionalStopCodons,
    StandardDonorDinucleotides,
    DinoDonorDinucleotides,
    ConventionalAcceptorDinucleotides,
)


def main():
    p = argparse.ArgumentParser(description="Export per-motif aux stream values with positives/negatives")
    p.add_argument('--fna', required=True, help='FASTA/FNA file with sequences')
    p.add_argument('--tsv', required=True, help='Annotations TSV (row-per-exon)')
    p.add_argument('--stream', required=True, help='Path to aux stream pickle (.pkl or .pkl.gz)')
    p.add_argument('--dss-motifs', choices=['standard', 'dino'], default='standard', help='Donor set for DSS scanning')
    p.add_argument('--output-csv', required=True, help='Where to write the CSV table')
    args = p.parse_args()

    # Load data
    ds = AnnotatedGenomeDataset(
        fasta_path=args.fna,
        annotations_tsv_path=args.tsv,
        random_prefix_ns=False,
        window=None,
        stride=None,
        aux_stream_path=None,
        aux_normalize=False,
    )
    ns = NumericalStream(args.stream)
    fasta_records = load_fasta(args.fna)

    if ns.channels is None:
        raise ValueError("Stream has no channels metadata")
    channel_names = list(ns.channels)

    # Motif sets
    stop_set = ConventionalStopCodons
    donor_set = StandardDonorDinucleotides if args.dss_motifs == 'standard' else DinoDonorDinucleotides
    acceptor_set = ConventionalAcceptorDinucleotides

    header = ['sequence_id', 'pos', 'motif_type', 'is_positive'] + channel_names
    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(header)

        # Build contig -> target map from dataset, but scan sequences from FASTA directly
        contig_to_tgt = {sid: ds.targets[idx] for idx, sid in enumerate(ds.contig_ids)}
        for sid, seq in fasta_records.items():
            if sid not in contig_to_tgt:
                continue
            tgt = contig_to_tgt[sid]
            L = len(seq)
            # Aux stream for this contig
            arr = ns.get(sid)
            if int(arr.shape[0]) != L:
                raise ValueError(f"Aux length mismatch for {sid}: {arr.shape[0]} vs {L}")

            # START (ATG)
            i = seq.find('ATG', 0)
            while i != -1 and i + 2 < L:
                span = slice(i, i+3)
                is_pos = int(np.all(tgt[span] == P.START))
                vals = np.mean(arr[span, :], axis=0)
                w.writerow([sid, i, 'START', is_pos] + [f"{float(v):.6f}" for v in vals])
                i = seq.find('ATG', i + 1)

            # STOP (TAA/TAG/TGA)
            for stop in stop_set:
                i = seq.find(stop, 0)
                while i != -1 and i + 2 < L:
                    span = slice(i, i+3)
                    is_pos = int(np.all(tgt[span] == P.STOP))
                    vals = np.mean(arr[span, :], axis=0)
                    w.writerow([sid, i, 'STOP', is_pos] + [f"{float(v):.6f}" for v in vals])
                    i = seq.find(stop, i + 1)

            # DSS (donor)
            for don in donor_set:
                i = seq.find(don, 0)
                while i != -1 and i + 1 < L:
                    span = slice(i, i+2)
                    is_pos = int(np.all(tgt[span] == P.DSS))
                    vals = np.mean(arr[span, :], axis=0)
                    w.writerow([sid, i, 'DSS', is_pos] + [f"{float(v):.6f}" for v in vals])
                    i = seq.find(don, i + 1)

            # ASS (acceptor)
            for acc in acceptor_set:
                i = seq.find(acc, 0)
                while i != -1 and i + 1 < L:
                    span = slice(i, i+2)
                    is_pos = int(np.all(tgt[span] == P.ASS))
                    vals = np.mean(arr[span, :], axis=0)
                    w.writerow([sid, i, 'ASS', is_pos] + [f"{float(v):.6f}" for v in vals])
                    i = seq.find(acc, i + 1)

    print(f"✓ Wrote motif table to: {out_path}")


if __name__ == '__main__':
    main()



#!/usr/bin/env python3

import argparse
from pathlib import Path
import numpy as np

from utils.constants import GenePredictionClass as P
from utils.stream import NumericalStream, load_fasta
from utils.genome import AnnotatedGenomeDataset


def main():
    p = argparse.ArgumentParser(description="Add target class index channel to a NumericalStream from TSV annotations")
    p.add_argument('--stream', required=True, help='Path to stream pickle file; created if missing')
    p.add_argument('--fna', required=True, help='FASTA/FNA file with sequences')
    p.add_argument('--tsv', required=True, help='Annotations TSV (row-per-exon)')
    args = p.parse_args()

    stream_path = Path(args.stream)
    if stream_path.exists():
        ns = NumericalStream(str(stream_path))
    else:
        ns = NumericalStream.create_empty(str(stream_path))

    # Load FASTA records and dataset (no random prefix; no windowing)
    records = load_fasta(args.fna)
    ds = AnnotatedGenomeDataset(
        fasta_path=args.fna,
        annotations_tsv_path=args.tsv,
        random_prefix_ns=False,
        window=None,
        stride=None,
        aux_stream_path=None,
        aux_normalize=False,
    )

    # Build per-contig target array (union across any multiple annotations per contig)
    # Initialize to INTERGENIC for all contigs in FASTA (so unannotated contigs become all-intergenic)
    contig_to_target = {sid: np.full(len(seq), P.INTERGENIC, dtype=np.int64) for sid, seq in records.items()}
    for sid, tgt in zip(ds.contig_ids, ds.targets):
        base = contig_to_target.get(sid)
        if base is None:
            # Annotation for a contig not in FASTA; dataset already validates, but be safe
            raise ValueError(f"Annotation contig not found in FASTA: {sid}")
        if int(base.shape[0]) != int(tgt.shape[0]):
            raise ValueError(f"Length mismatch for {sid}: fasta={int(base.shape[0])} tsv_targets={int(tgt.shape[0])}")
        # Overlay: fill only positions that are currently INTERGENIC
        mask = (base == P.INTERGENIC) & (tgt != P.INTERGENIC)
        base[mask] = tgt[mask]

    # Map sequence string -> target array; require identical targets for duplicate sequences
    # This relies on add_channel iterating over the same FASTA sequences
    seq_to_target: dict[str, np.ndarray] = {}
    for sid, seq in records.items():
        arr = contig_to_target[sid].astype(np.float32)
        if seq in seq_to_target:
            # Enforce identical targets across duplicated sequence strings
            if not np.array_equal(seq_to_target[seq].astype(np.int64), arr.astype(np.int64)):
                print(f"Duplicate sequence with different targets for IDs including {sid}, skip")
            # else: identical, keep existing
        else:
            seq_to_target[seq] = arr

    def gen(seq: str) -> np.ndarray:
        try:
            return seq_to_target[seq]
        except KeyError:
            raise ValueError("Sequence from FASTA not found in precomputed mapping; ensure same --fna is used")

    ns.add_channel(args.fna, "target", gen)
    ns.save()
    print(f"✓ Added channel 'target' and saved to {stream_path}")


if __name__ == '__main__':
    main()



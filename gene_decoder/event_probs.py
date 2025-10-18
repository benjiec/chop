#!/usr/bin/env python3

import argparse
import pickle
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np

from gene_decoder import PredictedSequence
from gene_decoder.decoder import _scan_events, _event_prob
from gene_decoder.evaluate_decoding import _parse_expected
from utils.constants import (
    GenePredictionClass as P,
    StandardDonorDinucleotides,
    DinoDonorDinucleotides,
)


def _expected_event_positions_from_exons(
    exons: List[Tuple[int, int]]
) -> Dict[str, Set[int]]:
    """
    Compute expected event positions (0-based) from 0-based half-open exon coordinates.

    Conventions aligned with decoder:
    - START at first exon start index
    - STOP at last exon end - 3
    - DSS at each exon end (except last exon): position == exon_end
    - ASS at each next exon start - 2
    """
    expected: Dict[str, Set[int]] = {"start": set(), "stop": set(), "dss": set(), "ass": set()}
    if not exons:
        return expected

    # START and STOP
    first_s0 = int(exons[0][0])
    last_e0 = int(exons[-1][1])
    if last_e0 - 3 >= 0:
        expected["stop"].add(last_e0 - 3)
    expected["start"].add(first_s0)

    # Splice sites
    for i in range(len(exons) - 1):
        prev_s0, prev_e0 = exons[i]
        next_s0, _ = exons[i + 1]
        expected["dss"].add(int(prev_e0))          # donor at intron start
        expected["ass"].add(int(next_s0) - 2)      # acceptor di-nucleotide starts 2bp before exon start

    return expected


def _collect_expected_by_sequence(expected_tsv: str) -> Dict[str, Dict[str, Set[int]]]:
    """
    Build mapping: sequence_id -> { 'start'|'stop'|'dss'|'ass' : set(positions0) }
    Only '+' strand annotations are considered; coordinates are already 0-based half-open.
    """
    parsed = _parse_expected(expected_tsv)
    out: Dict[str, Dict[str, Set[int]]] = {}
    for sid, genes in parsed.items():
        by_type: Dict[str, Set[int]] = {"start": set(), "stop": set(), "dss": set(), "ass": set()}
        for strand, exons in genes:
            if strand != '+':
                # Decoder scanning is forward-only in this implementation
                continue
            ev = _expected_event_positions_from_exons(list(exons))
            for k in by_type.keys():
                by_type[k].update(ev[k])
        out[sid] = by_type
    return out


def _write_header(f):
    header = ["sequence_id", "class", "type", "pos", "prob"]
    f.write('\t'.join(header) + '\n')


def _class_key_to_name(k: str) -> str:
    if k == 'start':
        return 'START'
    if k == 'stop':
        return 'STOP'
    if k == 'dss':
        return 'DSS'
    if k == 'ass':
        return 'ASS'
    return k.upper()


def _class_key_to_idx(k: str) -> int:
    if k == 'start':
        return int(P.START)
    if k == 'stop':
        return int(P.STOP)
    if k == 'dss':
        return int(P.DSS)
    if k == 'ass':
        return int(P.ASS)
    raise KeyError(k)


def compute_event_rows(
    items: List[PredictedSequence],
    expected_tsv: str,
    dss_motifs_mode: str,
) -> List[Tuple[str, str, str, int, float]]:
    """Compute rows for TSV: (sequence_id, class, type, pos1, prob)."""
    assert dss_motifs_mode in ('standard', 'dino')
    dss_motifs = StandardDonorDinucleotides
    if dss_motifs_mode == 'dino':
        dss_motifs = dss_motifs.union(DinoDonorDinucleotides)

    expected_by_sid = _collect_expected_by_sequence(expected_tsv)

    rows: List[Tuple[str, str, str, int, float]] = []
    for ps in items:
        sid = ps.sequence_id if getattr(ps, 'sequence_id', None) else str(ps.sequence_index)
        seq = ps.sequence
        probs = ps.probabilities

        # Expected positive positions for this sequence
        exp = expected_by_sid.get(sid, {"start": set(), "stop": set(), "dss": set(), "ass": set()})

        # Enumerate all motif positions using decoder's scanner (no filtering by prob thresholds)
        ev_positions = _scan_events(seq, list(dss_motifs), probs=None,
                                    min_logp_start=None, min_logp_stop=None,
                                    min_logp_dss=None, min_logp_ass=None)

        for key in ("start", "stop", "dss", "ass"):
            cls_name = _class_key_to_name(key)
            cls_idx = _class_key_to_idx(key)
            pos_list = list(ev_positions.get(key, []))

            # All positives: those in expected set
            exp_set = exp.get(key, set())

            for pos0 in pos_list:
                pos1 = int(pos0) + 1
                is_positive = pos0 in exp_set
                typ = 'positive' if is_positive else 'negative'
                prob = float(_event_prob(probs, int(pos0), cls_idx, negative=False))
                rows.append((sid, cls_name, typ, pos1, prob))

    return rows


def main():
    p = argparse.ArgumentParser(description='Compute event probabilities at motif positions and label as positive/negative by expected annotations.')
    p.add_argument('--input-pkl', required=True, help='Pickle file containing List[PredictedSequence]')
    p.add_argument('--expected-tsv', required=True, help='Expected annotations TSV (row-per-exon).')
    p.add_argument('--dss-motifs', required=True, choices=['standard', 'dino'], help='Donor motif mode.')
    p.add_argument('--num-sequences', type=int, default=0)
    p.add_argument('--output-tsv', required=True)
    args = p.parse_args()

    with open(args.input_pkl, 'rb') as f:
        items: List[PredictedSequence] = pickle.load(f)
    if args.num_sequences:
        items = items[:args.num_sequences]

    rows = compute_event_rows(items, args.expected_tsv, args.dss_motifs)

    out_path = Path(args.output_tsv)
    with open(out_path, 'w') as f:
        _write_header(f)
        for sid, cls_name, typ, pos1, prob in rows:
            f.write(f"{sid}\t{cls_name}\t{typ}\t{pos1}\t{prob:.6f}\n")

    print(f"\u2713 Wrote {out_path}")


if __name__ == '__main__':
    main()



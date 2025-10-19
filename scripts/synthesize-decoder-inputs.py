#!/usr/bin/env python3

import argparse
import pickle
from typing import Tuple, Dict, List, Optional

from gene_decoder.synthetic import (
    build_synthetic_decoder_inputs,
    MeanStdParams,
)
from utils.constants import (
    GenePredictionClass as P,
    ConventionalStopCodons,
    StandardDonorDinucleotides,
    DinoDonorDinucleotides,
    ConventionalAcceptorDinucleotides,
)
from gene_decoder import PredictedSequence
from utils.events import compute_event_spans_vectorized
import numpy as np
import torch


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
    p = argparse.ArgumentParser(description='Synthesize or modify decoder input pickle with event-only probabilities from Normal(mean,std) distributions')
    p.add_argument('--fna-fn', required=False, help='Genome sequence FASTA (can be .gz). Required when --input-pkl is not provided')
    p.add_argument('--tsv-fn', required=True, help='Annotations TSV (training format)')
    p.add_argument('--input-pkl', required=False, help='Optional input pickle (List[PredictedSequence]) to modify in-place for specified classes')
    p.add_argument('--output-pkl', required=True, help='Output pickle path (List[PredictedSequence])')
    p.add_argument('--dss-motifs', required=True, choices=['standard', 'dino'], help='DSS motif set: standard or dino')
    p.add_argument('--num-contigs', type=int, default=0, help='Limit number of contigs (0 = all)')

    # Per-class distribution params (mean,std). Optional individually; validated post-parse.
    p.add_argument('--start-tp', type=_parse_mean_std, required=False, help='START TP as mean,std')
    p.add_argument('--start-tn', type=_parse_mean_std, required=False, help='START TN as mean,std')
    p.add_argument('--stop-tp', type=_parse_mean_std, required=False, help='STOP TP as mean,std')
    p.add_argument('--stop-tn', type=_parse_mean_std, required=False, help='STOP TN as mean,std')
    p.add_argument('--dss-tp', type=_parse_mean_std, required=False, help='DSS TP as mean,std')
    p.add_argument('--dss-tn', type=_parse_mean_std, required=False, help='DSS TN as mean,std')
    p.add_argument('--ass-tp', type=_parse_mean_std, required=False, help='ASS TP as mean,std')
    p.add_argument('--ass-tn', type=_parse_mean_std, required=False, help='ASS TN as mean,std')

    args = p.parse_args()

    # Validate required combinations based on mode
    if args.input_pkl:
        # At least one class pair must be provided
        provided_pairs = 0
        provided_pairs += int(bool(args.start_tp and args.start_tn))
        provided_pairs += int(bool(args.stop_tp and args.stop_tn))
        provided_pairs += int(bool(args.dss_tp and args.dss_tn))
        provided_pairs += int(bool(args.ass_tp and args.ass_tn))
        if provided_pairs == 0:
            raise SystemExit("When using --input-pkl, provide at least one class pair like --start-tp/--start-tn or --stop-tp/--stop-tn, etc.")
    else:
        # Building from scratch requires fna and all eight params
        if not args.fna_fn:
            raise SystemExit("--fna-fn is required when --input-pkl is not provided")
        missing = []
        for name in ['start_tp','start_tn','stop_tp','stop_tn','dss_tp','dss_tn','ass_tp','ass_tn']:
            if getattr(args, name) is None:
                missing.append('--' + name.replace('_','-'))
        if missing:
            raise SystemExit("Missing required arguments for generation: " + ' '.join(missing))

    def _build_motifs_map(mode: str) -> Dict[int, set]:
        dss = StandardDonorDinucleotides
        if mode == 'dino':
            dss = dss.union(DinoDonorDinucleotides)
        return {
            int(P.START): {'ATG'},
            int(P.STOP): set(m.upper() for m in ConventionalStopCodons),
            int(P.DSS): set(m.upper() for m in dss),
            int(P.ASS): set(m.upper() for m in ConventionalAcceptorDinucleotides),
        }

    def _parse_tsv_annotations(tsv_path: str) -> Dict[str, Dict[str, List[Tuple[int, int]]]]:
        # Returns: seq_id -> { 'exons': [(xs, xe), ...] } grouped by gene, but we only need events per seq
        # We will collect per-sequence per-gene exons, then derive event spans.
        import csv
        by_seq_gene: Dict[Tuple[str, str], List[Tuple[int, int]]] = {}
        with open(tsv_path, 'r') as f:
            reader = csv.DictReader(f, delimiter='\t')
            use_header = reader.fieldnames is not None and 'sequence_id' in reader.fieldnames
            if use_header:
                for row in reader:
                    sid = row['sequence_id']
                    gid = row['gene_id']
                    xs = int(row['exon_start']) - 1  # to 0-based
                    xe = int(row['exon_end'])        # already exclusive in code that wrote it
                    by_seq_gene.setdefault((sid, gid), []).append((xs, xe))
            else:
                f.seek(0)
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split('\t')
                    # Assume training TSV ordering: sequence_id, gene_id, gene_start, gene_end, exon_start, exon_end, ...
                    sid = parts[0]
                    gid = parts[1]
                    xs = int(parts[4]) - 1
                    xe = int(parts[5])
                    by_seq_gene.setdefault((sid, gid), []).append((xs, xe))

        # Derive per-seq event spans dict[class] -> list[(s,e)]
        events_by_seq: Dict[str, Dict[int, List[Tuple[int, int]]]] = {}
        from collections import defaultdict
        for (sid, gid), exons in by_seq_gene.items():
            exons_sorted = sorted(exons, key=lambda t: t[0])
            if not exons_sorted:
                continue
            # START span: first exon start [xs, xs+3)
            xs0, xe0 = exons_sorted[0]
            ev = events_by_seq.setdefault(sid, defaultdict(list))
            ev[int(P.START)].append((xs0, xs0 + 3))
            # STOP span: last exon end -3 to end
            xsn, xen = exons_sorted[-1]
            ev[int(P.STOP)].append((xen - 3, xen))
            # Splice sites for multi-exon genes
            for i in range(len(exons_sorted) - 1):
                xs_i, xe_i = exons_sorted[i]
                xs_j, xe_j = exons_sorted[i + 1]
                # DSS starts at exon end (inclusive end in TSV); motif span [xe_i, xe_i+2)
                ev[int(P.DSS)].append((xe_i, xe_i + 2))
                # ASS ends at next exon start; motif span [xs_j-2, xs_j)
                ev[int(P.ASS)].append((xs_j - 2, xs_j))
        return events_by_seq

    def _encode_tokens(seq: str) -> torch.Tensor:
        mp = {'A': 0, 'T': 1, 'G': 2, 'C': 3}
        arr = [mp.get(ch.upper(), 4) for ch in seq]
        return torch.tensor(arr, dtype=torch.long)

    if args.input_pkl:
        # Load existing items and selectively override classes provided on CLI
        with open(args.input_pkl, 'rb') as f:
            items = pickle.load(f)
        # Build motif map and parse TSV to derive expected event spans per sequence_id
        motifs = _build_motifs_map(args.dss_motifs)
        events_from_tsv = _parse_tsv_annotations(args.tsv_fn)

        # Determine which classes to override based on provided args
        override_classes: Dict[int, Tuple[MeanStdParams, MeanStdParams]] = {}
        if args.start_tp and args.start_tn:
            override_classes[int(P.START)] = (args.start_tp, args.start_tn)
        if args.stop_tp and args.stop_tn:
            override_classes[int(P.STOP)] = (args.stop_tp, args.stop_tn)
        if args.dss_tp and args.dss_tn:
            override_classes[int(P.DSS)] = (args.dss_tp, args.dss_tn)
        if args.ass_tp and args.ass_tn:
            override_classes[int(P.ASS)] = (args.ass_tp, args.ass_tn)

        out_items: List[PredictedSequence] = []
        for ps in items:
            seq_id = getattr(ps, 'sequence_id', None)
            seq_str = ps.sequence
            L = len(seq_str)
            probs = ps.probabilities.copy()
            # Build labels array of length L for events (only classes we care about)
            labels = np.zeros(L, dtype=int)
            if seq_id and seq_id in events_from_tsv:
                ev_map = events_from_tsv[seq_id]
                for cls_id, spans in ev_map.items():
                    for (s, e) in spans:
                        s0 = max(0, int(s))
                        e0 = min(L, int(e))
                        if e0 > s0:
                            labels[s0:e0] = int(cls_id)

            # Compute motif spans in this sequence
            tokens_t = _encode_tokens(seq_str)
            spans_map = compute_event_spans_vectorized(tokens_t, motifs)

            # Map class names to column indices using ps.class_order when available; fallback to constants
            name_to_col: Dict[str, int] = {}
            if getattr(ps, 'class_order', None):
                for idx, name in enumerate(ps.class_order):
                    name_to_col[str(name).upper()] = int(idx)
            else:
                for idx, name in P.idx_to_cls.items():
                    name_to_col[str(name).upper()] = int(idx)

            for override_cls, (tp_ms, tn_ms) in override_classes.items():
                spans = spans_map.get(int(override_cls), [])
                col = name_to_col.get(P.idx_to_cls[int(override_cls)].upper(), int(override_cls))
                for (s, e) in spans:
                    s0 = max(0, int(s))
                    e0 = min(L, int(e))
                    if e0 <= s0:
                        continue
                    is_pos = bool(np.any(labels[s0:e0].astype(int) == int(override_cls)))
                    ms = tp_ms if is_pos else tn_ms
                    val = float(np.clip(np.random.normal(loc=float(ms.mean), scale=float(ms.std)), 1e-6, 1.0 - 1e-6))
                    probs[s0:e0, col] = val

            out_items.append(PredictedSequence(
                sequence_index=ps.sequence_index,
                sequence=seq_str,
                probabilities=probs,
                class_order=ps.class_order,
                sequence_id=seq_id,
            ))
        items = out_items
    else:
        if not args.fna_fn:
            raise SystemExit("--fna-fn is required when --input-pkl is not provided")
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



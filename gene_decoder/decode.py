#!/usr/bin/env python3

import argparse
import json
import pickle
from pathlib import Path
from typing import List
from gene_decoder.types import PredictedSequence, DecodedResult, CandidateGene
from gene_decoder.decoder import decode_sequence
from gene_decoder.codon_usage import CodonUsageModel
from utils.constants import GenePredictionClass as P
import math


def _cds_from_exons(seq: str, exons: List[tuple]) -> str:
    return ''.join(seq[s:e] for (s, e) in exons)


def _event_span_len(cls_idx: int) -> int:
    if int(cls_idx) in (int(P.START), int(P.STOP)):
        return 3
    if int(cls_idx) in (int(P.DSS), int(P.ASS)):
        return 2
    return 1


def _event_logp(probs: np.ndarray, pos: int, cls_idx: int) -> float:
    L = probs.shape[0]
    span = _event_span_len(cls_idx)
    s = pos
    e = min(L, pos + span)
    v = 0.0
    for i in range(s, e):
        p = float(probs[i, cls_idx])
        v += math.log(max(p, 1e-12))
    return v


def main():
    p = argparse.ArgumentParser(description='Decode gene structures from predicted probabilities')
    p.add_argument('--input-pkl', required=True, help='Pickle file containing List[PredictedSequence]')
    p.add_argument('--topk-per-start', type=int, default=3)
    p.add_argument('--topk-global', type=int, default=10)
    p.add_argument('--beam-size', type=int, default=16)
    p.add_argument('--max-introns', type=int, default=None)
    p.add_argument('--no-overlap', action='store_true')
    p.add_argument('--codon-usage-json', default=None)
    p.add_argument('--lambda', dest='lambda_', type=float, default=0.0)
    p.add_argument('--output-json', required=True)
    p.add_argument('--output-tsv', required=True)
    p.add_argument('--output-fna', required=True)
    args = p.parse_args()

    with open(args.input_pkl, 'rb') as f:
        items: List[PredictedSequence] = pickle.load(f)

    codon_model = None
    if args.codon_usage_json:
        codon_model = CodonUsageModel.from_json(args.codon_usage_json)

    decoded: List[DecodedResult] = []
    ps_map = {ps.sequence_index: ps for ps in items}
    for ps in items:
        res = decode_sequence(
            ps,
            k_per_start=args.topk_per_start,
            k_global=args.topk_global,
            beam_size=args.beam_size,
            max_introns=args.max_introns,
            allow_overlap=not args.no_overlap,
        )

        # Rerank with codon usage if requested
        if codon_model is not None and args.lambda_ != 0.0:
            for start_pos, cands in res.per_start.items():
                for c in cands:
                    cds = _cds_from_exons(ps.sequence, c.exons)
                    c.codon_logp = codon_model.score(cds)
                    c.total = c.boundary_logp + args.lambda_ * c.codon_logp
            for c in res.global_topk:
                cds = _cds_from_exons(ps.sequence, c.exons)
                c.codon_logp = codon_model.score(cds)
                c.total = c.boundary_logp + args.lambda_ * c.codon_logp
            # Resort global by total
            res.global_topk.sort(key=lambda c: c.total, reverse=True)

        decoded.append(res)

    # Write JSON
    out_json = {
        "sequences": [
            {
                "sequence_index": dr.sequence_index,
                "per_start": {str(s): [
                    {
                        "exons": c.exons,
                        "events": c.events,
                        "boundary_logp": c.boundary_logp,
                        "codon_logp": c.codon_logp,
                        "total": c.total,
                        "start_logp": _event_logp(ps_map[dr.sequence_index].probabilities, c.events.get('start', [None])[0], P.START) if c.events.get('start') else None,
                    } for c in cands] for s, cands in dr.per_start.items()
                },
                "global": [
                    {
                        "exons": c.exons,
                        "events": c.events,
                        "boundary_logp": c.boundary_logp,
                        "codon_logp": c.codon_logp,
                        "total": c.total,
                        "start_logp": _event_logp(ps_map[dr.sequence_index].probabilities, c.events.get('start', [None])[0], P.START) if c.events.get('start') else None,
                    } for c in dr.global_topk
                ]
            } for dr in decoded
        ]
    }
    with open(args.output_json, 'w') as f:
        json.dump(out_json, f)

    # Write TSV matching training columns + extra fields
    header = [
        'sequence_id', 'gene_id', 'gene_start', 'gene_end', 'exon_start', 'exon_end', 'strand',
        'k', 'start_rank', 'start_index', 'start_logp', 'boundary_logp', 'codon_logp', 'total'
    ]
    with open(args.output_tsv, 'w') as f:
        f.write('\t'.join(header) + '\n')
        for dr in decoded:
            # Build per-start ranking maps (1-based)
            per_start_rank: dict = {}
            for s, cands in decoded[[d.sequence_index for d in decoded].index(dr)].per_start.items():
                # Determine sort key depending on reranking
                sorted_cands = sorted(cands, key=lambda c: (c.total if c.codon_logp is not None else c.boundary_logp), reverse=True)
                rank_map = {}
                for idx, c in enumerate(sorted_cands, start=1):
                    rank_map[id(c)] = idx
                per_start_rank[s] = rank_map

            # Global ranking k
            for k_rank, cand in enumerate(dr.global_topk, start=1):
                gene_id = f"gene_{k_rank}"
                gene_start = cand.exons[0][0] + 1  # back to 1-based
                gene_end = cand.exons[-1][1]      # end is inclusive in TSV
                # Try to find the originating START index for reporting
                start_idx = cand.events.get('start', [None])[0]
                # Compute start_logp from probabilities
                ps = ps_map[dr.sequence_index]
                start_logp = _event_logp(ps.probabilities, start_idx, P.START) if start_idx is not None else None
                # Determine start_rank based on per-start ordering
                srank = ''
                if start_idx is not None and start_idx in per_start_rank:
                    srank = str(per_start_rank[start_idx].get(id(cand), 1))
                for ex_idx, (xs, xe) in enumerate(cand.exons):
                    row = [
                        str(dr.sequence_index),
                        gene_id,
                        str(gene_start),
                        str(gene_end),
                        str(xs + 1),
                        str(xe),
                        '+',
                        str(k_rank),
                        srank if srank != '' else '1',
                        str(start_idx if start_idx is not None else ''),
                        f"{start_logp:.6f}" if start_logp is not None else '',
                        f"{cand.boundary_logp:.6f}",
                        f"{cand.codon_logp:.6f}" if cand.codon_logp is not None else '',
                        f"{cand.total:.6f}",
                    ]
                    f.write('\t'.join(row) + '\n')

    # Write FNA (input sequences)
    with open(args.output_fna, 'w') as f:
        for ps in items:
            f.write(f">sequence_{ps.sequence_index}\n")
            f.write(ps.sequence + "\n")

    print(f"✓ Wrote {args.output_json}, {args.output_tsv}, {args.output_fna}")


if __name__ == '__main__':
    main()



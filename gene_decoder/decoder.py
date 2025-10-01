from typing import List, Dict, Tuple, Optional
import numpy as np
import math
from utils.constants import GenePredictionClass as P, ConventionalStopCodons, ConventionalDonorDinucleotides, ConventionalAcceptorDinucleotides
from gene_decoder import PredictedSequence, CandidateGene, DecodedResult


def _log(x: float) -> float:
    return math.log(max(x, 1e-12))


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
        v += _log(float(probs[i, cls_idx]))
    return v


def _scan_events(seq: str, probs: np.ndarray) -> Dict[str, List[int]]:
    starts: List[int] = []
    stops: List[int] = []
    dss: List[int] = []
    ass: List[int] = []
    L = len(seq)
    for i in range(L - 2):
        tri = seq[i:i+3]
        if tri == 'ATG':
            starts.append(i)
        if tri in ConventionalStopCodons:
            stops.append(i)
    for i in range(L - 1):
        di = seq[i:i+2]
        if di in ConventionalDonorDinucleotides:
            dss.append(i)
        if di in ConventionalAcceptorDinucleotides:
            ass.append(i)
    return {"start": starts, "stop": stops, "dss": dss, "ass": ass}


def _decode_from_start(ps: PredictedSequence, start_pos: int, k: int = 3, beam_size: int = 16,
                       max_introns: Optional[int] = None) -> List[CandidateGene]:
    seq = ps.sequence
    probs = ps.probabilities
    events = _scan_events(seq, probs)

    # Pre-filter ASS and DSS after START
    dss_positions = [d for d in events["dss"] if d >= start_pos + 3]
    ass_positions = [a for a in events["ass"] if a >= start_pos + 3]
    stop_positions = [t for t in events["stop"] if t >= start_pos + 3]

    start_lp = _event_logp(probs, start_pos, P.START)

    # Beam items: (pos, phase, boundary_logp, exons, ev_map, intron_count, current_exon_start)
    # Start in exon phase 0 at start_pos
    beams: List[Tuple[int, int, float, List[Tuple[int, int]], Dict[str, List[int]], int, int]] = []
    beams.append((start_pos + 3, 0, start_lp, [], {"start": [start_pos], "stop": [], "dss": [], "ass": []}, 0, start_pos))

    completed: List[CandidateGene] = []

    # Pre-index ASS by following DSS and vice versa using genomic order
    ass_set = set(ass_positions)
    dss_set = set(dss_positions)

    L = len(seq)
    while beams:
        # Expand beam by next event (either STOP to finish or DSS to intron)
        new_beam: List[Tuple[int, int, float, List[Tuple[int, int]], Dict[str, List[int]], int, int]] = []
        for (pos, phase, lp, exons, ev, n_introns, exon_start) in beams:
            # Try STOPs after current position that align with phase 0
            for stop in stop_positions:
                if stop < pos:
                    continue
                # Exon extends to stop+3
                exon_len = (stop + 3) - exon_start
                new_phase = (phase + (exon_len % 3)) % 3
                if new_phase != 0:
                    continue
                st_lp = _event_logp(probs, stop, P.STOP)
                total_lp = lp + _log(1.0) + st_lp
                exons_fin = list(exons)
                exons_fin.append((exon_start, stop + 3))
                ev_fin = {k: list(v) for k, v in ev.items()}
                ev_fin["stop"].append(stop)
                completed.append(CandidateGene(exons=exons_fin, events=ev_fin, boundary_logp=total_lp, codon_logp=None, total=total_lp))
                if len(completed) >= k:
                    break
            if len(completed) >= k:
                break

            # Try DSS->ASS to create intron and continue exon
            if max_introns is not None and n_introns >= max_introns:
                continue
            # Next donor after current pos
            for d in dss_positions:
                if d < pos:
                    continue
                # Find the first acceptor after donor
                for a in ass_positions:
                    if a <= d:
                        continue
                    # Close current exon at donor, open next exon at acceptor+2
                    exon_end = d
                    exon_len = exon_end - exon_start
                    new_phase = (phase + (exon_len % 3)) % 3
                    next_exon_start = a + 2
                    d_lp = _event_logp(probs, d, P.DSS)
                    a_lp = _event_logp(probs, a, P.ASS)
                    new_lp = lp + d_lp + a_lp
                    new_exons = list(exons)
                    new_exons.append((exon_start, exon_end))
                    new_ev = {k: list(v) for k, v in ev.items()}
                    new_ev["dss"].append(d)
                    new_ev["ass"].append(a)
                    new_beam.append((next_exon_start, new_phase, new_lp, new_exons, new_ev, n_introns + 1, next_exon_start))
                    # Beam pruning by size
                    if len(new_beam) > beam_size:
                        # Keep top by logp
                        new_beam.sort(key=lambda t: t[2], reverse=True)
                        new_beam = new_beam[:beam_size]
                # Optional early stop if we have enough expansions
        # Prepare next iteration
        if not new_beam:
            break
        new_beam.sort(key=lambda t: t[2], reverse=True)
        beams = new_beam[:beam_size]
        if len(completed) >= k:
            break

    # Sort completed by boundary_logp and return top-k
    completed.sort(key=lambda c: c.boundary_logp, reverse=True)
    return completed[:k]


def decode_sequence(ps: PredictedSequence, k_per_start: int = 3, k_global: int = 10, beam_size: int = 16,
                    max_introns: Optional[int] = None, allow_overlap: bool = True) -> DecodedResult:
    seq = ps.sequence
    probs = ps.probabilities
    events = _scan_events(seq, probs)

    per_start: Dict[int, List[CandidateGene]] = {}
    all_candidates: List[CandidateGene] = []
    for s in events["start"]:
        cands = _decode_from_start(ps, s, k=k_per_start, beam_size=beam_size, max_introns=max_introns)
        per_start[s] = cands
        all_candidates.extend(cands)

    # Global top-k with optional non-overlap policy (simple score sort; no packing)
    all_candidates.sort(key=lambda c: c.total, reverse=True)
    global_top = []
    if allow_overlap:
        global_top = all_candidates[:k_global]
    else:
        # Greedy non-overlapping selection by score
        used: List[Tuple[int, int]] = []
        for c in all_candidates:
            span = (c.exons[0][0], c.exons[-1][1]) if c.exons else (0, 0)
            if any(not (span[1] <= u[0] or span[0] >= u[1]) for u in used):
                continue
            global_top.append(c)
            used.append(span)
            if len(global_top) >= k_global:
                break

    return DecodedResult(sequence_index=ps.sequence_index, per_start=per_start, global_topk=global_top)



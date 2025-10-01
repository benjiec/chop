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


def _scan_events(seq: str) -> Dict[str, List[int]]:
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


def _decode_from_start(ps: PredictedSequence, start_pos: int, events: Dict[str, List[int]],
                      k: int = 3, beam_size: int = 16, max_introns: Optional[int] = None,
                      scoring: str = 'boundary') -> List[CandidateGene]:
    seq = ps.sequence
    probs = ps.probabilities

    # Precompute event presence and scores from start onward
    L = len(seq)
    has_dss = np.zeros(L, dtype=bool)
    has_ass = np.zeros(L, dtype=bool)
    has_stop = np.zeros(L, dtype=bool)
    lp_dss = np.full(L, float('-inf'))
    lp_ass = np.full(L, float('-inf'))
    lp_stop = np.full(L, float('-inf'))
    p_dss = np.zeros(L, dtype=float)
    p_ass = np.zeros(L, dtype=float)
    p_stop = np.zeros(L, dtype=float)

    for d in events["dss"]:
        if d >= start_pos + 3:
            has_dss[d] = True
            lp_dss[d] = _event_logp(probs, d, P.DSS)
            # convert to prob over span for hazard skipping
            span = _event_span_len(P.DSS)
            p = 1.0
            for i in range(d, min(L, d + span)):
                p *= max(1e-12, float(probs[i, P.DSS]))
            p_dss[d] = p
    for a in events["ass"]:
        if a >= start_pos + 3:
            has_ass[a] = True
            lp_ass[a] = _event_logp(probs, a, P.ASS)
            span = _event_span_len(P.ASS)
            p = 1.0
            for i in range(a, min(L, a + span)):
                p *= max(1e-12, float(probs[i, P.ASS]))
            p_ass[a] = p
    for t in events["stop"]:
        if t >= start_pos + 3:
            has_stop[t] = True
            lp_stop[t] = _event_logp(probs, t, P.STOP)
            span = _event_span_len(P.STOP)
            p = 1.0
            for i in range(t, min(L, t + span)):
                p *= max(1e-12, float(probs[i, P.STOP]))
            p_stop[t] = p

    start_lp = _event_logp(probs, start_pos, P.START)

    # Beam item: (pos, state, phase, log_score, exons, events_map, intron_count, exon_start)
    EXON, INTRON = 0, 1
    beams: List[Tuple[int, int, int, float, List[Tuple[int, int]], Dict[str, List[int]], int, int]] = []
    beams.append((start_pos + 3, EXON, 0, start_lp, [], {"start": [start_pos], "stop": [], "dss": [], "ass": []}, 0, start_pos))

    completed: List[CandidateGene] = []

    for i in range(start_pos + 3, L):
        if not beams:
            break
        next_beam: List[Tuple[int, int, int, float, List[Tuple[int, int]], Dict[str, List[int]], int, int]] = []
        for (pos, state, phase, lp, exons, ev, n_introns, exon_start) in beams:
            if pos > i:
                # carry forward untouched
                next_beam.append((pos, state, phase, lp, exons, ev, n_introns, exon_start))
                continue
            if state == EXON:
                # STOP (only if in-frame)
                if has_stop[i] and phase == 0:
                    total_lp = lp + lp_stop[i]
                    exons_fin = list(exons)
                    exons_fin.append((exon_start, i + 3))
                    ev_fin = {k: list(v) for k, v in ev.items()}
                    ev_fin["stop"].append(i)
                    completed.append(CandidateGene(exons=exons_fin, events=ev_fin, boundary_logp=total_lp, codon_logp=None, total=total_lp))
                    # Do not allow staying or splicing past an in-frame STOP at the same position
                    continue
                else:
                    # DSS -> INTRON
                    if (max_introns is None or n_introns < max_introns) and has_dss[i]:
                        new_lp = lp + lp_dss[i]
                        new_exons = list(exons)
                        new_exons.append((exon_start, i))
                        new_ev = {k: list(v) for k, v in ev.items()}
                        new_ev["dss"].append(i)
                        next_beam.append((i + 2, INTRON, phase, new_lp, new_exons, new_ev, n_introns + 1, -1))
                    # stay in EXON
                    if i + 1 < L:
                        stay_lp = lp
                        if scoring == 'hazard':
                            if has_dss[i]:
                                stay_lp += _log(max(1e-12, 1.0 - p_dss[i]))
                        next_beam.append((i + 1, EXON, (phase + 1) % 3, stay_lp, exons, ev, n_introns, exon_start))
            else:  # INTRON
                # ASS -> EXON
                if has_ass[i]:
                    new_lp = lp + lp_ass[i]
                    new_ev = {k: list(v) for k, v in ev.items()}
                    new_ev["ass"].append(i)
                    next_beam.append((i + 2, EXON, phase, new_lp, exons, new_ev, n_introns, i + 2))
                # stay in INTRON
                if i + 1 < L:
                    stay_lp = lp
                    if scoring == 'hazard' and has_ass[i]:
                        stay_lp += _log(max(1e-12, 1.0 - p_ass[i]))
                    next_beam.append((i + 1, INTRON, phase, stay_lp, exons, ev, n_introns, exon_start))

        if not next_beam:
            break
        next_beam.sort(key=lambda t: t[3], reverse=True)
        beams = next_beam[:beam_size]
        if len(completed) >= k:
            break

    completed.sort(key=lambda c: c.boundary_logp, reverse=True)
    return completed[:k]


def decode_sequence(ps: PredictedSequence, k_per_start: int = 3, k_global: int = 10, beam_size: int = 16,
                    max_introns: Optional[int] = None, allow_overlap: bool = True, scoring: str = 'boundary') -> DecodedResult:
    seq = ps.sequence
    events = _scan_events(seq)

    per_start: Dict[int, List[CandidateGene]] = {}
    all_candidates: List[CandidateGene] = []
    for s in events["start"]:
        cands = _decode_from_start(ps, s, events, k=k_per_start, beam_size=beam_size, max_introns=max_introns, scoring=scoring)
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



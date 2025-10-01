from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import uuid
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

    @dataclass
    class Beam:
        id: Optional[str]
        pos: int
        state: int
        phase: int
        boundary_lp: float
        total_lp: float
        exons: List[Tuple[int, int]]
        events: Dict[str, List[int]]
        intron_count: int
        exon_start: int

        def score(self, mode: str) -> float:
            return self.total_lp if mode == 'hazard' else self.boundary_lp

        def __post_init__(self) -> None:
            if self.id is None:
                self.id = uuid.uuid4().hex[:8]

    EXON, INTRON = 0, 1
    beams: List[Beam] = []
    beams.append(Beam(
        id=None,
        pos=start_pos + 3,
        state=EXON,
        phase=0,
        boundary_lp=start_lp,
        total_lp=start_lp,
        exons=[],
        events={"start": [start_pos], "stop": [], "dss": [], "ass": []},
        intron_count=0,
        exon_start=start_pos,
    ))
    print("START @", start_pos)

    completed: List[CandidateGene] = []

    for i in range(start_pos + 3, L):
        if not beams:
            break
        next_beam: List[Beam] = []
        for b in beams:
            if b.pos > i:
                # carry forward untouched
                print(b.id, "carry forward @", i)
                next_beam.append(b)
                continue
            if b.state == EXON:
                # STOP (only if in-frame)
                if has_stop[i] and b.phase == 0:
                    # Recompute boundary strictly from taken events to avoid drift
                    exons_fin = list(b.exons)
                    exons_fin.append((b.exon_start, i + 3))
                    ev_fin = {k: list(v) for k, v in b.events.items()}
                    ev_fin["stop"].append(i)
                    # Sum boundary over events
                    boundary_stop = 0.0
                    for pos in ev_fin.get("start", []):
                        boundary_stop += _event_logp(probs, pos, P.START)
                    for pos in ev_fin.get("dss", []):
                        boundary_stop += _event_logp(probs, pos, P.DSS)
                    for pos in ev_fin.get("ass", []):
                        boundary_stop += _event_logp(probs, pos, P.ASS)
                    for pos in ev_fin.get("stop", []):
                        boundary_stop += _event_logp(probs, pos, P.STOP)
                    # Total = boundary + accumulated hazard penalties (difference between total and boundary trackers)
                    hazard_delta = b.total_lp - b.boundary_lp
                    total_stop = boundary_stop + hazard_delta
                    completed.append(CandidateGene(exons=exons_fin, events=ev_fin, boundary_logp=boundary_stop, codon_logp=None, total=total_stop))
                    # Do not allow staying or splicing past an in-frame STOP at the same position
                    print(b.id, "EXON+STOP @", i, "total score", b.total_lp)
                    continue
                else:
                    # DSS -> INTRON
                    if (max_introns is None or b.intron_count < max_introns) and has_dss[i]:
                        new_b_lp = b.boundary_lp + lp_dss[i]
                        new_t_lp = b.total_lp + lp_dss[i]
                        new_exons = list(b.exons)
                        new_exons.append((b.exon_start, i))
                        new_ev = {k: list(v) for k, v in b.events.items()}
                        new_ev["dss"].append(i)
                        next_beam.append(Beam(id=None, pos=i + 2, state=INTRON, phase=b.phase, boundary_lp=new_b_lp, total_lp=new_t_lp, exons=new_exons, events=new_ev, intron_count=b.intron_count + 1, exon_start=-1))
                        print(b.id, "EXON+DSS @", i, "new INTRON beam at", i+2, "total score", new_t_lp, "id", next_beam[-1].id)
                    # stay in EXON
                    if i + 1 < L:
                        stay_b_lp = b.boundary_lp
                        stay_t_lp = b.total_lp
                        if scoring == 'hazard':
                            if has_dss[i]:
                                stay_t_lp += _log(max(1e-12, 1.0 - p_dss[i]))
                                print(b.id, "EXON+DSS @", i, "skipped total score", stay_t_lp)
                        next_beam.append(Beam(id=b.id, pos=i + 1, state=EXON, phase=(b.phase + 1) % 3, boundary_lp=stay_b_lp, total_lp=stay_t_lp, exons=b.exons, events=b.events, intron_count=b.intron_count, exon_start=b.exon_start))
            else:  # INTRON
                # ASS -> EXON
                if has_ass[i]:
                    new_b_lp = b.boundary_lp + lp_ass[i]
                    new_t_lp = b.total_lp + lp_ass[i]
                    new_ev = {k: list(v) for k, v in b.events.items()}
                    new_ev["ass"].append(i)
                    next_beam.append(Beam(id=None, pos=i + 2, state=EXON, phase=b.phase, boundary_lp=new_b_lp, total_lp=new_t_lp, exons=b.exons, events=new_ev, intron_count=b.intron_count, exon_start=i + 2))
                    print(b.id, "INTRON+ASS @", i, "new EXON beam at", i+2, "total score", new_t_lp, "id", next_beam[-1].id)
                # stay in INTRON
                if i + 1 < L:
                    stay_b_lp = b.boundary_lp
                    stay_t_lp = b.total_lp
                    if scoring == 'hazard' and has_ass[i]:
                        stay_t_lp += _log(max(1e-12, 1.0 - p_ass[i]))
                        print(b.id, "INTRON+ASS @", i, "skipped total score", stay_t_lp)
                    next_beam.append(Beam(id=b.id, pos=i + 1, state=INTRON, phase=b.phase, boundary_lp=stay_b_lp, total_lp=stay_t_lp, exons=b.exons, events=b.events, intron_count=b.intron_count, exon_start=b.exon_start))

        if not next_beam:
            break
        # Sort by mode-appropriate score
        next_beam.sort(key=lambda bb: bb.score(scoring), reverse=True)
        beams = next_beam[:beam_size]
        if len(completed) >= k:
            break

    # Sort completed candidates by appropriate score
    if scoring == 'hazard':
        completed.sort(key=lambda c: c.total, reverse=True)
    else:
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



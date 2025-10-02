from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
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


def _event_logp(probs: np.ndarray, pos: int, cls_idx: int, negative: Optional[bool] = False) -> float:
    L = probs.shape[0]
    span = _event_span_len(cls_idx)
    s = pos
    e = min(L, pos + span)
    if e <= s:
        return _log(1e-12)
    vals = probs[s:e, cls_idx].astype(float)
    mean_p = float(np.clip(np.mean(vals), 1e-12, 1.0))
    if negative:
        mean_p = 1.0 - mean_p
    return _log(mean_p)


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


@dataclass
class Beam:
    # Beam states (class attributes) - define after fields to avoid dataclass init ordering issues
    id: Optional[str]
    pos: int
    state: int
    phase: int
    exons: List[Tuple[int, int]]
    events: Dict[str, List[int]] = field(default_factory=lambda: {"start": [], "dss": [], "ass": [], "stop": []})
    skips: Dict[str, List[int]] = field(default_factory=lambda: {"dss": [], "ass": []})
    intron_count: int = 0
    exon_start: int = -1

    # Class attributes (not part of __init__ parameters)
    EXON: int = 0
    INTRON: int = 1

    def __post_init__(self) -> None:
        if self.id is None:
            self.id = uuid.uuid4().hex[:8]

    def forward(self, not_taken: Optional[str], idx: int, hazard_penalty: float = 0.0) -> None:
        # Advance position by one and update phase if in EXON
        self.pos = self.pos + 1
        if self.state == Beam.EXON:
            self.phase = (self.phase + 1) % 3
        # Record a skip when applicable (for hazard scoring)
        if not_taken in ("dss", "ass"):
            self.skips[not_taken].append(idx)

    def branch_to_new_state(self, event: str, idx: int) -> "Beam":
        # Clone current beam to new branch with new id and flip state according to event
        if event == "dss" and self.state == Beam.EXON:
            # Close exon at idx and enter INTRON at idx+2
            new_exons = list(self.exons)
            new_exons.append((self.exon_start, idx))
            new_events = {k: list(v) for k, v in self.events.items()}
            new_events["dss"].append(idx)
            return Beam(
                id=None,
                pos=idx + 2,
                state=Beam.INTRON,
                phase=self.phase,
                exons=new_exons,
                events=new_events,
                skips={k: list(v) for k, v in self.skips.items()},
                intron_count=self.intron_count + 1,
                exon_start=-1,
            )
        elif event == "ass" and self.state == Beam.INTRON:
            # Enter EXON at idx+2 and set new exon_start
            new_events = {k: list(v) for k, v in self.events.items()}
            new_events["ass"].append(idx)
            return Beam(
                id=None,
                pos=idx + 2,
                state=Beam.EXON,
                phase=self.phase,
                exons=list(self.exons),
                events=new_events,
                skips={k: list(v) for k, v in self.skips.items()},
                intron_count=self.intron_count,
                exon_start=idx + 2,
            )
        raise ValueError(f"Invalid branch_to_new_state from state={self.state} via event={event}")

    @staticmethod
    def start(start_pos: int) -> "Beam":
        return Beam(
            id=None,
            pos=start_pos + 3,
            state=Beam.EXON,
            phase=0,
            exons=[],
            events={"start": [start_pos], "dss": [], "ass": [], "stop": []},
            skips={"dss": [], "ass": []},
            intron_count=0,
            exon_start=start_pos,
        )

    def events_copy(self) -> Dict[str, List[int]]:
        return {k: list(v) for k, v in self.events.items()}

    def stop(self, idx: int) -> "Beam":
        # Append stop event and close the current exon if in EXON
        new_events = self.events_copy()
        new_events["stop"].append(idx)
        new_exons = list(self.exons)
        if self.state == Beam.EXON:
            new_exons.append((self.exon_start, idx + 3))
        return Beam(
            id=self.id,
            pos=self.pos,
            state=self.state,
            phase=self.phase,
            exons=new_exons,
            events=new_events,
            skips={k: list(v) for k, v in self.skips.items()},
            intron_count=self.intron_count,
            exon_start=self.exon_start,
        )

    def compute_exons(self) -> List[Tuple[int, int]]:
        return list(self.exons)

    def compute_scores(self, probs: np.ndarray, verbose: Optional[bool] = False) -> Tuple[float, float]:
        ordered_log = []

        # boundary = sum of taken events
        boundary = 0.0
        for pos in self.events.get("start", []):
            event_score = _event_logp(probs, pos, P.START, False)
            boundary += event_score
            ordered_log.append((pos, "start", event_score))
        for pos in self.events.get("dss", []):
            event_score = _event_logp(probs, pos, P.DSS, False)
            boundary += event_score
            ordered_log.append((pos, "dss+", event_score))
        for pos in self.events.get("ass", []):
            event_score = _event_logp(probs, pos, P.ASS, False)
            boundary += event_score
            ordered_log.append((pos, "ass+", event_score))
        for pos in self.events.get("stop", []):
            event_score = _event_logp(probs, pos, P.STOP, False)
            boundary += event_score
            ordered_log.append((pos, "stop", event_score))

        # hazard penalties from skipped eligible transitions recorded during forward moves
        hazard = 0.0
        for idx in self.skips.get("dss", []):
            event_score = _event_logp(probs, idx, P.DSS, True)
            hazard += event_score
            ordered_log.append((idx, "dss-", event_score))
        for idx in self.skips.get("ass", []):
            event_score = _event_logp(probs, idx, P.ASS, True)
            hazard += event_score
            ordered_log.append((idx, "ass-", event_score))

        total = boundary + hazard

        if verbose:
            for pos, event, score in sorted(ordered_log, key=lambda t: t[0]):
                print(self.id, event, "@", pos, "+=", score)
            print(self.id, "total", total)

        return boundary, total

    def create_candidate_gene(self, probs: np.ndarray) -> CandidateGene:
        boundary, total = self.compute_scores(probs, verbose=True)
        return CandidateGene(
            exons=self.compute_exons(),
            events=self.events_copy(),
            boundary_logp=boundary,
            codon_logp=None,
            total=total,
        )


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

    for d in events["dss"]:
        if d >= start_pos + 3:
            has_dss[d] = True
    for a in events["ass"]:
        if a >= start_pos + 3:
            has_ass[a] = True
    for t in events["stop"]:
        if t >= start_pos + 3:
            has_stop[t] = True

    beams: List[Beam] = []
    beams.append(Beam.start(start_pos))

    completed: List[CandidateGene] = []

    for i in range(start_pos + 3, L):
        if not beams:
            break
        next_beam: List[Beam] = []
        for b in beams:
            if b.pos > i:
                # carry forward untouched
                next_beam.append(b)
                continue
            if b.state == Beam.EXON:
                # STOP (only if in-frame)
                if has_stop[i] and b.phase == 0:
                    # Build final candidate with both boundary and hazard totals
                    b_final = b.stop(i)
                    candidate = b_final.create_candidate_gene(probs)
                    completed.append(candidate)
                    # Do not allow staying or splicing past an in-frame STOP at the same position
                    continue
                else:
                    # DSS -> INTRON
                    if (max_introns is None or b.intron_count < max_introns) and has_dss[i]:
                        child = b.branch_to_new_state("dss", i)
                        next_beam.append(child)
                    # stay in EXON
                    if i + 1 < L:
                        b.forward("dss" if has_dss[i] else None, i)
                        next_beam.append(b)
            else:  # INTRON
                # ASS -> EXON
                if has_ass[i]:
                    child = b.branch_to_new_state("ass", i)
                    next_beam.append(child)
                # stay in INTRON
                if i + 1 < L:
                    b.forward("ass" if has_ass[i] else None, i)
                    next_beam.append(b)

        if not next_beam:
            break
        # Sort by mode-appropriate score (compute on the fly)
        if scoring == 'hazard':
            next_beam.sort(key=lambda bb: bb.compute_scores(probs)[1], reverse=True)
        else:
            next_beam.sort(key=lambda bb: bb.compute_scores(probs)[0], reverse=True)
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



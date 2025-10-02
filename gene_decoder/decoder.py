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


def _scan_events(seq: str, probs: Optional[np.ndarray] = None, min_logp: Optional[float] = None) -> Dict[str, List[int]]:
    starts: List[int] = []
    stops: List[int] = []
    dss: List[int] = []
    ass: List[int] = []
    L = len(seq)
    for i in range(L - 2):
        tri = seq[i:i+3]
        if tri == 'ATG':
            if probs is not None and min_logp is not None:
                if _event_logp(probs, i, P.START) > min_logp:
                    starts.append(i)
            else:
                starts.append(i)
        if tri in ConventionalStopCodons:
            if probs is not None and min_logp is not None:
                if _event_logp(probs, i, P.STOP) > min_logp:
                    stops.append(i)
            else:
                stops.append(i)
    for i in range(L - 1):
        di = seq[i:i+2]
        if di in ConventionalDonorDinucleotides:
            if probs is not None and min_logp is not None:
                if _event_logp(probs, i, P.DSS) > min_logp:
                    dss.append(i)
            else:
                dss.append(i)
        if di in ConventionalAcceptorDinucleotides:
            if probs is not None and min_logp is not None:
                if _event_logp(probs, i, P.ASS) > min_logp:
                    ass.append(i)
            else:
                ass.append(i)
    return {"start": starts, "stop": stops, "dss": dss, "ass": ass}


@dataclass
class Beam:
    # Beam states (class attributes) - define after fields to avoid dataclass init ordering issues
    id: str
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

    def events_copy(self) -> Dict[str, List[int]]:
        return {k: list(v) for k, v in self.events.items()}

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
                      k: int = 3, beam_size: int = 16,
                      scoring: str = 'boundary') -> List[CandidateGene]:
    seq = ps.sequence
    probs = ps.probabilities
    beam_id = uuid.uuid4().hex[:8]

    # Precompute event presence from start onward
    L = len(seq)
    has_dss = np.zeros(L, dtype=bool)
    has_ass = np.zeros(L, dtype=bool)
    has_stop = np.zeros(L, dtype=bool)
    for d in sorted(events["dss"]):
        if d >= start_pos + 3:
            has_dss[d] = True
    for a in sorted(events["ass"]):
        if a >= start_pos + 3:
            has_ass[a] = True
    for t in sorted(events["stop"]):
        if t >= start_pos + 3:
            has_stop[t] = True

    # DP: memoize top-k suffix beams from (i,state,phase,exon_start)
    from functools import lru_cache

    def score_key(b: Beam) -> float:
        boundary, total = b.compute_scores(probs)
        return total if scoring == 'hazard' else boundary

    def merge_prefix_suffix(prefix: Beam, tail: Beam) -> Beam:
        # Combine events, skips, and exons; take tail's position/state/phase trackers
        ev = {
            "start": list(prefix.events.get("start", [])) + list(tail.events.get("start", [])),
            "dss": list(prefix.events.get("dss", [])) + list(tail.events.get("dss", [])),
            "ass": list(prefix.events.get("ass", [])) + list(tail.events.get("ass", [])),
            "stop": list(prefix.events.get("stop", [])) + list(tail.events.get("stop", [])),
        }
        sk = {
            "dss": list(prefix.skips.get("dss", [])) + list(tail.skips.get("dss", [])),
            "ass": list(prefix.skips.get("ass", [])) + list(tail.skips.get("ass", [])),
        }
        ex = list(prefix.exons) + list(tail.exons)
        return Beam(
            id=beam_id,
            pos=tail.pos,
            state=tail.state,
            phase=tail.phase,
            exons=ex,
            events=ev,
            skips=sk,
            intron_count=tail.intron_count,
            exon_start=tail.exon_start,
        )

    @lru_cache(maxsize=None)
    def dp(i: int, state: int, phase: int, exon_start: int) -> Tuple[Beam, ...]:
        if i >= L:
            return tuple()
        results: List[Beam] = []

        if state == Beam.EXON:
            # Scan forward until next event; recurse only at event points
            j = i
            while j < L:
                # If STOP begins at j and in-frame, terminate here (earliest STOP)
                if has_stop[j] and ((phase + (j - i)) % 3 == 0):
                    stop_ev = {"start": [], "dss": [], "ass": [], "stop": [j]}
                    exons_list: List[Tuple[int, int]] = []
                    if exon_start >= 0:
                        exons_list.append((exon_start, j + 3))
                    stop_beam = Beam(
                        id=beam_id,
                        pos=j + 3,
                        state=Beam.EXON,
                        phase=(phase + (j - i)) % 3,
                        exons=exons_list,
                        events=stop_ev,
                        skips={"dss": [], "ass": []},
                        intron_count=0,
                        exon_start=exon_start,
                    )
                    results.append(stop_beam)
                    break
                # If DSS at j: branch by taking it; or skip and continue scanning
                if has_dss[j]:
                    # take branch: enter INTRON at j+2
                    take_prefix = Beam(
                        id=beam_id,
                        pos=j + 2,
                        state=Beam.INTRON,
                        phase=(phase + (j - i)) % 3,
                        exons=[(exon_start, j)],
                        events={"start": [], "dss": [j], "ass": [], "stop": []},
                        skips={"dss": [], "ass": []},
                        intron_count=0,
                        exon_start=-1,
                    )
                    for tail in dp(j + 2, Beam.INTRON, (phase + (j - i)) % 3, -1):
                        results.append(merge_prefix_suffix(take_prefix, tail))
                    # skip-one branch: remain in EXON at j+1, record single skip at j
                    skip_prefix = Beam(
                        id=beam_id,
                        pos=j + 1,
                        state=Beam.EXON,
                        phase=(phase + (j + 1 - i)) % 3,
                        exons=[],
                        events={"start": [], "dss": [], "ass": [], "stop": []},
                        skips={"dss": [j], "ass": []},
                        intron_count=0,
                        exon_start=exon_start,
                    )
                    for tail in dp(j + 1, Beam.EXON, (phase + (j + 1 - i)) % 3, exon_start):
                        results.append(merge_prefix_suffix(skip_prefix, tail))
                    break
                j += 1
        else:
            # INTRON: scan to next ASS; recurse only at ASS points
            j = i
            while j < L:
                if has_ass[j]:
                    # take branch: enter EXON at j+2
                    take_prefix = Beam(
                        id=beam_id,
                        pos=j + 2,
                        state=Beam.EXON,
                        phase=phase,
                        exons=[],
                        events={"start": [], "dss": [], "ass": [j], "stop": []},
                        skips={"dss": [], "ass": []},
                        intron_count=0,
                        exon_start=j + 2,
                    )
                    for tail in dp(j + 2, Beam.EXON, phase, j + 2):
                        results.append(merge_prefix_suffix(take_prefix, tail))
                    # skip-one branch: stay in INTRON at j+1, record single skip at j
                    skip_prefix = Beam(
                        id=beam_id,
                        pos=j + 1,
                        state=Beam.INTRON,
                        phase=phase,
                        exons=[],
                        events={"start": [], "dss": [], "ass": [], "stop": []},
                        skips={"dss": [], "ass": [j]},
                        intron_count=0,
                        exon_start=-1,
                    )
                    for tail in dp(j + 1, Beam.INTRON, phase, -1):
                        results.append(merge_prefix_suffix(skip_prefix, tail))
                    break
                j += 1

        # Rank and keep top-k at this state
        results.sort(key=score_key, reverse=True)
        if len(results) > k:
            results = results[:k]
        return tuple(results)

    # Get suffix completions from start state, then add START event and build candidates
    suffixes = dp(start_pos + 3, Beam.EXON, 0, start_pos)
    candidates: List[CandidateGene] = []
    for tail in suffixes:
        # add START event
        with_start = Beam(
            id=beam_id,
            pos=tail.pos,
            state=tail.state,
            phase=tail.phase,
            exons=list(tail.exons),
            events={
                "start": [start_pos] + list(tail.events.get("start", [])),
                "dss": list(tail.events.get("dss", [])),
                "ass": list(tail.events.get("ass", [])),
                "stop": list(tail.events.get("stop", [])),
            },
            skips={"dss": list(tail.skips.get("dss", [])), "ass": list(tail.skips.get("ass", []))},
            intron_count=tail.intron_count,
            exon_start=tail.exon_start,
        )
        candidates.append(with_start.create_candidate_gene(probs))

    # Sort and return top-k
    if scoring == 'hazard':
        candidates.sort(key=lambda c: c.total, reverse=True)
    else:
        candidates.sort(key=lambda c: c.boundary_logp, reverse=True)
    return candidates[:k]


def decode_sequence(ps: PredictedSequence, k_per_start: int = 3, k_global: int = 10, beam_size: int = 16,
                    allow_overlap: bool = True, scoring: str = 'boundary', min_logp: Optional[float] = None) -> DecodedResult:
    seq = ps.sequence
    if min_logp is None:
        min_logp = math.log(0.1)
    events = _scan_events(seq, ps.probabilities, min_logp=min_logp)

    per_start: Dict[int, List[CandidateGene]] = {}
    all_candidates: List[CandidateGene] = []
    for s in events["start"]:
        cands = _decode_from_start(ps, s, events, k=k_per_start, beam_size=beam_size, scoring=scoring)
        per_start[s] = cands
        all_candidates.extend(cands)

    # Global top-k with optional non-overlap policy (simple score sort; no packing)
    key_fn = (lambda c: c.total) if scoring == 'hazard' else (lambda c: c.boundary_logp)
    all_candidates.sort(key=key_fn, reverse=True)
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



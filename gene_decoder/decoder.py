from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
import uuid
import numpy as np
import math
from gene_decoder import PredictedSequence, CandidateGene, DecodedResult
from utils.constants import (
    GenePredictionClass as P,
    ConventionalStopCodons,
    ConventionalAcceptorDinucleotides
)


def _debug_log_scan(j):
    return True


def _debug_log_predicted():
    return False


def _debug_log_dp(j):
    idxs = []
    for idx in idxs:
        if abs(j-idx) < 5:
            return True
    return False


def _log(x: float) -> float:
    return math.log(max(x, 1e-12))


def _event_span_len(cls_idx: int) -> int:
    if int(cls_idx) in (int(P.START), int(P.STOP)):
        return 3
    if int(cls_idx) in (int(P.DSS), int(P.ASS)):
        return 2
    return 1


def _event_prob(probs: np.ndarray, pos: int, cls_idx: int, negative: Optional[bool] = False) -> float:
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
    return mean_p


def _event_logp(probs: np.ndarray, pos: int, cls_idx: int, negative: Optional[bool] = False) -> float:
    mean_p = _event_prob(probs, pos, cls_idx, negative=negative)
    return _log(mean_p)


def _scan_events(seq: str, dss_motifs: List[str], probs: Optional[np.ndarray] = None,
                 min_logp_start: Optional[float] = None,
                 min_logp_stop: Optional[float] = None,
                 min_logp_dss: Optional[float] = None,
                 min_logp_ass: Optional[float] = None) -> Dict[str, List[int]]:
    starts: List[int] = []
    stops: List[int] = []
    dss: List[int] = []
    ass: List[int] = []
    L = len(seq)
    for i in range(L - 2):
        tri = seq[i:i+3]
        if tri == 'ATG':
            if probs is not None and min_logp_start is not None:
                if _event_logp(probs, i, P.START) > min_logp_start:
                    if _debug_log_scan(i):
                        print("START", i, _event_prob(probs, i, P.START), _event_logp(probs, i, P.START))
                    starts.append(i)
            else:
                starts.append(i)
        if tri in ConventionalStopCodons:
            if probs is not None and min_logp_stop is not None:
                if _event_logp(probs, i, P.STOP) > min_logp_stop:
                    if _debug_log_scan(i):
                        print(" STOP", i, _event_prob(probs, i, P.STOP), _event_logp(probs, i, P.STOP))
                    stops.append(i)
            else:
                stops.append(i)
    for i in range(L - 1):
        di = seq[i:i+2]
        if di in dss_motifs:
            if probs is not None and min_logp_dss is not None:
                if _event_logp(probs, i, P.DSS) > min_logp_dss:
                    if _debug_log_scan(i):
                        print("  DSS", i, _event_prob(probs, i, P.DSS), _event_logp(probs, i, P.DSS))
                    dss.append(i)
            else:
                dss.append(i)
        if di in ConventionalAcceptorDinucleotides:
            if probs is not None and min_logp_ass is not None:
                if _event_logp(probs, i, P.ASS) > min_logp_ass:
                    if _debug_log_scan(i):
                        print("  ASS", i, _event_prob(probs, i, P.ASS), _event_logp(probs, i, P.ASS))
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

        # boundary_score: ONLY START and STOP contributions
        boundary_score = 0.0
        for pos in self.events.get("start", []):
            event_score = _event_logp(probs, pos, P.START, False)
            boundary_score += event_score
            ordered_log.append((pos, "start", event_score))
        for pos in self.events.get("stop", []):
            event_score = _event_logp(probs, pos, P.STOP, False)
            boundary_score += event_score
            ordered_log.append((pos, "stop", event_score))

        # transition_score: DSS/ASS positives + penalties for skipped DSS/ASS
        transition_score = 0.0
        for pos in self.events.get("dss", []):
            event_score = _event_logp(probs, pos, P.DSS, False)
            transition_score += event_score
            ordered_log.append((pos, "dss+", event_score))
        for pos in self.events.get("ass", []):
            event_score = _event_logp(probs, pos, P.ASS, False)
            transition_score += event_score
            ordered_log.append((pos, "ass+", event_score))
        for idx in self.skips.get("dss", []):
            event_score = _event_logp(probs, idx, P.DSS, True)
            transition_score += event_score
            ordered_log.append((idx, "dss-", event_score))
        for idx in self.skips.get("ass", []):
            event_score = _event_logp(probs, idx, P.ASS, True)
            transition_score += event_score
            ordered_log.append((idx, "ass-", event_score))

        if verbose:
            for pos, event, score in sorted(ordered_log, key=lambda t: t[0]):
                print(self.id, event, "@", pos, "+=", score)
            print(self.id, "boundary", boundary_score, "transition", transition_score)

        return boundary_score, transition_score

    def create_candidate_gene(self, probs: np.ndarray) -> CandidateGene:
        boundary, transition = self.compute_scores(probs, verbose=_debug_log_predicted())
        return CandidateGene(
            exons=self.compute_exons(),
            events=self.events_copy(),
            boundary_score=boundary,
            transition_score=transition,
            codon_logp=None,
        )


def _decode_from_start(ps: PredictedSequence, start_pos: int, events: Dict[str, List[int]],
                      top_k_splicing: int = 3, beam_size: int = 16) -> List[CandidateGene]:
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
        _, transition = b.compute_scores(probs)
        return transition

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
                        if _debug_log_dp(j):
                            print("dss take", j, "phase", (phase + (j - i)) % 3)
                            results[-1].compute_scores(probs, verbose=True)

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
                        if _debug_log_dp(j):
                            print("dss skip", j, "phase", (phase + (j - i)) % 3)
                            results[-1].compute_scores(probs, verbose=True)

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
                        if _debug_log_dp(j):
                            print("ass take", j, "phase", phase)
                            results[-1].compute_scores(probs, verbose=True)

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
                        if _debug_log_dp(j):
                            print("ass skip", j, "phase", phase)
                            results[-1].compute_scores(probs, verbose=True)

                    break
                j += 1

        # Rank and keep top-k at this state
        results.sort(key=score_key, reverse=True)
        if len(results) > top_k_splicing:
            results = results[:top_k_splicing]
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

    # Sort by transition_score and return top_k_splicing
    candidates.sort(key=lambda c: c.transition_score, reverse=True)
    return candidates[:top_k_splicing]


def decode_sequence(ps: PredictedSequence, dss_motifs: List[str], top_k_splicing: int = 3, top_k_starts: int = 10, beam_size: int = 16,
                    min_logp_start: Optional[float] = None,
                    min_logp_stop: Optional[float] = None,
                    min_logp_dss: Optional[float] = None,
                    min_logp_ass: Optional[float] = None) -> DecodedResult:
    seq = ps.sequence
    min_logp_start = math.log(0.1) if min_logp_start is None else min_logp_start
    min_logp_stop = math.log(0.1) if min_logp_stop is None else min_logp_stop
    min_logp_dss = math.log(0.1) if min_logp_dss is None else min_logp_dss
    min_logp_ass = math.log(0.1) if min_logp_ass is None else min_logp_ass
    events = _scan_events(seq, dss_motifs, probs=ps.probabilities,
                          min_logp_start=min_logp_start, min_logp_stop=min_logp_stop, min_logp_dss=min_logp_dss, min_logp_ass=min_logp_ass)

    # Precompute START logp and process starts in descending order for pruning
    import heapq
    start_positions = list(events.get("start", []))
    start_logp: Dict[int, float] = {s: _event_logp(ps.probabilities, s, P.START, False) for s in start_positions}
    starts_desc: List[int] = sorted(start_positions, key=lambda s: start_logp[s], reverse=True)

    per_start: Dict[int, List[CandidateGene]] = {}
    start_best_boundary: Dict[int, float] = {}
    best_k_heap: List[float] = []  # min-heap of best per-start boundary scores

    for s in starts_desc:
        if len(best_k_heap) >= top_k_starts and start_logp[s] < best_k_heap[0]:
            break

        cands = _decode_from_start(ps, s, events, top_k_splicing=top_k_splicing, beam_size=beam_size)
        if cands:
            # order per-start candidates by boundary then transition desc
            cands.sort(key=lambda c: (c.boundary_score, c.transition_score), reverse=True)
            per_start[s] = cands
            bmax = cands[0].boundary_score
            start_best_boundary[s] = bmax
            if len(best_k_heap) < top_k_starts:
                heapq.heappush(best_k_heap, bmax)
            else:
                if bmax > best_k_heap[0]:
                    heapq.heapreplace(best_k_heap, bmax)
        else:
            per_start[s] = []

    # Select top_k_starts by max per-start boundary; include ties at cutoff (no tie-break)
    sorted_starts = sorted(start_best_boundary.items(), key=lambda kv: kv[1], reverse=True)
    chosen_starts: List[int] = []
    if sorted_starts:
        cutoff_index = min(top_k_starts, len(sorted_starts)) - 1
        threshold = sorted_starts[cutoff_index][1]
        for s, b in sorted_starts:
            if b >= threshold:
                chosen_starts.append(s)

    # Build global list from chosen starts
    global_candidates: List[CandidateGene] = []
    for s in chosen_starts:
        global_candidates.extend(per_start.get(s, []))

    # Always allow overlap: order by boundary then transition, both descending
    global_candidates.sort(key=lambda c: (c.boundary_score, c.transition_score), reverse=True)
    global_top = global_candidates

    return DecodedResult(sequence_index=ps.sequence_index, per_start=per_start, global_topk=global_top)



"""Gene decoder package for decoding gene structures from per-base class probabilities.

Public API:
- PredictedSequence, CandidateGene, DecodedResult (DTOs)
- decoder, codon_usage modules
"""

from dataclasses import dataclass
from typing import List, Tuple, Union, Dict, Optional
import numpy as np


@dataclass
class PredictedSequence:
    sequence_index: Union[int, str]
    sequence: str
    probabilities: np.ndarray
    class_order: List[str]


@dataclass
class CandidateGene:
    exons: List[Tuple[int, int]]
    events: Dict[str, List[int]]
    boundary_score: float
    transition_score: float
    codon_logp: Optional[float]


@dataclass
class DecodedResult:
    sequence_index: Union[int, str]
    per_start: Dict[int, List[CandidateGene]]
    global_topk: List[CandidateGene]


__all__ = [
    "PredictedSequence",
    "CandidateGene",
    "DecodedResult",
    "decoder",
    "codon_usage",
]


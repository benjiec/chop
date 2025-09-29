from dataclasses import dataclass
from typing import List, Tuple, Union, Dict, Optional
import numpy as np


@dataclass
class PredictedSequence:
    """Container for decoder inputs for a single sequence.

    - sequence_index: index or id for the sequence (opaque)
    - sequence: DNA bases string (A/T/G/C/N) in 5'->3' orientation for the decoded strand
    - probabilities: array of shape [L, C] where C follows utils.constants.GenePredictionClass order
    - class_order: list of class names by index to ensure compatibility across versions
    """
    sequence_index: Union[int, str]
    sequence: str
    probabilities: np.ndarray
    class_order: List[str]


@dataclass
class CandidateGene:
    """Decoded gene candidate.

    Exons are 0-based half-open genomic coordinates [start, end).
    Events record chosen boundary indices (0-based, event starts).
    Scores are natural-log probabilities.
    """
    exons: List[Tuple[int, int]]
    events: Dict[str, List[int]]
    boundary_logp: float
    codon_logp: Optional[float]
    total: float


@dataclass
class DecodedResult:
    sequence_index: Union[int, str]
    per_start: Dict[int, List[CandidateGene]]
    global_topk: List[CandidateGene]



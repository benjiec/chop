from typing import Dict, Iterable, Optional
from dataclasses import dataclass
import json
import math
import numpy as np


CODONS = [a + b + c for a in "ATGC" for b in "ATGC" for c in "ATGC"]


def _compute_probs(counts: Dict[str, float], alpha: float = 1.0) -> Dict[str, float]:
    total = 0.0
    for codon in CODONS:
        total += counts.get(codon, 0.0) + alpha
    probabilities: Dict[str, float] = {}
    for codon in CODONS:
        num = counts.get(codon, 0.0) + alpha
        probabilities[codon] = float(num / total)
    return probabilities


@dataclass
class CodonUsageModel:
    probabilities: Dict[str, float]

    def rare_codon_penalty(self, cds: str, percentile: Optional[int] = 5, penalty_scale: Optional[float] = 0.1) -> float:
        """
	Returns logp(avg) of bottom percentiles of codon usage. Lower the
        average prob, the more negative the penalty is.
        """

        s = 0.0
        L = len(cds)
        probs = []
        for i in range(0, L - (L % 3), 3):
            codon = cds[i:i+3]
            if 'N' not in codon:
                probs.append(self.probabilities[codon])

        probs.sort()
        assert percentile <= 100 and percentile >= 0
        perc = max(1, len(probs) * percentile // 100)

        probs = probs[:perc]
        return math.log(np.mean(probs)) * penalty_scale

    def to_json(self, path: str) -> None:
        with open(path, 'w') as f:
            json.dump(self.probabilities, f)

    @staticmethod
    def from_json(path: str) -> "CodonUsageModel":
        with open(path, 'r') as f:
            data = json.load(f)
        return CodonUsageModel(probabilities={str(k): float(v) for k, v in data.items()})


def build_codon_usage_from_cds(cds_iter: Iterable[str], alpha: float = 1.0) -> CodonUsageModel:
    counts: Dict[str, float] = {}
    for cds in cds_iter:
        L = len(cds)
        for i in range(0, L - (L % 3), 3):
            codon = cds[i:i+3]
            if len(codon) == 3 and all(ch in 'ATGC' for ch in codon):
                counts[codon] = counts.get(codon, 0.0) + 1.0
    probs = _compute_probs(counts, alpha=alpha)
    return CodonUsageModel(probabilities=probs)

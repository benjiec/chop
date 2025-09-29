from typing import Dict, Iterable, Optional
from dataclasses import dataclass
import json
import math


CODONS = [a + b + c for a in "ATGC" for b in "ATGC" for c in "ATGC"]


def _normalize_logprobs(counts: Dict[str, float], alpha: float = 1.0) -> Dict[str, float]:
    total = 0.0
    for codon in CODONS:
        total += counts.get(codon, 0.0) + alpha
    logps: Dict[str, float] = {}
    for codon in CODONS:
        num = counts.get(codon, 0.0) + alpha
        logps[codon] = math.log(num / total)
    return logps


@dataclass
class CodonUsageModel:
    logp: Dict[str, float]

    def score(self, cds: str) -> float:
        s = 0.0
        L = len(cds)
        for i in range(0, L - (L % 3), 3):
            codon = cds[i:i+3]
            s += self.logp.get(codon, float('-inf'))
        return s

    def to_json(self, path: str) -> None:
        with open(path, 'w') as f:
            json.dump(self.logp, f)

    @staticmethod
    def from_json(path: str) -> "CodonUsageModel":
        with open(path, 'r') as f:
            data = json.load(f)
        return CodonUsageModel(logp={str(k): float(v) for k, v in data.items()})


def build_codon_usage_from_cds(cds_iter: Iterable[str], alpha: float = 1.0) -> CodonUsageModel:
    counts: Dict[str, float] = {}
    for cds in cds_iter:
        L = len(cds)
        for i in range(0, L - (L % 3), 3):
            codon = cds[i:i+3]
            if len(codon) == 3 and all(ch in 'ATGC' for ch in codon):
                counts[codon] = counts.get(codon, 0.0) + 1.0
    logp = _normalize_logprobs(counts, alpha=alpha)
    return CodonUsageModel(logp=logp)



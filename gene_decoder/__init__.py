"""Gene decoder package for decoding gene structures from per-base class probabilities.

Modules:
- types: Data transfer objects used by decoder CLI and exporter
- decoder: Phase-aware k-best decoder enforcing gene structure constraints
- codon_usage: Utilities to build and score codon usage models
- decode: CLI entry point to run the decoder
- build_codon_usage: CLI to compute codon usage JSON from training data
"""

__all__ = [
    "types",
    "decoder",
    "codon_usage",
]



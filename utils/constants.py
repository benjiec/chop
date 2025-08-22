"""
Constants for gene prediction model.

This module defines all the class indices and other constants used throughout
the gene prediction pipeline to avoid magic numbers and ensure consistency.
"""

# Gene boundary class indices
class GeneBoundaryClass:
    NO_GENE = 0
    START = 1
    END = 2

# Exon/intron class indices  
class ExonIntronClass:
    INTERGENIC = 0
    EXON = 1
    INTRON = 2

# Splice site class indices
class SpliceSiteClass:
    NO_SPLICE = 0
    DONOR = 1
    ACCEPTOR = 2

# DNA vocabulary
DNA_VOCAB = {
    'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4
}

# Model architecture constants
DEFAULT_VOCAB_SIZE = 5
DEFAULT_MAX_SEQ_LENGTH = 8192
DEFAULT_D_MODEL = 512
DEFAULT_N_LAYERS = 6
DEFAULT_N_HEADS = 8
DEFAULT_DROPOUT = 0.1

# Training constants
DEFAULT_THRESHOLD = 0.5

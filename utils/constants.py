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

# Data loading constants
DEFAULT_WINDOW_SIZE = 12288
DEFAULT_STRIDE = 6144
DEFAULT_MIN_GENE_COVERAGE = 0.5
DEFAULT_CACHE_DIR = "cache"
DEFAULT_MAX_CACHE_SIZE_GB = 5

# TSV format constants
TSV_COLUMNS = ['sequence_id', 'gene_id', 'gene_start', 'gene_end', 'exon_start', 'exon_end', 'strand']
TSV_HEADER = '\t'.join(TSV_COLUMNS)

# Gene structure constants (for annotation processing only)
MIN_GENE_LENGTH = 300      # Minimum 100 codons
MIN_EXON_LENGTH = 1        # Allow micro-exons (biologically valid)
MAX_INTRON_LENGTH = 50000  # Maximum reasonable intron for algae
MAX_GENE_EXONS = 500       # Maximum exons per gene (safety limit)

# Gene ID tracking constants
UNKNOWN_GENE_ID = -1       # For intergenic regions
MAX_GENES_PER_WINDOW = 100 # Maximum genes that can overlap in one window

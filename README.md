# CHOP: Chromosome Open Prediction

A transformer-based gene prediction tool designed for complex genomic sequences, optimized for Symbiodinium microadriaticum and other organisms with intricate gene structures.

## Overview

CHOP uses attention mechanisms to identify genes, exons, introns, splice sites, and coding regions in genomic DNA sequences. The system is specifically designed to handle extraordinarily complex gene structures found in Symbiodinium (dinoflagellate) species, featuring:

- **99.7% multi-exon genes** with average 40.7 exons per gene
- **Sliding window processing** for genes up to 24kb+ 
- **Gene ID tracking** for overlapping gene structures
- **Memory-optimized** for 24GB RAM systems

## Quick Start

### 1. Setup Environment

```bash
# Create and activate virtual environment
python -m venv chop_env
source chop_env/bin/activate  # On macOS/Linux
# or: chop_env\Scripts\activate  # On Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Data

Convert your GFF annotations to the standardized TSV format:

```bash
# For Symbiodinium data
python scripts/parse_symbiodinium_gff.py

# For other organisms
python scripts/gff_to_tsv.py input.gff output.tsv --verbose
```

This creates a TSV file with columns: `sequence_id, gene_id, gene_start, gene_end, exon_start, exon_end, strand`

### 3. Train Model

```bash
python train_cpu_only.py --config configs/m2_cpu_model7M_layer4_head6_seqlen12k_sliding.yaml
```

### 4. Make Predictions

```bash
python inference/predict.py --model models/checkpoints/best_model.pt --config configs/your_config.yaml --input your_sequences.fna
```

## Configuration

The system is driven by YAML configuration files that specify:

- **Model architecture**: Layers, attention heads, sequence length
- **Sliding windows**: Window size and stride for long gene coverage
- **Data processing**: Validation, caching, augmentation settings
- **Training parameters**: Batch size, learning rate, optimization settings

Example configuration:
```yaml
model:
  max_seq_length: 12288  # 12k windows
  d_model: 256
  n_layers: 4
  n_heads: 6

data:
  sequences_path: "data/sequences.fna"
  tsv_annotations_path: "data/annotations.tsv"
  use_sliding_windows: true
  window_size: 12288
  stride: 6144  # 50% overlap
```

## Model Architecture

The transformer architecture is specifically designed for biological pattern recognition:

- **4 layers**: Hierarchical learning from local motifs to gene architecture
- **6 attention heads**: Capture diverse biological patterns (splice sites, codons, boundaries)
- **12k sequence length**: Optimal balance of gene coverage and memory usage

For detailed model design decisions, see `notes/symbiodinium-model-design.txt`.

## Data Format

CHOP uses a standardized TSV format for annotations:

| Column | Description | Example |
|--------|-------------|---------|
| sequence_id | FASTA sequence identifier | LSRX01000005.1 |
| gene_id | Unique gene identifier | rna-gnl\|WGS:LSRX\|Smic502_mrna |
| gene_start | Gene start position (0-based) | 491426 |
| gene_end | Gene end position | 625329 |
| exon_start | Exon start position (0-based) | 491426 |
| exon_end | Exon end position | 492092 |
| strand | Strand orientation | + |

## Testing

Run the regression test suite:

```bash
python tests/run_tests.py
```

This validates:
- GFF→TSV conversion accuracy
- Sliding window annotation mapping
- Gene ID tracking for overlapping genes
- Data validation and error handling

## Advanced Usage

### Custom Data Augmentation
Enable data augmentation in your config:
```yaml
data:
  augmentation:
    enable: true
    reverse_complement_prob: 0.5
    masking_prob: 0.1
    max_mask_length: 50
```

### Memory Management
For systems with different RAM:
```yaml
optimization:
  max_memory_usage_gb: 19  # Adjust for your system
  gradient_checkpointing: true
  empty_cache_frequency: 50
```

### Custom Validation
Adjust validation thresholds:
```yaml
data:
  validate_sequences: true
  min_gene_coverage: 0.5  # Minimum gene coverage in windows
```

## Troubleshooting

### Common Issues

**Memory errors**: Reduce `max_seq_length` or `batch_size` in config
**No annotations found**: Ensure sequence IDs in FASTA match TSV `sequence_id` column
**Poor predictions**: Verify TSV format and gene structure validation
**Slow loading**: Enable caching with `use_cache: true`

### Getting Help

1. Check `notes/cursor-context.txt` for project overview
2. Review `notes/symbiodinium-model-design.txt` for architecture decisions
3. Run `python scripts/example_data_loading.py` for usage examples
4. Examine test cases in `tests/` for implementation details


---

**Note**: This system was specifically optimized for Symbiodinium microadriaticum genome data but can be adapted for other organisms with complex gene structures.

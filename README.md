# CHOP: Chromosome Open Prediction

A transformer-based gene prediction tool designed for complex genomic sequences, optimized for organisms with intricate gene structures.

## Overview

CHOP uses attention mechanisms to identify genes, exons, introns, and coding regions in genomic DNA sequences. The system is designed to handle complex gene structures featuring:

- **Multi-exon genes** with complex structures
- **Sliding window processing** for long sequences
- **Gene ID tracking** for overlapping gene structures  
- **Memory-optimized** for efficient training and inference

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

Convert your raw genomic data (FASTA + GFF) to gene context format:

```bash
python scripts/preprocess_gene_data.py \
    --fasta raw_genome.fna \
    --gff annotations.gff \
    --output-fasta gene_contexts.fna \
    --output-tsv gene_annotations.tsv \
    --flanking-bp 2000
```

This preprocessing script:
- Extracts gene ± 2kb contexts from full genomic sequences
- Normalizes all genes to + strand orientation
- Adjusts coordinates to gene context coordinate system
- Validates start/stop codons and filters invalid genes

### 3. Train Model

```bash
# CPU training
python train_cpu_only.py --config configs/m4_cpu_model7M_layer4_head6_seqlen12k_sliding.yaml

# MPS (Apple Silicon) training
python train_mps.py --config configs/m4_mps_model7M_layer4_head6_seqlen12k_sliding.yaml
```

### 4. Make Predictions

```bash
python inference/predict.py \
    --model models/checkpoints/final_model.pt \
    --config configs/m4_mps_longer.yaml \
    --input test_sequences.fna
```

## Configuration

The system uses YAML configuration files that specify:

- **Model architecture**: Layers, attention heads, sequence length
- **Sliding windows**: Window size and stride for long sequence coverage
- **Data processing**: Validation and caching settings
- **Training parameters**: Batch size, learning rate, optimization

Example configuration:
```yaml
model:
  max_seq_length: 12288  # 12k windows
  d_model: 256
  n_layers: 4
  n_heads: 6

data:
  sequences_path: "data/gene_contexts.fna"
  tsv_annotations_path: "data/gene_annotations.tsv"
  use_sliding_windows: true
  window_size: 12288
  stride: 6144  # 50% overlap
  validate_normalization: true

training:
  batch_size: 4
  learning_rate: 1e-4
  accumulate_grad_batches: 16
```

## Model Architecture

The transformer architecture is designed for biological pattern recognition:

- **4 layers**: Hierarchical learning from local motifs to gene architecture
- **6 attention heads**: Capture diverse biological patterns 
- **12k sequence length**: Optimal balance of gene coverage and memory usage

For detailed design decisions, see `notes/symbiodinium-model-design.txt`.

## Data Pipeline

CHOP uses a two-stage data pipeline:

### 1. Preprocessing Stage
```bash
scripts/preprocess_gene_data.py
```
- **Input**: Raw genomic FASTA + GFF annotations
- **Output**: Normalized gene contexts + adjusted annotations
- **Features**: Strand normalization, coordinate adjustment, codon validation

### 2. Training Stage
```bash
training/train.py
```
- **Input**: Preprocessed gene contexts + annotations
- **Features**: Data validation, sliding windows, caching

## Data Format

Gene context sequences use FASTA format with gene IDs as sequence names:
```
>gene_id_1
ATGAAATAAG...
>gene_id_2
ATGCCCGGGTAG...
```

Annotations use TSV format:
| Column | Description | Example |
|--------|-------------|---------|
| sequence_id | Gene identifier | gene_id_1 |
| gene_id | Gene identifier | gene_id_1 |
| gene_start | Gene start (0-based) | 2000 |
| gene_end | Gene end | 2009 |
| exon_start | Exon start (0-based) | 2000 |
| exon_end | Exon end | 2009 |
| strand | Always '+' (normalized) | + |

## Testing

Run the test suite:

```bash
python tests/run_tests.py
```

This validates:
- GFF parsing and conversion
- Gene context extraction with strand normalization
- Coordinate transformations
- Data loader validation
- End-to-end preprocessing pipeline

## Scripts

### Core Scripts
- `scripts/preprocess_gene_data.py` - Main preprocessing pipeline
- `scripts/gff_to_tsv.py` - GFF format conversion utilities
- `training/train.py` - Model training
- `inference/predict.py` - Gene prediction

### Training Scripts
- `train_cpu_only.py` - CPU-only training wrapper
- `train_mps.py` - Apple Silicon MPS training wrapper

## Advanced Usage

### Memory Management
For different system configurations:
```yaml
training:
  batch_size: 2          # Reduce for less RAM
  accumulate_grad_batches: 32  # Maintain effective batch size
  
data:
  max_sequence_length: 8192    # Reduce for memory constraints
  window_size: 8192
  pin_memory: false      # Disable for MPS training
```

### Biological Constraints
CHOP enforces biological accuracy:
```yaml
loss:
  enforce_start_stop_codons: true  # Penalize invalid start/stop codons
```

## Troubleshooting

### Common Issues

**Memory errors**: Reduce `max_seq_length` or `batch_size`
**Data loading errors**: Ensure preprocessing completed successfully
**Poor predictions**: Verify gene context extraction and validation
**Training instability**: Increase `accumulate_grad_batches` for stable gradients

### Getting Help

1. Check `notes/cursor-context.txt` for project overview
2. Review `notes/symbiodinium-model-design.txt` for architecture decisions  
3. Examine test cases in `tests/` for implementation examples
4. Run preprocessing with `--help` for parameter details

---

**Note**: CHOP has been optimized for complex gene structures but can be adapted for various organisms by adjusting the preprocessing parameters.
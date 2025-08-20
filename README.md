# CHOP: Chromosome Open Prediction

A transformer-based gene prediction tool that uses attention mechanisms to identify genes in genomic DNA sequences.

## Project Overview

This tool aims to predict gene structures (exons, introns, start/stop codons) using modern deep learning approaches, specifically transformer models with biological constraints.

## Features

- **Multi-task Learning**: Predicts gene boundaries, exon-intron structure, and splice sites
- **Biological Constraints**: Incorporates domain knowledge about codon usage, splice motifs, and length distributions
- **Transformer Architecture**: Uses attention mechanisms to capture long-range dependencies in DNA sequences
- **Flexible Input**: Accepts various DNA sequence formats and genome assemblies

## Project Structure

```
chop/
├── data/               # Training and test data
├── models/             # Model architectures and weights
├── training/           # Training scripts and utilities
├── inference/          # Prediction and evaluation scripts
├── utils/              # Helper functions and data processing
├── configs/            # Configuration files
└── notebooks/          # Jupyter notebooks for exploration
```

## Getting Started

1. Install dependencies: `pip install -r requirements.txt`
2. Download training data (see data/README.md)
3. Train the model: `python training/train.py --config configs/default.yaml`
4. Make predictions: `python inference/predict.py --model models/best.pt --sequence input.fasta`

## Dependencies

- PyTorch
- Transformers (Hugging Face)
- BioPython
- PyTorch Lightning
- NumPy, Pandas
- Matplotlib, Seaborn

## Current Status

🚧 **Under Development** - This is an active research project.

# CHOP Quick Start Guide

## 🚀 Getting Started

### 1. Installation
```bash
# Install dependencies
pip install -r requirements.txt

# Or use Docker
docker-compose build
```

### 2. Basic Usage

#### Training a Model
```bash
# Train with default configuration
python training/train.py --config configs/default.yaml

# With Docker
docker-compose run --rm chop python training/train.py --config configs/default.yaml
```

#### Making Predictions
```bash
# Predict on a single sequence
python inference/predict.py \
  --model models/best.pt \
  --config configs/default.yaml \
  --input "ATGAAACGCATTAGCAACCCCGATCGATCGAACGCTACGATCGATAA" \
  --output results/ \
  --visualize

# Predict on a FASTA file
python inference/predict.py \
  --model models/best.pt \
  --config configs/default.yaml \
  --input sequences.fasta \
  --output results/ \
  --visualize
```

### 3. Data Preparation

Your training data should include:
- **DNA sequences**: FASTA format (`data/sequences.fasta`)
- **Gene annotations**: GFF format (`data/annotations.gff`)

Update the paths in `configs/default.yaml`:
```yaml
data:
  sequences_path: "data/sequences.fasta"
  annotations_path: "data/annotations.gff"
```

### 4. Key Configuration Options

Edit `configs/default.yaml` to customize:

```yaml
model:
  vocab_size: 5          # A, C, G, T, N
  d_model: 512          # Model dimension
  n_layers: 6           # Transformer layers
  max_seq_length: 8192  # Maximum sequence length

training:
  batch_size: 8
  learning_rate: 1e-4
  max_epochs: 100
```

### 5. Splice Sites Configuration

Splice sites are defined in `utils/dna_processor.py`:

```python
# In DNATokenizer class (lines 30-32)
self.donor_motifs = ['GT', 'GC']      # 5' splice site
self.acceptor_motifs = ['AG', 'AC']   # 3' splice site
```

To customize, modify these lists or extend the `DNATokenizer` class.

### 6. Understanding Results

Predictions include:
- **Gene boundaries**: Start/end positions
- **Exon/intron structure**: Coding vs non-coding regions  
- **Splice sites**: Donor (5') and acceptor (3') sites
- **Coding potential**: Probability of coding regions

Example output:
```json
{
  "genes": [
    {"start": 0, "end": 30, "length": 30}
  ],
  "exons": [
    {"start": 0, "end": 15, "length": 16}
  ],
  "splice_sites": {
    "donor_sites": [16],
    "acceptor_sites": [19]
  }
}
```

### 7. Common Commands

```bash
# Run the usage examples
python example_usage.py

# Check model configuration
python -c "import yaml; print(yaml.safe_load(open('configs/default.yaml')))"

# Test tokenization
python -c "
from utils.dna_processor import DNATokenizer
tokenizer = DNATokenizer()
seq = 'ATGAAACGCATTAGC'
print('Tokens:', tokenizer.tokenize(seq))
print('Start codons:', tokenizer.find_start_codons(seq))
"
```

### 8. Troubleshooting

- **CUDA errors**: Set `accelerator: "cpu"` in config for CPU-only training
- **Memory issues**: Reduce `batch_size` or `max_seq_length`
- **Import errors**: Make sure you're in the project root directory

### 9. Next Steps

1. Prepare your genomic data in FASTA/GFF format
2. Adjust model parameters in the config file  
3. Train your model on your specific dataset
4. Evaluate predictions and fine-tune as needed

For detailed examples, see `example_usage.py`!

# Detecting START and STOP codons

This directory contains example code for training a transformer model to detect
START and STOP codons.

## Analyzing Predictions

```
python gene_boundary/predict_and_analyze.py \
  --model-path gene_boundary/boundary_run_<timestamp>/checkpoints/last.ckpt \
  --output-dir gene_boundary/boundary_run_<timestamp> \
  --num-sequences 30
```

Prediction results appear under the test run directory
- `prediction_<timestamp>_<hash>_input.fa` - Input sequences used for analysis in FASTA format
- `prediction_<timestamp>_<hash>_report.txt` - Visual sequence context showing TP/FP/FN predictions with 60bp upstream and 20bp downstream context
- `prediction_<timestamp>_<hash>_attn.fa` - Attention sequence fragments showing where head attentions are


## Running Trainings

```
# Train with asymmetric attention masks
python gene_boundary/train.py --class-weights --start-weight 5 --kmer 0 --attention-masks "0:4,1:20:5,2:50:0" --layers 4
```

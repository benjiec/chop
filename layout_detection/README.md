# Detecting START codons downstream of UTRs

This directory contains tools for detecting START codons in genomic sequences
with upstream UTR context.


## Analyzing Predictions

```
python layout_detection/predict_and_analyze.py \
  --model-path layout_detection/utr_start_test_run_<timestamp>/checkpoints/last.ckpt \
  --output-dir layout_detection/utr_start_test_run_<timestamp> \
  --num-sequences 30
```

Prediction results appear under the test run directory
- `prediction_<timestamp>_input.fa` - Input sequences used for analysis in FASTA format
- `prediction_<timestamp>_report.txt` - Visual sequence context showing TP/FP/FN predictions with 60bp upstream and 20bp downstream context

You can then analyze which sequences are in the FP or FN categories, using the following

```
python layout_detection/breakdown_prediction_report.py \
    --report-file layout_detection/utr_start_test_run_<timestamp>/prediction_<timestamp_and_hash>.txt
```

## Running Trainings

```
# Train with asymmetric attention masks
python layout_detection/train.py --class-weights --start-weight 5 --kmer 0 --attention-masks "0:4,1:20:5,2:50:0" --layers 4
```

After training completes you will want to evaluate the model by running
prediction, like above, and also analyze the attention weights with respect to
the START position (note that this is NOT a general attention weight analysis;
all weight distribution is relative to the START position).

```
# Extract attention ranges to TSV format
python layout_detection/generate_attention_summary_visual.py --run-dir layout_detection/utr_start_test_run_<timestamp>
```

Use the attention_dynamics.twb Tableau workbook to visualize the attention
weights: edit the data connection to the output of the above command
`layout_detection/utr_start_test_run_<timestamp>/attention_ranges.tsv`.


## Training Output Files

Training runs generate comprehensive analysis data in timestamped directories
(e.g., `utr_start_test_run_20250901_150439/`):

### Model Checkpoints (`checkpoints/`)
- `last.ckpt` - Final model checkpoint
- `utr_start_model_epoch=X_val_loss=Y.ckpt` - Best validation checkpoints
- Used for model inference and continued training

### Training Dynamics (`training_dynamics/`)
- `training_dynamics.json` - Epoch-by-epoch attention evolution data [START-SPECIFIC]
  - How attention patterns changed during training (epochs 0, 5, 10, 15, 20, 24)
  - Layer specialization trends over time measured from START positions
  - START prediction accuracy progression
  - Individual head focus evolution (upstream/local/downstream relative to START codons)
  - ~100MB, contains temporal analysis data

- `training_dynamics_summary.json` - Condensed training trends [START-SPECIFIC]
  - Overall learning insights and layer specialization patterns for START detection
  - Key performance metrics across epochs

### Layer Analysis (`layer_analysis/`)
- `attention_weights.json` - Detailed final model attention patterns (4.7GB) [START-SPECIFIC]
  - Individual sample attention breakdowns from START positions to all other positions
  - Head-by-head attention weights and matrices queried from START codons
  - Top attended positions for each head when processing START positions
  - Upstream/local/downstream attention scores relative to START codons
  - Essential for spatial attention analysis (e.g., "Head 2 focuses 20-50 bases upstream of START")

- `analysis_summary.json` - High-level attention statistics [START-SPECIFIC]
  - Average importance scores across regions for START prediction
  - START prediction accuracy summary

- `layer_features.json` - Layer activation patterns [START-SPECIFIC]
  - Mean/max/min activations per layer at START positions
  - Feature evolution at START positions

- `gradient_attribution.json` - Attribution analysis [START-SPECIFIC]
  - Gradient-based feature importance for START prediction
  - Position-wise attribution scores relative to START positions

- `combined_analysis.png` - Visualization of attention patterns [START-SPECIFIC]
  - Multi-panel plot showing attention evolution and patterns for START detection

### Lightning Logs (`lightning_logs/`)
- TensorBoard event files for training visualization
- Hyperparameters and training configuration

# Detecting START codons downstream of UTRs

This directory contains tools for detecting START codons in genomic sequences
with upstream UTR context.


## Analyzing Predictions

```
python layout_detection/analyze_fresh_start_predictions.py --model-path layout_detection/utr_start_test_run_20250901_150439/checkpoints/last.ckpt --output-dir layout_detection/utr_start_test_run_20250901_150439 --num-sequences 30
```

Prediction results appear under the test run directory
- `prediction_<timestamp>_input.fa` - Input sequences used for analysis in FASTA format
- `prediction_<timestamp>_report.txt` - Visual sequence context showing TP/FP/FN predictions with 60bp upstream and 20bp downstream context


## Running Trainings

```
# try to detect START codons that are after UTRs
python layout_detection/test_utr_start_controlled.py --class-weights --class-weights --start-weight 5 --kmer 0 --attention-masks "0:4,1:20:5,2:50:0" --layers 4

# to test target generation code
python layout_detection/test_target_generation.py

# Test on 100 new contigs and analyze data
python layout_detection/analyze_saved_model.py --model-path your_model.ckpt --contigs 100

# Test on 50 contigs with different layout patterns
python layout_detection/analyze_saved_model.py --model-path your_model.ckpt --contigs 50 --layouts 2

# Quick analysis with fewer samples
python layout_detection/analyze_saved_model.py --model-path your_model.ckpt --max-samples 5
```


## Training Output Files

Training runs generate comprehensive analysis data in timestamped directories (e.g., `utr_start_test_run_20250901_150439/`):

### Model Checkpoints (`checkpoints/`)
- `last.ckpt` - Final model checkpoint
- `utr_start_model_epoch=X_val_loss=Y.ckpt` - Best validation checkpoints
- Used for model inference and continued training

### Training Dynamics (`training_dynamics/`)
- `training_dynamics.json` - Epoch-by-epoch attention evolution data **[START-SPECIFIC]**
  - How attention patterns changed during training (epochs 0, 5, 10, 15, 20, 24)
  - Layer specialization trends over time **measured from START positions**
  - START prediction accuracy progression
  - Individual head focus evolution (upstream/local/downstream **relative to START codons**)
  - ~100MB, contains temporal analysis data

- `training_dynamics_summary.json` - Condensed training trends **[START-SPECIFIC]**
  - Overall learning insights and layer specialization patterns **for START detection**
  - Key performance metrics across epochs

### Layer Analysis (`layer_analysis/`)
- `attention_weights.json` - Detailed final model attention patterns (4.7GB) **[START-SPECIFIC]**
  - Individual sample attention breakdowns **from START positions to all other positions**
  - Head-by-head attention weights and matrices **queried from START codons**
  - Top attended positions for each head **when processing START positions**
  - Upstream/local/downstream attention scores **relative to START codons**
  - Essential for spatial attention analysis (e.g., "Head 2 focuses 20-50 bases upstream of START")

- `analysis_summary.json` - High-level attention statistics **[START-SPECIFIC]**
  - Average importance scores across regions **for START prediction**
  - START prediction accuracy summary

- `layer_features.json` - Layer activation patterns **[START-SPECIFIC]**
  - Mean/max/min activations per layer **at START positions**
  - Feature evolution at START positions

- `gradient_attribution.json` - Attribution analysis **[START-SPECIFIC]**
  - Gradient-based feature importance **for START prediction**
  - Position-wise attribution scores **relative to START positions**

- `combined_analysis.png` - Visualization of attention patterns **[START-SPECIFIC]**
  - Multi-panel plot showing attention evolution and patterns **for START detection**

### Lightning Logs (`lightning_logs/`)
- TensorBoard event files for training visualization
- Hyperparameters and training configuration


## Attention Analysis Tools

### Generate Attention Range Data

After training runs complete, extract attention patterns into structured data:

```
# Extract attention ranges to TSV format
python layout_detection/generate_attention_summary_visual.py --run-dir layout_detection/utr_start_test_run_YYYYMMDD_HHMMSS
```

**All attention analysis data is START-position specific**. The analysis
measures attention patterns **from START codon positions** to other sequence
positions. This means:

- What it shows: How the model processes information when making START codon
  predictions
- What it doesn't show: Attention patterns for UTR prediction, exon/intron
  classification, or other tasks
- Task dependency: The same model heads would likely show different attention
  patterns when analyzed from UTR positions or other genomic features
- Architecture insights: The finding that "all heads focus locally" applies
  specifically to START detection - UTR or exon prediction might require and
  utilize the broader attention ranges

For comprehensive model understanding, similar analysis would need to be
conducted from UTR positions, exon boundaries, and other relevant genomic
features.

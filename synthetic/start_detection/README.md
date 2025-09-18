# Detecting START codons downstream of UTRs

This directory contains example code for training a transformer model to detect
START codons.

## Analyzing Predictions

```
python synthetic/start_detection/predict_and_analyze.py \
  --model-path synthetic/start_detection/utr_start_test_run_<timestamp>/checkpoints/last.ckpt \
  --output-dir synthetic/start_detection/utr_start_test_run_<timestamp> \
  --num-sequences 30
```

Prediction results appear under the test run directory
- `prediction_<timestamp>_<hash>_input.fa` - Input sequences used for analysis in FASTA format
- `prediction_<timestamp>_<hash>_report.txt` - Visual sequence context showing TP/FP/FN predictions with 60bp upstream and 20bp downstream context
- `prediction_<timestamp>_<hash>_attn.fa` - Attention sequence fragments showing where head attentions are


## Running Trainings

```
# Train with asymmetric attention masks
python synthetic/start_detection/train.py --class-weights --start-weight 5 --kmer 0 --attention-masks "0:4,1:20:5,2:50:0" --layers 4
```

After training completes you can evaluate the model by running prediction, like
above, and also analyze the attention weights with respect to the START
position (note that this is NOT a general attention weight analysis; all weight
distribution is relative to the START position).

```
# Extract attention ranges to TSV format
python synthetic/start_detection/summarize_attention_weights.py --run-dir synthetic/start_detection/utr_start_test_run_<timestamp>
```

Use the attention_dynamics.twb Tableau workbook to visualize the attention
weights: edit the data connection to the output of the above command
`synthetic/start_detection/utr_start_test_run_<timestamp>/attention_ranges.tsv`.

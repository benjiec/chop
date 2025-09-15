#!/usr/bin/env python3
"""
START prediction analysis with new test data.

Creates new test sequences, runs predictions, and analyzes:
- True positives vs false positives with sensitivity/precision/specificity
- Sequence context around all predictions
- Kozak patterns and biological features
- Actual codons at predicted positions
"""

import sys
from pathlib import Path
import torch
import numpy as np
import json
import argparse
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import hashlib
import re

# Add project paths
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "layout_detection"))

from utils.constants import DNAEmbed, GenePredictionClass
# Removed: parent-sequence catalogs (KOZAK/UTR5/IRES) no longer used
from gene_predictor.model import GenePredictorModule as ModelModule
from torch.utils.data import DataLoader
from layout_detection.layouts import generate_dataset


def load_trained_model(model_path: Path, device='cpu'):
    """Load the trained model from checkpoint, restoring exact architecture."""
    print(f"Loading model from: {model_path}")

    # First, try Lightning's native loader which restores saved hyperparameters
    try:
        model = ModelModule.load_from_checkpoint(model_path, map_location=device)
        model.eval()
        model = model.to(device)
        cfg = getattr(model, 'config', None)
        if cfg and 'model' in cfg:
            m = cfg['model']
            print(f"✓ Model loaded successfully (layers={m.get('n_layers')}, heads={m.get('n_heads')}, kmer={m.get('kmer_size')})")
        else:
            print("✓ Model loaded successfully")
        return model
    except Exception:
        pass  # Fallback to manual

    try:
        checkpoint = torch.load(model_path, map_location=device)

        # Extract config from checkpoint hyperparameters
        config = None
        for hp_key in ('hyper_parameters', 'hparams'):
            if hp_key in checkpoint:
                hp = checkpoint[hp_key]
                if isinstance(hp, dict):
                    if 'model' in hp and 'training' in hp:
                        config = hp
                        break
                    if 'config' in hp and isinstance(hp['config'], dict):
                        config = hp['config']
                        break

        if config is None:
            raise RuntimeError("Checkpoint does not contain saved hyperparameters/config; cannot safely reconstruct model.")

        model = ModelModule(config)

        # Load state dict, filtering out non-model parameters
        state_dict = checkpoint.get('state_dict', checkpoint)
        model_state_dict = {k: v for k, v in state_dict.items() if not k.startswith('criterion')}

        missing, unexpected = model.load_state_dict(model_state_dict, strict=False)
        if missing:
            print(f"Warning: Missing keys while loading: {sorted(missing)[:5]}{' ...' if len(missing) > 5 else ''}")
        if unexpected:
            print(f"Warning: Unexpected keys while loading: {sorted(unexpected)[:5]}{' ...' if len(unexpected) > 5 else ''}")

        model.eval()
        model = model.to(device)
        m = config.get('model', {})
        print(f"✓ Model loaded successfully (layers={m.get('n_layers')}, heads={m.get('n_heads')}, kmer={m.get('kmer_size')})")
        return model

    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def generate_test_data(num_sequences: int, max_seq_length: int, layout_version: int, layouts_per_contig: int = 1):
    """Generate fresh test data aligned to the model's max sequence length."""
    print(f"Generating {num_sequences} test sequences...")

    dataset = generate_dataset(num_sequences, max_seq_length, layout_version, layouts_per_contig=layouts_per_contig)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    print(f"✓ Generated {len(dataset)} test windows")
    return data_loader, dataset

def run_predictions(model, data_loader, device='cpu', return_attention: bool = False):
    """Run predictions on test data. Optionally return encoder attention per layer."""
    
    print("Running predictions on test data...")
    
    all_results = []
    
    with torch.no_grad():
        for batch_idx, (sequences, targets) in enumerate(data_loader):
            sequences = sequences.to(device)
            targets = targets.to(device)
            
            # Run model
            outputs = model(sequences, return_attention=return_attention)
            
            # Handle both dictionary and tensor outputs
            if isinstance(outputs, dict):
                gene_boundaries = outputs['gene_boundaries']
                layer_attn = outputs.get('attentions') if 'attentions' in outputs else None
            else:
                # Model returns tensor or (logits, attentions)
                if return_attention and isinstance(outputs, tuple) and len(outputs) == 2:
                    gene_boundaries, layer_attn = outputs
                else:
                    gene_boundaries = outputs
                    layer_attn = None
            
            predictions = torch.argmax(gene_boundaries, dim=-1)
            probabilities = torch.softmax(gene_boundaries, dim=-1)
            
            # Convert to numpy for analysis
            seq_np = sequences[0].cpu().numpy()
            target_np = targets[0].cpu().numpy()
            pred_np = predictions[0].cpu().numpy()
            prob_np = probabilities[0].cpu().numpy()
            
            result_entry = {
                'sequence_index': batch_idx,
                'sequence_tokens': seq_np,
                'targets': target_np,
                'predictions': pred_np,
                'probabilities': prob_np
            }
            if return_attention and layer_attn is not None:
                # Convert attention dict to per-layer numpy arrays for this sequence
                attn_dict = {}
                for layer_name, attn_tensor in layer_attn.items():
                    if attn_tensor is None:
                        continue
                    # attn_tensor: (batch, heads, L, L)
                    attn_dict[layer_name] = attn_tensor[0].cpu().numpy()
                result_entry['attentions'] = attn_dict

            all_results.append(result_entry)
            
            if (batch_idx + 1) % 10 == 0:
                print(f"  Processed {batch_idx + 1} sequences...")
    
    print(f"✓ Completed predictions for {len(all_results)} sequences")
    return all_results

def convert_tokens_to_sequence(tokens):
    """Convert token indices back to DNA sequence."""
    idx_to_nucleotide = DNAEmbed.idx_to_bp
    return ''.join([idx_to_nucleotide.get(int(token), 'N') for token in tokens])

def analyze_kozak_pattern(upstream: str, atg_codon: str):
    """Analyze Kozak consensus pattern."""
    
    features = {}
    score = 0
    
    if len(upstream) >= 3:
        # Position -3 (purine preferred)
        minus3 = upstream[-3]
        features['minus3_base'] = minus3
        features['minus3_purine'] = minus3 in ['A', 'G']
        if features['minus3_purine']:
            score += 2
    
    if len(upstream) >= 1:
        # Position -1
        minus1 = upstream[-1]
        features['minus1_base'] = minus1
        features['minus1_optimal'] = minus1 == 'C'
        if features['minus1_optimal']:
            score += 1
    
    # GC content in upstream region
    if len(upstream) >= 6:
        upstream_6 = upstream[-6:]
        gc_content = (upstream_6.count('G') + upstream_6.count('C')) / len(upstream_6)
        features['upstream_gc_content'] = gc_content
        features['gc_rich'] = gc_content > 0.5
        if features['gc_rich']:
            score += 1
    
    # ATG codon
    features['is_atg'] = atg_codon == 'ATG'
    if features['is_atg']:
        score += 3
    
    return {'score': score, 'features': features}

# Note: region grouping helper removed as unused.

def calculate_metrics(all_predictions):
    """Calculate sensitivity, precision, and specificity for ATG-based START detection.

    Triplet-aware: for each ATG starting at pos, consider START predicted/true if any
    of positions pos,pos+1,pos+2 are START.
    """
    
    # Count ATG-level metrics (only positions with actual ATG codons)
    tp_positions = 0
    fp_positions = 0
    fn_positions = 0
    tn_positions = 0
    
    for result in all_predictions:
        targets = result['targets']
        predictions = result['predictions']
        sequence = convert_tokens_to_sequence(result['sequence_tokens'])
        
        # Only analyze positions that could contain ATG (sequence length - 2)
        for pos in range(len(targets) - 2):
            if pos + 2 >= len(sequence):
                continue
            if sequence[pos:pos+3] != 'ATG':
                continue
            # Triplet-aware START flags
            target_is_start = bool((targets[pos:pos+3] == 2).any())
            pred_is_start = bool((predictions[pos:pos+3] == 2).any())
            if pred_is_start and target_is_start:
                tp_positions += 1
            elif pred_is_start and not target_is_start:
                fp_positions += 1
            elif (not pred_is_start) and target_is_start:
                fn_positions += 1
            else:
                tn_positions += 1
    
    # Calculate metrics
    sensitivity = tp_positions / (tp_positions + fn_positions) if (tp_positions + fn_positions) > 0 else 0
    precision = tp_positions / (tp_positions + fp_positions) if (tp_positions + fp_positions) > 0 else 0
    specificity = tn_positions / (tn_positions + fp_positions) if (tn_positions + fp_positions) > 0 else 0
    
    return {
        'tp': tp_positions,
        'fp': fp_positions,
        'fn': fn_positions,
        'tn': tn_positions,
        'sensitivity': sensitivity,
        'precision': precision,
        'specificity': specificity
    }

def analyze_all_predictions(results_data):
    """Analyze all START predictions with sequence context using direct ATG-based approach.

    Triplet-aware: classify per ATG start by whether any position within the ATG triplet
    is predicted/true START.
    """
    
    print("Analyzing all START predictions...")
    
    all_predictions = []
    
    for result in results_data:
        seq_idx = result['sequence_index']
        sequence = convert_tokens_to_sequence(result['sequence_tokens'])
        targets = result['targets']
        predictions = result['predictions']
        probabilities = result['probabilities']
        
        # print(f"Sequence {seq_idx}: length {len(sequence)}")
        
        # Find all ATG positions and analyze each one
        atg_count = 0
        tp_count = 0
        fp_count = 0
        fn_count = 0
        seen_positions = set()
        
        for pos in range(len(sequence) - 2):
            if sequence[pos:pos+3] != 'ATG':
                continue
            atg_count += 1

            # Triplet-aware signals
            target_triplet = targets[pos:pos+3] if pos+2 < len(targets) else np.array([], dtype=np.int64)
            pred_triplet = predictions[pos:pos+3] if pos+2 < len(predictions) else np.array([], dtype=np.int64)
            prob_pos = probabilities[pos, 2] if pos < len(probabilities) else 0.0
            # Triplet probabilities around the ATG site for START class
            if pos + 2 < probabilities.shape[0]:
                prob_triplet_vec = probabilities[pos:pos+3, 2]
                prob_triplet_max = float(np.max(prob_triplet_vec))
                prob_triplet_avg = float(np.mean(prob_triplet_vec))
            else:
                prob_triplet_max = float(prob_pos)
                prob_triplet_avg = float(prob_pos)

            is_target_start = bool((target_triplet == 2).any())
            is_predicted_start = bool((pred_triplet == 2).any())

            # Only include ATGs that were either predicted as START or should be START
            if is_predicted_start or is_target_start:
                key = (seq_idx, pos)
                if key in seen_positions:
                    raise ValueError(f"Duplicate ATG position detected for sequence {seq_idx} at position {pos}")
                seen_positions.add(key)

                if is_predicted_start and is_target_start:
                    classification = "TP"
                    tp_count += 1
                elif is_predicted_start and not is_target_start:
                    classification = "FP"
                    fp_count += 1
                elif (not is_predicted_start) and is_target_start:
                    classification = "FN"
                    fn_count += 1
                else:
                    continue

                # Get context
                upstream = sequence[max(0, pos-20):pos]
                downstream = sequence[pos+3:pos+23]

                # Analyze Kozak pattern
                kozak_analysis = analyze_kozak_pattern(upstream, 'ATG')

                # Determine target class name (at pos)
                target_class = targets[pos] if pos < len(targets) else -1
                target_name = GenePredictionClass.idx_to_cls[int(target_class)]

                prediction_data = {
                    'sequence_index': seq_idx,
                    'region_positions': [pos],
                    'atg_position': pos,
                    'actual_codon': 'ATG',
                    'is_true_positive': (classification == 'TP'),
                    'classification': classification,
                    'target_class': target_name,
                    'start_probability': float(prob_pos),
                    'start_prob_max_triplet': float(prob_triplet_max),
                    'start_prob_avg_triplet': float(prob_triplet_avg),
                    'upstream_20': upstream,
                    'downstream_20': downstream,
                    'kozak_score': kozak_analysis['score'],
                    'kozak_features': kozak_analysis['features'],
                    'region_length': 1,
                    'pred_triplet': pred_triplet.tolist() if hasattr(pred_triplet, 'tolist') else list(pred_triplet),
                }

                all_predictions.append(prediction_data)

                # print(f"    ATG at {pos}: {classification}, target: {target_name}, prob: {prob_pos:.3f}")
        
        # print(f"  Total ATGs: {atg_count}, Analyzed: TP={tp_count}, FP={fp_count}, FN={fn_count}")
    
    return all_predictions

def validate_predictions(results_data: List[Dict], predictions: List[Dict]):
    """Validate prediction entries for correctness and consistency.
    - Ensure each reported site is an ATG in the source sequence
    - Ensure no duplicate (sequence_index, atg_position)
    - Ensure FN+TP equals number of real ATG STARTs in targets per sequence
    """
    # Map sequences for quick lookup
    seq_map = {}
    target_map = {}
    for result in results_data:
        seq_idx = result['sequence_index']
        seq_map[seq_idx] = convert_tokens_to_sequence(result['sequence_tokens'])
        target_map[seq_idx] = result['targets']

    seen = set()
    for p in predictions:
        seq_idx = p['sequence_index']
        pos = p['atg_position']
        key = (seq_idx, pos)
        if key in seen:
            raise AssertionError(f"Duplicate predicted site: sequence {seq_idx} position {pos}")
        seen.add(key)

        seq = seq_map[seq_idx]
        assert seq[pos:pos+3] == 'ATG', f"Reported site is not ATG at sequence {seq_idx} pos {pos}"

    # Per-sequence TP+FN should equal number of target==2 ATG sites
    # Identify real ATG START positions from targets
    from collections import defaultdict
    real_start_positions = defaultdict(set)
    for seq_idx, targets in target_map.items():
        seq = seq_map[seq_idx]
        for pos in range(0, len(seq) - 2):
            if seq[pos:pos+3] == 'ATG' and targets[pos] == 2:
                real_start_positions[seq_idx].add(pos)

    counted_tp_fn = defaultdict(set)
    for p in predictions:
        if p.get('classification') in ('TP', 'FN'):
            counted_tp_fn[p['sequence_index']].add(p['atg_position'])

    for seq_idx in real_start_positions:
        if real_start_positions[seq_idx] != counted_tp_fn.get(seq_idx, set()):
            raise AssertionError(
                f"TP+FN positions do not match real START ATGs for sequence {seq_idx}:\n"
                f"  expected={sorted(real_start_positions[seq_idx])}\n"
                f"  got={sorted(counted_tp_fn.get(seq_idx, set()))}"
            )

def save_input_sequences_fasta(results_data: List[Dict], output_path: Path):
    """Save input sequences as FASTA file."""
    
    with open(output_path, 'w') as f:
        for result in results_data:
            seq_idx = result['sequence_index']
            sequence = convert_tokens_to_sequence(result['sequence_tokens'])
            
            # Write FASTA header and sequence
            f.write(f">sequence_{seq_idx}\n")
            f.write(f"{sequence}\n")
    
    print(f"✓ Input sequences saved to: {output_path}")

def generate_visual_output(predictions: List[Dict], results_data: List[Dict], output_path: Path):
    """Generate visual sequence output showing predictions with context."""
    
    # Collect all predictions by explicit classification to prevent overlap
    tps = [p for p in predictions if p.get('classification') == 'TP']
    fps = [p for p in predictions if p.get('classification') == 'FP']
    fns = [p for p in predictions if p.get('classification') == 'FN']
    
    with open(output_path, 'w') as f:
        f.write("START Prediction Visual Analysis\n")
        f.write("=" * 80 + "\n\n")
        
        # True Positives
        f.write("TRUE POSITIVES:\n")
        f.write("-" * 40 + "\n")
        for i, tp in enumerate(tps):  # Show all TPs
            seq_idx = tp['sequence_index']
            pos = tp['atg_position']
            prob_max = tp.get('start_prob_max_triplet', tp.get('start_probability', 0.0))
            prob_avg = tp.get('start_prob_avg_triplet', tp.get('start_probability', 0.0))
            
            # Get full context from original sequence
            for result in results_data:
                if result['sequence_index'] == seq_idx:
                    sequence = convert_tokens_to_sequence(result['sequence_tokens'])
                    upstream_200 = sequence[max(0, pos-200):pos]
                    codon = sequence[pos:pos+3]
                    downstream_20 = sequence[pos+3:pos+23]
                    
                    # Create header with sequence name and position
                    header = f">sequence_{seq_idx}@{pos}"
                    
                    # Create visual line
                    context_line = upstream_200 + codon + downstream_20
                    marker_line = " " * len(upstream_200) + f"^^^ TP max={prob_max:.2f} avg={prob_avg:.2f}"

                    # Per-position targets/predictions for window [-200..+20]
                    start_idx = max(0, pos - 200)
                    end_idx = min(len(sequence), pos + 3 + 20)
                    targets_window = result['targets'][start_idx:end_idx].tolist()
                    targets_line = ''.join(str(int(x)) for x in targets_window)
                    preds_window = result['predictions'][start_idx:end_idx].tolist()
                    preds_line = ''.join(str(int(x)) for x in preds_window)
                    
                    f.write(f"{header}\n")
                    f.write(f"{targets_line}\n")
                    f.write(f"{context_line}\n")
                    f.write(f"{preds_line}\n")
                    f.write(f"{marker_line}\n")
                    break
        
        # False Positives
        f.write("\nFALSE POSITIVES:\n")
        f.write("-" * 40 + "\n")
        for i, fp in enumerate(fps):  # Show all FPs
            seq_idx = fp['sequence_index']
            pos = fp['atg_position']
            prob_max = fp.get('start_prob_max_triplet', fp.get('start_probability', 0.0))
            prob_avg = fp.get('start_prob_avg_triplet', fp.get('start_probability', 0.0))
            
            # Get full context from original sequence
            for result in results_data:
                if result['sequence_index'] == seq_idx:
                    sequence = convert_tokens_to_sequence(result['sequence_tokens'])
                    upstream_200 = sequence[max(0, pos-200):pos]
                    codon = sequence[pos:pos+3]
                    downstream_20 = sequence[pos+3:pos+23]
                    
                    # Create header with sequence name and position
                    header = f">sequence_{seq_idx}@{pos}"
                    
                    # Create visual line
                    context_line = upstream_200 + codon + downstream_20
                    marker_line = " " * len(upstream_200) + f"^^^ FP max={prob_max:.2f} avg={prob_avg:.2f}"

                    # Per-position targets/predictions for window [-200..+20]
                    start_idx = max(0, pos - 200)
                    end_idx = min(len(sequence), pos + 3 + 20)
                    targets_window = result['targets'][start_idx:end_idx].tolist()
                    targets_line = ''.join(str(int(x)) for x in targets_window)
                    preds_window = result['predictions'][start_idx:end_idx].tolist()
                    preds_line = ''.join(str(int(x)) for x in preds_window)
                    
                    f.write(f"{header}\n")
                    f.write(f"{targets_line}\n")
                    f.write(f"{context_line}\n")
                    f.write(f"{preds_line}\n")
                    f.write(f"{marker_line}\n")
                    break
        
        # False Negatives
        f.write("\nFALSE NEGATIVES (Missed STARTs):\n")
        f.write("-" * 40 + "\n")
        for i, fn in enumerate(fns):  # Show all FNs
            seq_idx = fn['sequence_index']
            pos = fn['atg_position']
            prob_max = fn.get('start_prob_max_triplet', fn.get('start_probability', 0.0))
            prob_avg = fn.get('start_prob_avg_triplet', fn.get('start_probability', 0.0))
            
            # Get full context from original sequence
            for result in results_data:
                if result['sequence_index'] == seq_idx:
                    sequence = convert_tokens_to_sequence(result['sequence_tokens'])
                    upstream_200 = sequence[max(0, pos-200):pos]
                    codon = sequence[pos:pos+3]
                    downstream_20 = sequence[pos+3:pos+23]
                    
                    # Create header with sequence name and position
                    header = f">sequence_{seq_idx}@{pos}"
                    
                    # Create visual line
                    context_line = upstream_200 + codon + downstream_20
                    marker_line = " " * len(upstream_200) + f"^^^ FN max={prob_max:.2f} avg={prob_avg:.2f}"

                    # Per-position targets/predictions for window [-200..+20]
                    start_idx = max(0, pos - 200)
                    end_idx = min(len(sequence), pos + 3 + 20)
                    targets_window = result['targets'][start_idx:end_idx].tolist()
                    targets_line = ''.join(str(int(x)) for x in targets_window)
                    preds_window = result['predictions'][start_idx:end_idx].tolist()
                    preds_line = ''.join(str(int(x)) for x in preds_window)
                    
                    f.write(f"{header}\n")
                    f.write(f"{targets_line}\n")
                    f.write(f"{context_line}\n")
                    f.write(f"{preds_line}\n")
                    f.write(f"{marker_line}\n")
                    break
    
    print(f"✓ Visual analysis saved to: {output_path}")


def dump_attention_fragments(results_data: List[Dict], predictions: List[Dict], output_fasta: Path, k: int = 5, window: int = 20):
    """Dump top-k attended sequence fragments around predicted START sites to FASTA.

    - For each predicted START (classification=='TP' or 'FP'), for each layer/head, pick top-k attention positions
      from attentions['layer_i'][head, pos, :].
    - Extract a sequence slice [j-window .. j+window] around each top index j.
    - Write FASTA entries named: >input-sequence_layer{L}_head{H}_{weight:.3f}_{j}
    """
    with open(output_fasta, 'w') as f:
        # Build sequence map for quick slicing
        seq_map = {r['sequence_index']: convert_tokens_to_sequence(r['sequence_tokens']) for r in results_data}
        attn_map = {r['sequence_index']: r.get('attentions') for r in results_data}
        for p in predictions:
            if p.get('classification') not in ('TP', 'FP'):
                continue
            sid = p['sequence_index']
            pos = p['atg_position']
            seq = seq_map.get(sid)
            layer_attn = attn_map.get(sid)
            if seq is None or not layer_attn:
                continue
            L = len(seq)
            # For each layer
            for layer_name, att in layer_attn.items():
                # att: (heads, L, L)
                for h in range(att.shape[0]):
                    row = att[h, pos, :]
                    # Get top-k indices and weights
                    top_idx = row.argsort()[-k:][::-1]
                    for j in top_idx:
                        w = float(row[j])
                        s = max(0, j - window)
                        e = min(L, j + window + 1)
                        frag = seq[s:e]
                        header = f">sequence_{sid}_{layer_name}_head_{h}_bp_{j}_weight_{1000*w:.0f}\n"
                        f.write(header)
                        f.write(f"{frag}\n")


# ... existing code ...

def save_analysis_results(predictions: List[Dict], metrics: Dict, results_data: List[Dict], output_dir: Path):
    """Save analysis results with timestamped filenames."""
    
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Compute dataset hash from sequences to bind FASTA and report
    hasher = hashlib.sha256()
    for result in results_data:
        seq = convert_tokens_to_sequence(result['sequence_tokens'])
        hasher.update(seq.encode('utf-8'))
        hasher.update(b"\n")
    dataset_hash = hasher.hexdigest()[:10]

    base_name = f"prediction_{timestamp}_{dataset_hash}"

    # Save input sequences as FASTA
    fasta_output = output_dir / f"{base_name}.fa"
    save_input_sequences_fasta(results_data, fasta_output)
    
    # Generate visual output
    report_output = output_dir / f"{base_name}.txt"
    generate_visual_output(predictions, results_data, report_output)

    # Run validations and print a short footer to stdout for confidence
    validate_predictions(results_data, predictions)

    # Simplified: no threshold sweeps

    # Return base_name so callers can dump additional artifacts named consistently
    return base_name

    # Simplified: no distance-binned analysis

def print_summary(predictions: List[Dict], metrics: Dict):
    """Print concise summary of position-level metrics only."""
    print(f"\nMETRICS (Position-level):")
    print(f"  True Positives: {metrics['tp']}")
    print(f"  False Positives: {metrics['fp']}")
    print(f"  False Negatives: {metrics['fn']}")
    print(f"  True Negatives: {metrics['tn']}")
    print()
    print(f"  Sensitivity: {metrics['sensitivity']:.1%}")
    print(f"  Precision: {metrics['precision']:.1%}")
    print(f"  Specificity: {metrics['specificity']:.1%}")

def _select_checkpoint(run_dir: Path, model_file: Optional[str], legacy_model_path: Optional[str]) -> Path:
    """Select a checkpoint to use.
    Priority: legacy --model-path > run_dir/checkpoints/model_file > best (lowest val_loss) in run_dir/checkpoints.
    """
    if legacy_model_path:
        return Path(legacy_model_path)
    ckpt_dir = run_dir / 'checkpoints'
    if model_file:
        return ckpt_dir / model_file
    # Prefer explicit best alias if present
    best_alias = ckpt_dir / 'best.ckpt'
    if best_alias.exists():
        return best_alias
    # Find best by lowest val_loss encoded as a trailing number before .ckpt
    candidates = list(ckpt_dir.glob('*.ckpt'))
    best_path = None
    best_val = None
    for p in candidates:
        fname = p.name  # include extension for regex
        m = re.search(r'([0-9]+(?:\.[0-9]+)?)\.ckpt$', fname)
        if not m:
            continue
        try:
            val = float(m.group(1))
        except ValueError:
            continue
        if best_val is None or val < best_val:
            best_val = val
            best_path = p
    if best_path is not None:
        return best_path
    # Fallback to last.ckpt
    last = ckpt_dir / 'last.ckpt'
    if last.exists():
        return last
    raise FileNotFoundError(f"No checkpoints found under {ckpt_dir}")


def main():
    parser = argparse.ArgumentParser(description='START prediction analysis')
    parser.add_argument('--run-dir', type=str, required=True,
                       help='Run directory (parent of checkpoints). Outputs will be written here by default.')
    parser.add_argument('--model-file', type=str, default=None,
                       help='Specific checkpoint filename within run_dir/checkpoints to use (overrides best-by-val_loss selection).')
    # Back-compat: allow explicit model path; if provided, overrides run-dir selection
    parser.add_argument('--model-path', type=str, default=None,
                       help='[Deprecated] Explicit path to a model checkpoint; overrides --run-dir/--model-file selection.')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for analysis results. If omitted, defaults to --run-dir.')
    parser.add_argument('--num-sequences', type=int, default=50,
                       help='Number of test sequences to generate')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device to run on (cpu/cuda)')
    # Removed: mismatch threshold (no breakdown TSV)
    parser.add_argument('--layout-version', type=int, default=3,
                       help='Contig layout version, 1=utr5-start, 2=utr5-spacer-start, 3=unmarked_utr5-spacer-start')
    parser.add_argument('--dump-attention-k', type=int, default=3, help='Top-k attention positions per layer/head')
    parser.add_argument('--dump-attention-window', type=int, default=20, help='Sequence half-window around attended position')
    
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir) if args.output_dir else run_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("START Prediction Analysis")
    
    # Resolve checkpoint
    try:
        ckpt_path = _select_checkpoint(run_dir, args.model_file, args.model_path)
    except Exception as e:
        print(f"Error selecting checkpoint: {e}")
        return

    print(f"Selected checkpoint: {ckpt_path}")

    # Load model
    model = load_trained_model(ckpt_path, args.device)
    if model is None:
        return
    
    # Generate test data, aligned to model's max_seq_length
    model_max_len = getattr(getattr(model, 'config', {}).get('model', {}), 'get', lambda k, d=None: None)('max_seq_length', None)
    if model_max_len is None:
        # Fallback: try to read attribute directly from embedding
        try:
            model_max_len = int(model.model.embedding.max_seq_length)
        except Exception:
            model_max_len = 1000
    data_loader, dataset = generate_test_data(args.num_sequences, model_max_len, args.layout_version)
    
    # Run predictions (with attention if requested)
    results = run_predictions(model, data_loader, args.device, return_attention=True)
    
    # Calculate position-level metrics
    metrics = calculate_metrics(results)
    
    # Analyze predictions
    predictions = analyze_all_predictions(results)
    
    # Save results (FASTA + visual report)
    base_name = save_analysis_results(predictions, metrics, results, output_dir)
    
    # Dump attention fragments to FASTA
    attn_fa = output_dir / f"{base_name}_attn.fa"
    dump_attention_fragments(results, predictions, attn_fa, k=args.dump_attention_k, window=args.dump_attention_window)
    print(f"✓ Attention fragments written to: {attn_fa}")
    
    # Print summary
    print_summary(predictions, metrics)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3

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

from utils.constants import GenePredictionClass, ConventionalStopCodons as stop_codons
from dna_learner.model import GenePredictorModule as ModelModule
from torch.utils.data import DataLoader
from utils.genome import AnnotatedGenomeDataset
from utils.metrics import convert_tokens_to_sequence, calculate_generic_metrics

# DRY visualization window sizes
VIS_UPSTREAM_BP = 200
VIS_DOWNSTREAM_BP = 150


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

def generate_test_data(fna_fn: str, tsv_fn: str, max_seq_length: int, layouts_per_contig: int = 1, incl_start: bool = True, incl_stop: bool = True):
    dataset = AnnotatedGenomeDataset(fna_fn, tsv_fn, window = max_seq_length)
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
            
            # Convert to numpy for analysis (batch-agnostic), with optional windowed inference for long sequences
            # Defer model calls to per-sequence to support windowing when needed
            from utils.windowing import compute_window_slices, blend_logits

            B = sequences.size(0)
            for b in range(B):
                seq_tokens_b = sequences[b:b+1]  # (1, L)
                targets_b = targets[b].cpu().numpy()
                L = int(seq_tokens_b.size(1))

                # Resolve model max length
                try:
                    max_len = int(model.model.embedding.max_seq_length)
                except Exception:
                    # Fallback if model is a bare nn.Module with attribute
                    max_len = int(getattr(getattr(model, 'embedding', None), 'max_seq_length', L))

                if L <= max_len:
                    out = model(seq_tokens_b, return_attention=return_attention)
                    if return_attention and isinstance(out, tuple) and len(out) == 2:
                        logits_b, layer_attn_b = out
                    else:
                        logits_b = out
                        layer_attn_b = None
                    preds_b = torch.argmax(logits_b, dim=-1)[0].cpu().numpy()
                    probs_b = torch.softmax(logits_b, dim=-1)[0].cpu().numpy()
                    attn_export = None
                    if return_attention and layer_attn_b is not None:
                        attn_export = {name: tensor[0].cpu().numpy() for name, tensor in layer_attn_b.items() if tensor is not None}
                else:
                    # Windowed inference and blending
                    stride = max_len // 2 if max_len > 1 else 1
                    slices = compute_window_slices(L, window=max_len, stride=stride)
                    window_logits_np = []
                    for (s, e) in slices:
                        win_tokens = seq_tokens_b[:, s:e]  # (1, win_len)
                        out = model(win_tokens, return_attention=False)
                        if isinstance(out, tuple):
                            out = out[0]
                        wl = out[0].detach().cpu().numpy()  # (win_len, C)
                        window_logits_np.append(wl)
                    blended = blend_logits(L, slices, window_logits_np, weight_mode='cosine', margin=None)
                    probs_b = torch.softmax(torch.from_numpy(blended), dim=-1).cpu().numpy()
                    preds_b = np.argmax(probs_b, axis=-1)
                    attn_export = None  # Not supported for windowed case

                seq_np = seq_tokens_b[0].cpu().numpy()
                result_entry = {
                    'sequence_index': batch_idx if B == 1 else f"{batch_idx}:{b}",
                    'sequence_tokens': seq_np,
                    'targets': targets_b,
                    'predictions': preds_b,
                    'probabilities': probs_b,
                }
                if return_attention and attn_export is not None:
                    result_entry['attentions'] = attn_export

                all_results.append(result_entry)
            
            if (batch_idx + 1) % 10 == 0:
                print(f"  Processed {batch_idx + 1} sequences...")
    
    print(f"✓ Completed predictions for {len(all_results)} sequences")
    return all_results

def compute_triplet_prob_stats(probabilities: np.ndarray, position: int, class_index: int) -> Dict[str, float]:
    """Compute per-position probability and max/avg across codon triplet for given class index."""
    if probabilities is None or probabilities.shape[0] == 0:
        return {'pos': 0.0, 'max': 0.0, 'avg': 0.0}
    pos_prob = float(probabilities[position, class_index]) if 0 <= position < probabilities.shape[0] else 0.0
    if position + 2 < probabilities.shape[0]:
        vec = probabilities[position:position+3, class_index]
        return {'pos': pos_prob, 'max': float(np.max(vec)), 'avg': float(np.mean(vec))}
    return {'pos': pos_prob, 'max': pos_prob, 'avg': pos_prob}

 

def analyze_all_predictions(results_data):
    """Analyze START and STOP predictions with sequence context (triplet-aware)."""
    
    print("Analyzing START/STOP predictions...")
    
    all_predictions = []
    
    for result in results_data:
        seq_idx = result['sequence_index']
        sequence = convert_tokens_to_sequence(result['sequence_tokens'])
        targets = result['targets']
        predictions = result['predictions']
        probabilities = result['probabilities']
        
        # print(f"Sequence {seq_idx}: length {len(sequence)}")
        
        # Analyze START sites (ATG)
        atg_count = 0
        tp_count = 0
        fp_count = 0
        fn_count = 0
        seen_positions = set()
        
        for pos in range(len(sequence) - 2):
            if sequence[pos:pos+3] != 'ATG':
                continue
            atg_count += 1

            # Triplet-aware signals (START)
            target_triplet = targets[pos:pos+3] if pos+2 < len(targets) else np.array([], dtype=np.int64)
            pred_triplet = predictions[pos:pos+3] if pos+2 < len(predictions) else np.array([], dtype=np.int64)
            stats = compute_triplet_prob_stats(probabilities, pos, GenePredictionClass.START)
            prob_pos = stats['pos']
            prob_triplet_max = stats['max']
            prob_triplet_avg = stats['avg']

            is_target_start = bool((target_triplet == GenePredictionClass.START).any())
            is_predicted_start = bool((pred_triplet == GenePredictionClass.START).any())

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

                # Determine target class name (at pos)
                target_class = targets[pos] if pos < len(targets) else -1
                target_name = GenePredictionClass.idx_to_cls[int(target_class)]

                prediction_data = {
                    'sequence_index': seq_idx,
                    'region_positions': [pos],
                    'atg_position': pos,
                    'actual_codon': 'ATG',
                    'site_type': 'start',
                    'is_true_positive': (classification == 'TP'),
                    'classification': classification,
                    'target_class': target_name,
                    'start_probability': float(prob_pos),
                    'start_prob_max_triplet': float(prob_triplet_max),
                    'start_prob_avg_triplet': float(prob_triplet_avg),
                    'upstream_20': upstream,
                    'downstream_20': downstream,
                    'region_length': 1,
                    'pred_triplet': pred_triplet.tolist() if hasattr(pred_triplet, 'tolist') else list(pred_triplet),
                }

                all_predictions.append(prediction_data)

        # Analyze STOP sites (TAA/TAG/TGA)
        seen_positions_stop = set()
        for pos in range(len(sequence) - 2):
            triplet = sequence[pos:pos+3]
            if triplet not in stop_codons:
                continue
            target_triplet = targets[pos:pos+3] if pos+2 < len(targets) else np.array([], dtype=np.int64)
            pred_triplet = predictions[pos:pos+3] if pos+2 < len(predictions) else np.array([], dtype=np.int64)
            stats = compute_triplet_prob_stats(probabilities, pos, GenePredictionClass.STOP)
            prob_pos = stats['pos']
            prob_triplet_max = stats['max']
            prob_triplet_avg = stats['avg']

            is_target_stop = bool((target_triplet == GenePredictionClass.STOP).any())
            is_predicted_stop = bool((pred_triplet == GenePredictionClass.STOP).any())
            if not (is_predicted_stop or is_target_stop):
                continue
            key = (seq_idx, pos)
            if key in seen_positions_stop:
                raise ValueError(f"Duplicate STOP position detected for sequence {seq_idx} at position {pos}")
            seen_positions_stop.add(key)

            if is_predicted_stop and is_target_stop:
                classification = "TP"
            elif is_predicted_stop and not is_target_stop:
                classification = "FP"
            elif (not is_predicted_stop) and is_target_stop:
                classification = "FN"
            else:
                continue

            upstream = sequence[max(0, pos-20):pos]
            downstream = sequence[pos+3:pos+23]
            target_class = targets[pos] if pos < len(targets) else -1
            target_name = GenePredictionClass.idx_to_cls[int(target_class)]
            prediction_data = {
                'sequence_index': seq_idx,
                'region_positions': [pos],
                'atg_position': pos,
                'actual_codon': triplet,
                'site_type': 'stop',
                'is_true_positive': (classification == 'TP'),
                'classification': classification,
                'target_class': target_name,
                'stop_probability': float(prob_pos),
                'stop_prob_max_triplet': float(prob_triplet_max),
                'stop_prob_avg_triplet': float(prob_triplet_avg),
                'upstream_20': upstream,
                'downstream_20': downstream,
                'region_length': 1,
                'pred_triplet': pred_triplet.tolist() if hasattr(pred_triplet, 'tolist') else list(pred_triplet),
            }

            all_predictions.append(prediction_data)
        
        # print(f"  Total ATGs: {atg_count}, Analyzed: TP={tp_count}, FP={fp_count}, FN={fn_count}")
    
    return all_predictions

def validate_predictions(results_data: List[Dict], predictions: List[Dict]):
    """Validate prediction entries for correctness and consistency for START and STOP.
    - Ensure each reported site matches expected codon class (ATG for START; TAA/TAG/TGA for STOP)
    - Ensure no duplicate (sequence_index, atg_position)
    - Ensure FN+TP equals number of true START and true STOP sites respectively
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
        site_type = p.get('site_type', 'start')
        triplet = seq[pos:pos+3]
        if site_type == 'start':
            assert triplet == 'ATG', f"Reported START site is not ATG at sequence {seq_idx} pos {pos}"
        else:
            assert triplet in stop_codons, f"Reported STOP site is not a STOP codon at sequence {seq_idx} pos {pos}"

    # Per-sequence TP+FN should equal number of true START and STOP sites
    from collections import defaultdict
    real_start_positions = defaultdict(set)
    real_stop_positions = defaultdict(set)
    for seq_idx, targets in target_map.items():
        seq = seq_map[seq_idx]
        for pos in range(0, max(0, len(seq) - 2)):
            tri = seq[pos:pos+3]
            if tri == 'ATG' and targets[pos] == GenePredictionClass.START:
                real_start_positions[seq_idx].add(pos)
            if tri in stop_codons and targets[pos] == GenePredictionClass.STOP:
                real_stop_positions[seq_idx].add(pos)

    counted_start = defaultdict(set)
    counted_stop = defaultdict(set)
    for p in predictions:
        if p.get('classification') in ('TP', 'FN'):
            if p.get('site_type', 'start') == 'start':
                counted_start[p['sequence_index']].add(p['atg_position'])
            else:
                counted_stop[p['sequence_index']].add(p['atg_position'])

    for seq_idx in real_start_positions:
        if real_start_positions[seq_idx] != counted_start.get(seq_idx, set()):
            raise AssertionError(
                f"TP+FN positions do not match real START ATGs for sequence {seq_idx}:\n"
                f"  expected={sorted(real_start_positions[seq_idx])}\n"
                f"  got={sorted(counted_start.get(seq_idx, set()))}"
            )
    for seq_idx in real_stop_positions:
        if real_stop_positions[seq_idx] != counted_stop.get(seq_idx, set()):
            raise AssertionError(
                f"TP+FN positions do not match real STOP codons for sequence {seq_idx}:\n"
                f"  expected={sorted(real_stop_positions[seq_idx])}\n"
                f"  got={sorted(counted_stop.get(seq_idx, set()))}"
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
        f.write("Prediction Visual Analysis\n")
        f.write("=" * 80 + "\n\n")
        
        # True Positives
        f.write("TRUE POSITIVES:\n")
        f.write("-" * 40 + "\n")
        for i, tp in enumerate(tps):  # Show all TPs
            seq_idx = tp['sequence_index']
            pos = tp['atg_position']
            site_type = tp.get('site_type', 'start')
            if site_type == 'stop':
                prob_max = tp.get('stop_prob_max_triplet', tp.get('stop_probability', 0.0))
                prob_avg = tp.get('stop_prob_avg_triplet', tp.get('stop_probability', 0.0))
                label = 'STOP'
            else:
                prob_max = tp.get('start_prob_max_triplet', tp.get('start_probability', 0.0))
                prob_avg = tp.get('start_prob_avg_triplet', tp.get('start_probability', 0.0))
                label = 'START'
            
            # Get full context from original sequence
            for result in results_data:
                if result['sequence_index'] == seq_idx:
                    sequence = convert_tokens_to_sequence(result['sequence_tokens'])
                    upstream_200 = sequence[max(0, pos-VIS_UPSTREAM_BP):pos]
                    codon = sequence[pos:pos+3]
                    downstream_150 = sequence[pos+3:pos+3+VIS_DOWNSTREAM_BP]
                    
                    # Create header with sequence name and position
                    header = f">sequence_{seq_idx}@{pos}"
                    
                    # Create visual line
                    context_line = upstream_200 + codon + downstream_150
                    marker_line = " " * len(upstream_200) + f"^^^ TP {label} max={prob_max:.2f} avg={prob_avg:.2f}"

                    # Per-position targets/predictions for window [-200..+20]
                    start_idx = max(0, pos - VIS_UPSTREAM_BP)
                    end_idx = min(len(sequence), pos + 3 + VIS_DOWNSTREAM_BP)
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
            site_type = fp.get('site_type', 'start')
            if site_type == 'stop':
                prob_max = fp.get('stop_prob_max_triplet', fp.get('stop_probability', 0.0))
                prob_avg = fp.get('stop_prob_avg_triplet', fp.get('stop_probability', 0.0))
                label = 'STOP'
            else:
                prob_max = fp.get('start_prob_max_triplet', fp.get('start_probability', 0.0))
                prob_avg = fp.get('start_prob_avg_triplet', fp.get('start_probability', 0.0))
                label = 'START'
            
            # Get full context from original sequence
            for result in results_data:
                if result['sequence_index'] == seq_idx:
                    sequence = convert_tokens_to_sequence(result['sequence_tokens'])
                    upstream_200 = sequence[max(0, pos-VIS_UPSTREAM_BP):pos]
                    codon = sequence[pos:pos+3]
                    downstream_150 = sequence[pos+3:pos+3+VIS_DOWNSTREAM_BP]
                    
                    # Create header with sequence name and position
                    header = f">sequence_{seq_idx}@{pos}"
                    
                    # Create visual line
                    context_line = upstream_200 + codon + downstream_150
                    marker_line = " " * len(upstream_200) + f"^^^ FP {label} max={prob_max:.2f} avg={prob_avg:.2f}"

                    # Per-position targets/predictions for window [-200..+20]
                    start_idx = max(0, pos - VIS_UPSTREAM_BP)
                    end_idx = min(len(sequence), pos + 3 + VIS_DOWNSTREAM_BP)
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
        f.write("\nFALSE NEGATIVES (Missed):\n")
        f.write("-" * 40 + "\n")
        for i, fn in enumerate(fns):  # Show all FNs
            seq_idx = fn['sequence_index']
            pos = fn['atg_position']
            site_type = fn.get('site_type', 'start')
            if site_type == 'stop':
                prob_max = fn.get('stop_prob_max_triplet', fn.get('stop_probability', 0.0))
                prob_avg = fn.get('stop_prob_avg_triplet', fn.get('stop_probability', 0.0))
                label = 'STOP'
            else:
                prob_max = fn.get('start_prob_max_triplet', fn.get('start_probability', 0.0))
                prob_avg = fn.get('start_prob_avg_triplet', fn.get('start_probability', 0.0))
                label = 'START'
            
            # Get full context from original sequence
            for result in results_data:
                if result['sequence_index'] == seq_idx:
                    sequence = convert_tokens_to_sequence(result['sequence_tokens'])
                    upstream_200 = sequence[max(0, pos-VIS_UPSTREAM_BP):pos]
                    codon = sequence[pos:pos+3]
                    downstream_150 = sequence[pos+3:pos+3+VIS_DOWNSTREAM_BP]
                    
                    # Create header with sequence name and position
                    header = f">sequence_{seq_idx}@{pos}"
                    
                    # Create visual line
                    context_line = upstream_200 + codon + downstream_150
                    marker_line = " " * len(upstream_200) + f"^^^ FN {label} max={prob_max:.2f} avg={prob_avg:.2f}"

                    # Per-position targets/predictions for window [-200..+150]
                    start_idx = max(0, pos - VIS_UPSTREAM_BP)
                    end_idx = min(len(sequence), pos + 3 + VIS_DOWNSTREAM_BP)
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
    """Dump top-k attended sequence fragments around predicted START/STOP sites to FASTA.

    FASTA header format:
      >sequence_{sid}_{site_type}_{layer}_head_{h}_bp_{j}_weight_{1000*w}
    where site_type is 'start' or 'stop'.
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
            site_type = p.get('site_type', 'start')
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
                        header = f">sequence_{sid}_{site_type}_{layer_name}_head_{h}_bp_{j}_weight_{1000*w:.0f}\n"
                        f.write(header)
                        f.write(f"{frag}\n")

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

    # Return base_name so callers can dump additional artifacts named consistently
    return base_name

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
    parser = argparse.ArgumentParser(description='Gene prediction analysis')
    parser.add_argument('--fna-fn', type=str, required=True, help='File name for genome sequence in FASTA format')
    parser.add_argument('--tsv-fn', type=str, required=True, help='File name for annotations in TSV format')
    parser.add_argument('--run-dir', type=str, required=True,
                       help='Run directory (parent of checkpoints). Outputs will be written here by default.')
    parser.add_argument('--model-file', type=str, default=None,
                       help='Specific checkpoint filename within run_dir/checkpoints to use (overrides best-by-val_loss selection).')
    # Back-compat: allow explicit model path; if provided, overrides run-dir selection
    parser.add_argument('--model-path', type=str, default=None,
                       help='[Deprecated] Explicit path to a model checkpoint; overrides --run-dir/--model-file selection.')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for analysis results. If omitted, defaults to --run-dir.')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device to run on (cpu/cuda)')
    parser.add_argument('--dump-attention-k', type=int, default=1, help='Top-k attention positions per layer/head')
    parser.add_argument('--dump-attention-window', type=int, default=20, help='Sequence half-window around attended position')
    
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir) if args.output_dir else run_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("START/STOP Prediction Analysis")
    
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
    # Reflect CC status from checkpoint (informational only)
    try:
        cc_cfg = getattr(model, 'config', {}).get('model', {}).get('class_conditional_readouts')
        if cc_cfg and bool(cc_cfg.get('enabled')):
            print("Class-conditional readouts: enabled")
            for e in (cc_cfg.get('entries') or []):
                print(f"  - {e.get('class')} before={e.get('before')} gap={e.get('gap')} after={e.get('after')}")
        else:
            print("Class-conditional readouts: disabled")
    except Exception:
        pass
    
    # Generate test data, aligned to model's max_seq_length
    model_max_len = getattr(getattr(model, 'config', {}).get('model', {}), 'get', lambda k, d=None: None)('max_seq_length', None)
    if model_max_len is None:
        # Fallback: try to read attribute directly from embedding
        try:
            model_max_len = int(model.model.embedding.max_seq_length)
        except Exception:
            model_max_len = 1000

    data_loader, dataset = generate_test_data(args.fna_fn, args.tsv_fn, model_max_len)
    
    # Run predictions (with attention if requested)
    results = run_predictions(model, data_loader, args.device, return_attention=True)
    
    # Generic metrics: use class weights from config if available
    try:
        cw = getattr(model, 'config', {}).get('loss', {}).get('class_weights')
    except Exception:
        cw = None
    generic = calculate_generic_metrics(results, class_weights=cw, min_weight=1.0)
    
    # Analyze predictions
    predictions = analyze_all_predictions(results)

    # Save results (FASTA + visual report)
    base_name = save_analysis_results(predictions, generic, results, output_dir)

    # Print generic per-class metrics (for classes selected above)
    if generic:
        print("\nPer-class metrics:")
        for cls_idx in sorted(generic.keys()):
            name = GenePredictionClass.idx_to_cls.get(int(cls_idx), str(cls_idx))
            m = generic[cls_idx]
            print(f"  {name:>10s}  TP={m['tp']} FP={m['fp']} FN={m['fn']} TN={m['tn']}  "
                  f"Sensitivity={m['sensitivity']:.1%} Precision={m['precision']:.1%} Specificity={m['specificity']:.1%}")
    
    # Dump attention fragments to FASTA
    attn_fa = output_dir / f"{base_name}_attn.fa"
    dump_attention_fragments(results, predictions, attn_fa, k=args.dump_attention_k, window=args.dump_attention_window)
    print(f"✓ Attention fragments written to: {attn_fa}")
    
    

if __name__ == "__main__":
    main()

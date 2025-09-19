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
    # not windowing in the dataset class, but rely on windowing here and then blending the results here
    dataset = AnnotatedGenomeDataset(fna_fn, tsv_fn, window = None)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    print(f"✓ Generated {len(dataset)} test windows")
    return data_loader, dataset


def run_predictions(model, data_loader, device='cpu', return_attention: bool = False):
    """Run predictions on test data. Optionally return encoder attention per layer."""
    
    print("Running predictions on test data...")
    
    all_results = []
    predicted_count = 0
    
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
                    predicted_count += 1
                    if (predicted_count + 1) % 10 == 0:
                        print(f"  Processed {predicted_count + 1} windows...")
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
                        predicted_count += 1
                        if (predicted_count + 1) % 10 == 0:
                            print(f"  Processed {predicted_count + 1} windows...")
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


def _format_with_colors(text: str, positions_to_color: List[Tuple[int, int, str]], ansi_colors: bool) -> str:
    """Return text with ANSI colors applied for given segments.

    positions_to_color: list of (start, end_inclusive, classification) sorted by start.
    classification determines the color: TP=green, FP=red, FN=yellow.
    """
    if not ansi_colors or not positions_to_color:
        return text
    color_map = {
        'TP': '\u001b[1;32m',
        'FP': '\u001b[1;31m',
        'FN': '\u001b[0;37;42m',
    }
    reset = '\u001b[0m'
    out = []
    idx = 0
    for (s, e, clsf) in positions_to_color:
        s = max(0, s)
        e = max(s - 1, e)
        if s > idx:
            out.append(text[idx:s])
        out.append(color_map.get(clsf, ''))
        out.append(text[s:e+1])
        out.append(reset)
        idx = e + 1
    if idx < len(text):
        out.append(text[idx:])
    return ''.join(out)


def _compute_span_stats(probabilities: np.ndarray, start: int, end_incl: int, class_index: int) -> Tuple[float, float]:
    """Compute (max, avg) probability over [start, end_incl] for class_index."""
    if probabilities is None or probabilities.shape[0] == 0:
        return 0.0, 0.0
    s = max(0, start)
    e = min(probabilities.shape[0] - 1, end_incl)
    vec = probabilities[s:e+1, class_index]
    return float(np.max(vec)), float(np.mean(vec))


def _extract_sites_from_labels(labels: np.ndarray, class_index: int) -> List[Tuple[int, int]]:
    """Return list of (start, end_inclusive) spans where labels==class_index (contiguous runs)."""
    spans = []
    i = 0
    L = int(len(labels))
    while i < L:
        if int(labels[i]) == int(class_index):
            j = i + 1
            while j < L and int(labels[j]) == int(class_index):
                j += 1
            spans.append((i, j - 1))
            i = j
        else:
            i += 1
    return spans


def _expected_span_length(class_index: int) -> int:
    if int(class_index) in (int(GenePredictionClass.START), int(GenePredictionClass.STOP)):
        return 3
    if int(class_index) in (int(GenePredictionClass.DSS), int(GenePredictionClass.ASS)):
        return 2
    return 1


def _normalize_span_length(start: int, end_incl: int, length: int, class_index: int) -> Tuple[int, int]:
    """Ensure span covers at least the expected number of bases for the class.
    If shorter, extend to the right up to sequence end.
    """
    expected = _expected_span_length(class_index)
    if start < 0:
        start = 0
    if end_incl < start:
        end_incl = start
    current = end_incl - start + 1
    if current < expected:
        end_incl = min(length - 1, start + expected - 1)
    return start, end_incl


def _merge_site_priority_map(length: int, sites: List[Dict]) -> Tuple[List[bool], List[Optional[int]]]:
    """Create per-position uppercase mask and site assignment with priority TP>FN>FP."""
    uppercase = [False] * length
    site_at: List[Optional[int]] = [None] * length
    priority = {'FP': 1, 'FN': 2, 'TP': 3}
    for sid, s in enumerate(sites):
        clsf = s['classification']
        p = priority.get(clsf, 0)
        for pos in range(s['start'], s['end'] + 1):
            if 0 <= pos < length:
                prev = site_at[pos]
                if prev is None:
                    site_at[pos] = sid
                    uppercase[pos] = True
                else:
                    prev_p = priority.get(sites[prev]['classification'], 0)
                    if p > prev_p:
                        site_at[pos] = sid
                        uppercase[pos] = True
    return uppercase, site_at


def generate_per_contig_report(results_data: List[Dict], output_path: Path, class_weights: Optional[List[float]] = None,
                               line_width: int = 100, ansi_colors: bool = True):
    """Write one report per contig with colored spans and end-of-line annotations.

    Included classes: any with weight>1.0 from class_weights; if None, defaults to START, STOP, DSS, ASS.
    """
    # Determine included classes
    if class_weights and len(class_weights) > 0:
        include_classes = {idx for idx, w in enumerate(class_weights) if float(w) > 1.0}
    else:
        include_classes = {GenePredictionClass.START, GenePredictionClass.STOP, GenePredictionClass.DSS, GenePredictionClass.ASS}

    include_classes = {int(i) for i in include_classes if int(i) in GenePredictionClass.idx_to_cls}

    with open(output_path, 'w') as f:
        for result in results_data:
            seq_idx = result['sequence_index']
            sequence = convert_tokens_to_sequence(result['sequence_tokens'])
            L = len(sequence)
            targets = result['targets']
            preds = result['predictions']
            probs = result['probabilities']

            # Build sites from targets (TP/FN) and predictions (FP)
            sites: List[Dict] = []
            for cls_idx in sorted(include_classes):
                # True sites
                true_spans = _extract_sites_from_labels(targets, cls_idx)
                pred_spans = _extract_sites_from_labels(preds, cls_idx)

                # Index predicted spans for FP check
                pred_spans_copy = list(pred_spans)

                # For each true span, decide TP/FN
                for (s, e) in true_spans:
                    s, e = _normalize_span_length(s, e, L, int(cls_idx))
                    # predicted positive if any overlap with predicted labels
                    pred_pos = any(int(preds[j]) == int(cls_idx) for j in range(s, e + 1) if 0 <= j < L)
                    clsf = 'TP' if pred_pos else 'FN'
                    pmax, pavg = _compute_span_stats(probs, s, e, int(cls_idx))
                    sites.append({
                        'start': s,
                        'end': e,
                        'class_index': int(cls_idx),
                        'label': GenePredictionClass.idx_to_cls[int(cls_idx)],
                        'classification': clsf,
                        'prob_max': pmax,
                        'prob_avg': pavg,
                    })

                # Add FP spans: predicted spans that do not overlap any true span of same class
                for (ps, pe) in pred_spans_copy:
                    ps, pe = _normalize_span_length(ps, pe, L, int(cls_idx))
                    overlaps = any(not (pe < ts or ps > te) for (ts, te) in true_spans)
                    if not overlaps:
                        pmax, pavg = _compute_span_stats(probs, ps, pe, int(cls_idx))
                        sites.append({
                            'start': ps,
                            'end': pe,
                            'class_index': int(cls_idx),
                            'label': GenePredictionClass.idx_to_cls[int(cls_idx)],
                            'classification': 'FP',
                            'prob_max': pmax,
                            'prob_avg': pavg,
                        })

            # Prepare per-position mapping for uppercase and color priority
            uppercase, site_at = _merge_site_priority_map(L, sites)

            # Header per sequence
            f.write(f">sequence_{seq_idx}\n")

            # Render sequence in chunks
            base_seq = sequence.lower()
            for start in range(0, L, int(line_width)):
                end_excl = min(L, start + int(line_width))
                # Build line characters with uppercase at sites
                chars = []
                segments_for_color: List[Tuple[int, int, str]] = []
                current_sid = None
                seg_start = None
                for gpos in range(start, end_excl):
                    sid = site_at[gpos]
                    ch = base_seq[gpos]
                    if sid is not None:
                        ch = ch.upper()
                    chars.append(ch)
                    if sid != current_sid:
                        # close previous
                        if current_sid is not None:
                            segments_for_color.append((seg_start - start, gpos - 1 - start, sites[current_sid]['classification']))
                        # open new if present
                        if sid is not None:
                            seg_start = gpos
                        current_sid = sid
                if current_sid is not None:
                    segments_for_color.append((seg_start - start, end_excl - 1 - start, sites[current_sid]['classification']))

                line_text = ''.join(chars)
                # Apply colors if enabled
                if ansi_colors and segments_for_color:
                    # translate segments to local positions
                    colored_segments = []
                    for (ls, le, clsf) in segments_for_color:
                        colored_segments.append((ls, le, clsf))
                    line_text = _format_with_colors(line_text, colored_segments, ansi_colors=True)

                # Build end-of-line annotations
                line_sites = []
                for idx, s in enumerate(sites):
                    if not (s['end'] < start or s['start'] >= end_excl):
                        line_sites.append((s['start'], s))
                line_sites.sort(key=lambda t: t[0])
                if line_sites:
                    annotations = []
                    for _, s in line_sites:
                        annotations.append(f"{s['label']} {s['prob_max']:.2f}/{s['prob_avg']:.2f}")
                    line_text = f"{line_text}  " + "; ".join(annotations)

                f.write(f"{line_text}\n")
    print(f"✓ Per-contig report saved to: {output_path}")


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


def save_analysis_results(results_data: List[Dict], output_dir: Path, class_weights: Optional[List[float]] = None,
                         line_width: int = 100, ansi_colors: bool = True):
    """Save analysis results with timestamped filenames (FASTA + per-contig colored report)."""
    
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
    
    # Generate per-contig colored report
    report_output = output_dir / f"{base_name}.txt"
    generate_per_contig_report(results_data, report_output, class_weights=class_weights, line_width=line_width, ansi_colors=ansi_colors)

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
    parser.add_argument('--line-width', type=int, default=100, help='Number of base pairs per line in the report (.txt)')
    parser.add_argument('--no-ansi-colors', dest='ansi_colors', action='store_false', help='Disable ANSI colors in the report')
    parser.set_defaults(ansi_colors=True)
    
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
    
    # Analyze predictions for attention export only (legacy functionality)
    predictions = analyze_all_predictions(results)

    # Save results (FASTA + per-contig colored report)
    base_name = save_analysis_results(results, output_dir, class_weights=cw, line_width=args.line_width, ansi_colors=args.ansi_colors)

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

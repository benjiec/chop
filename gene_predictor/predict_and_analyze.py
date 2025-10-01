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
import pickle
import random
import re

from utils.constants import GenePredictionClass, ConventionalStopCodons as stop_codons
from dna_learner.model import GenePredictorModule as ModelModule
from torch.utils.data import DataLoader
from utils.genome import AnnotatedGenomeDataset
from utils.metrics import convert_tokens_to_sequence, calculate_generic_metrics_and_predictions
from utils.metrics import compute_brier_scores
from gene_decoder import PredictedSequence
from utils.metrics import convert_tokens_to_sequence
from utils.windowing import compute_window_slices, blend_logits
from utils.windowing import window_weights
from utils.constants import DNAEmbed


def predict_sequence_outputs(model, max_seq_len, seq_tokens_b: torch.Tensor,
                             stride: Optional[int] = None,
                             device: str = 'cpu',
                             return_attention: bool = False,
                             temperature: Optional[float] = None,
                             blending_window_margin_bp: int = 200,
                             aggregator: str = 'blend',  # 'blend' | 'max_weight' | 'max_prob'
                             random_prefix_ns: bool = True,
                             random_prefix_min: int = 100,
                             random_prefix_max: int = 400) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Predict a single sequence (shape (1, L)) and return (preds, probs, logits_raw).

    Handles windowed inference and blending when sequence length exceeds model max length.
    """

    L = int(seq_tokens_b.size(1))

    # Optional random N-prefix to reduce edge effects; apply only when windowing will occur
    pad_len = 0
    if random_prefix_ns and (L > max_seq_len) and random_prefix_max > 0 and random_prefix_max >= random_prefix_min:
        pad_len = random.randint(int(random_prefix_min), int(random_prefix_max))
        if pad_len > 0:
            pad = torch.full((1, pad_len), int(DNAEmbed.N), dtype=seq_tokens_b.dtype, device=seq_tokens_b.device)
            seq_tokens_b = torch.cat([pad, seq_tokens_b], dim=1)
            L = int(seq_tokens_b.size(1))

    # Normalize aggregator: default to 'blend' if unrecognized
    if aggregator not in ('blend', 'max_weight', 'max_prob'):
        aggregator = 'blend'

    if L <= max_seq_len:
        _layer_attn_b = None
        out = model(seq_tokens_b, return_attention=return_attention)
        if return_attention and isinstance(out, tuple) and len(out) == 2:
            logits_b, _layer_attn_b = out
        else:
            logits_b = out
        logits_raw_np = logits_b[0].detach().cpu().numpy()
        if temperature is not None and float(temperature) > 0:
            logits_b = logits_b / float(temperature)
        preds_b = torch.argmax(logits_b, dim=-1)[0].cpu().numpy()
        probs_b = torch.softmax(logits_b, dim=-1)[0].cpu().numpy()
        # Strip prefix if applied
        if pad_len > 0:
            preds_b = preds_b[pad_len:]
            probs_b = probs_b[pad_len:]
            logits_raw_np = logits_raw_np[pad_len:]
        return preds_b, probs_b, logits_raw_np, _layer_attn_b

    else:
        # Windowed inference and blending
        if stride is None:
            # Default stride: one-third overlap windows
            stride = max(max_seq_len // 3, 1)
        slices = compute_window_slices(L, window=max_seq_len, stride=stride)
        window_logits_np = []
        for (s, e) in slices:
            win_tokens = seq_tokens_b[:, s:e]  # (1, win_len)
            out = model(win_tokens, return_attention=False)
            if isinstance(out, tuple):
                out = out[0]
            wl = out[0].detach().cpu().numpy()  # (win_len, C)
            window_logits_np.append(wl)
        # Cap margin to avoid leaving uncovered gaps when stride is small
        eff_margin = int(blending_window_margin_bp)
        eff_margin = max(0, min(eff_margin, max(0, stride // 2 - 1)))

        if aggregator in ('max_weight', 'max_prob'):
            # Build per-position choices from overlapping windows
            num_classes = int(window_logits_np[0].shape[-1])
            logits_raw_np = np.zeros((L, num_classes), dtype=np.float32)
            weight_sums = np.zeros((L,), dtype=np.float32)
            for (s, e), wl in zip(slices, window_logits_np):
                win_len = e - s
                w = window_weights(win_len, mode='cosine', margin=eff_margin)
                if aggregator == 'max_weight':
                    # At each position, choose window with highest weight
                    for i in range(win_len):
                        pos = s + i
                        if w[i] >= weight_sums[pos]:
                            logits_raw_np[pos, :] = wl[i, :]
                            weight_sums[pos] = w[i]
                else:  # max_prob
                    # Choose window with highest top-class logit
                    top_vals = wl.max(axis=1)
                    for i in range(win_len):
                        pos = s + i
                        if top_vals[i] >= weight_sums[pos]:
                            logits_raw_np[pos, :] = wl[i, :]
                            weight_sums[pos] = top_vals[i]
        else:
            blended_raw = blend_logits(L, slices, window_logits_np, weight_mode='cosine', margin=eff_margin, exclude_edges=True)
            logits_raw_np = blended_raw

        blended = logits_raw_np
        if temperature is not None and float(temperature) > 0:
            blended = blended / float(temperature)
        probs_b = torch.softmax(torch.from_numpy(blended), dim=-1).cpu().numpy()
        preds_b = np.argmax(probs_b, axis=-1)

        # Strip prefix if applied
        if pad_len > 0:
            preds_b = preds_b[pad_len:]
            probs_b = probs_b[pad_len:]
            logits_raw_np = logits_raw_np[pad_len:]

        return preds_b, probs_b, logits_raw_np, None


def load_trained_model(model_path: Path, device='cpu', temperature: Optional[float] = None):
    """Load the trained model from checkpoint, restoring exact architecture."""
    print(f"Loading model from: {model_path}")

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


def generate_test_data(fna_fn: str, tsv_fn: str, num_contigs: int = 0):
    # not windowing in the dataset class, but rely on windowing here and then blending the results here
    dataset = AnnotatedGenomeDataset(fna_fn, tsv_fn, window = None, num_contigs = num_contigs, random_prefix_ns=False)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    print(f"✓ Generated {len(dataset)} test samples - windowing and blending results...")
    return data_loader, dataset


def run_predictions(model, data_loader, device='cpu', return_attention: bool = False, temperature: Optional[float] = None, log_every: Optional[int] = 10,
                    blending_window_margin_bp: int = 200, aggregator: str = 'blend', random_prefix_ns: bool = True,
                    random_prefix_min: int = 100, random_prefix_max: int = 400):
    """Run predictions on test data. Optionally return encoder attention per layer."""
    
    print("Running predictions on test data...")
    
    all_results = []
    predicted_count = 0

    max_len = int(model.model.embedding.max_seq_length)

    with torch.no_grad():
        for batch_idx, (sequences, targets) in enumerate(data_loader):
            sequences = sequences.to(device)
            targets = targets.to(device)

            B = sequences.size(0)
            for b in range(B):
                seq_tokens_b = sequences[b:b+1]  # (1, L)
                targets_b = targets[b].cpu().numpy()
                L = int(seq_tokens_b.size(1))

                # Run per-sequence prediction via helper
                preds_b, probs_b, logits_raw_np, layer_attn_b = predict_sequence_outputs(
                    model, max_len, seq_tokens_b,
                    device=device,
                    return_attention=False,
                    temperature=temperature,
                    blending_window_margin_bp=blending_window_margin_bp,
                    aggregator=aggregator,
                    random_prefix_ns=random_prefix_ns,
                    random_prefix_min=random_prefix_min,
                    random_prefix_max=random_prefix_max,
                )

                predicted_count += 1
                if (predicted_count + 1) % log_every == 0:
                    print(f"  Processed {predicted_count + 1} sequences...")

                attn_export = None
                if return_attention and layer_attn_b:
                    attn_export = {name: tensor[0].cpu().numpy() for name, tensor in layer_attn_b.items() if tensor is not None}

                seq_np = seq_tokens_b[0].cpu().numpy()
                result_entry = {
                    'sequence_index': batch_idx if B == 1 else f"{batch_idx}:{b}",
                    'sequence_tokens': seq_np,
                    'targets': targets_b,
                    'predictions': preds_b,
                    'probabilities': probs_b,
                    'logits_raw': logits_raw_np,
                }
                if return_attention and attn_export is not None:
                    result_entry['attentions'] = attn_export

                all_results.append(result_entry)
            
    print(f"✓ Completed predictions for {len(all_results)} sequences")
    return all_results


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
                               line_width: int = 100, ansi_colors: bool = True, events: Optional[List[Dict]] = None):
    """Write one report per contig with colored spans and end-of-line annotations.

    Included classes: any with weight>1.0 from class_weights; if None, defaults to START, STOP, DSS, ASS.
    """
    # Determine included classes (robust to tensor/array inputs)
    cw_list: Optional[List[float]] = None
    try:
        if class_weights is not None and isinstance(class_weights, torch.Tensor):
            cw_list = class_weights.detach().cpu().flatten().tolist()
        elif class_weights is not None:
            cw_list = list(class_weights)
    except Exception:
        cw_list = list(class_weights) if class_weights is not None else None

    if cw_list is not None and len(cw_list) > 0:
        include_classes = {idx for idx, w in enumerate(cw_list) if float(w) > 1.0}
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

            # Build sites from metrics events if provided; otherwise fall back to local derivation
            sites: List[Dict] = []
            if events is not None:
                for ev in events:
                    if ev.get('sequence_index') != seq_idx:
                        continue
                    cls_idx = int(ev.get('class_index'))
                    if cls_idx not in include_classes:
                        continue
                    s = int(ev.get('start'))
                    e = int(ev.get('end'))
                    pmax, pavg = _compute_span_stats(probs, s, e, cls_idx)
                    sites.append({
                        'start': s,
                        'end': e,
                        'class_index': cls_idx,
                        'label': GenePredictionClass.idx_to_cls.get(cls_idx, str(cls_idx)),
                        'classification': ev.get('classification', ''),
                        'prob_max': pmax,
                        'prob_avg': pavg,
                    })
            else:
                for cls_idx in sorted(include_classes):
                    # True sites
                    true_spans = _extract_sites_from_labels(targets, cls_idx)
                    pred_spans = _extract_sites_from_labels(preds, cls_idx)
                    pred_spans_copy = list(pred_spans)
                    for (s, e) in true_spans:
                        s, e = _normalize_span_length(s, e, L, int(cls_idx))
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
            cls_idx = int(p.get('class_index', -1))
            # Only export attention around START/STOP motif spans
            if cls_idx not in (GenePredictionClass.START, GenePredictionClass.STOP):
                continue
            sid = p['sequence_index']
            pos = int(p.get('start', 0))
            site_type = 'start' if cls_idx == GenePredictionClass.START else 'stop'
            seq = seq_map.get(sid)
            layer_attn = attn_map.get(sid)
            if seq is None or not layer_attn:
                continue
            L = len(seq)
            if pos < 0 or pos >= L:
                continue
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
                         line_width: int = 100, ansi_colors: bool = True, events: Optional[List[Dict]] = None):
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
    
    # Generate per-contig colored report (metrics-aligned coloring).
    report_output = output_dir / f"{base_name}.txt"
    generate_per_contig_report(results_data, report_output, class_weights=class_weights, line_width=line_width, ansi_colors=ansi_colors, events=events)

    # Return base_name so callers can dump additional artifacts named consistently
    return base_name


def _select_checkpoint_explicit(model_path: Optional[str]) -> Path:
    if not model_path:
        raise ValueError("--model-path is required; auto-selection has been removed.")
    return Path(model_path)


def main():
    parser = argparse.ArgumentParser(description='Gene prediction analysis')
    parser.add_argument('--fna-fn', type=str, required=True, help='File name for genome sequence in FASTA format')
    parser.add_argument('--tsv-fn', type=str, required=True, help='File name for annotations in TSV format')
    parser.add_argument('--num-contigs', type=int, default=0, help='Number of contigs, if 0 use all from input file')
    parser.add_argument('--run-dir', type=str, required=True,
                       help='Run directory that contains the checkpoints subdirectory.')
    parser.add_argument('--model-path', type=str, required=True,
                       help='Checkpoint path. If relative, it is resolved under <run-dir>/checkpoints/. Absolute paths are accepted.')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for analysis results. If omitted, defaults to the run directory.')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device to run on (cpu/cuda)')
    parser.add_argument('--temperature', type=float, default=3.0, help='Temperature scaling for logits at inference (softmax(logits/T)).')
    parser.add_argument('--t-sweep', type=str, default=None, help='Optional sweep "start:stop:step" over T; reports best Brier.')
    parser.add_argument('--dump-attention-k', type=int, default=1, help='Top-k attention positions per layer/head')
    parser.add_argument('--dump-attention-window', type=int, default=20, help='Sequence half-window around attended position')
    parser.add_argument('--line-width', type=int, default=100, help='Number of base pairs per line in the report (.txt)')
    parser.add_argument('--no-ansi-colors', dest='ansi_colors', action='store_false', help='Disable ANSI colors in the report')
    parser.set_defaults(ansi_colors=True)
    # New options for blending/windowing behavior
    parser.add_argument('--aggregator', type=str, default='blend', choices=['blend', 'max_weight', 'max_prob'], help='Window aggregation mode')
    parser.add_argument('--blending-window-margin-bp', type=int, default=200, help='Edge margin for blending/selection')
    parser.add_argument('--random-prefix-ns', action='store_true', default=True, help='Enable random N-prefix before windowing')
    parser.add_argument('--no-random-prefix-ns', dest='random_prefix_ns', action='store_false', help='Disable random N-prefix before windowing')
    parser.add_argument('--random-prefix-min', type=int, default=100, help='Minimum N-prefix length')
    parser.add_argument('--random-prefix-max', type=int, default=400, help='Maximum N-prefix length')
    parser.add_argument('--report-loss-components', action='store_true', help='Compute adjusted loss and its components per sequence (no temperature) and report means')
    parser.add_argument('--write-decoder-input-pkl', action='store_true', help='If set, write a pickle list of PredictedSequence for decoder input')
    
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    raw_model_path = Path(args.model_path)
    ckpt_path = raw_model_path if raw_model_path.is_absolute() else (run_dir / 'checkpoints' / raw_model_path)
    output_dir = Path(args.output_dir) if args.output_dir else run_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
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
    
    # Generic metrics: use class weights from config if available (needed for consistent Brier computation)
    try:
        cw = getattr(model, 'config', {}).get('loss', {}).get('class_weights')
    except Exception:
        cw = None

    # Generate test data, aligned to model's max_seq_length
    model_max_len = getattr(getattr(model, 'config', {}).get('model', {}), 'get', lambda k, d=None: None)('max_seq_length', None)
    if model_max_len is None:
        # Fallback: try to read attribute directly from embedding
        try:
            model_max_len = int(model.model.embedding.max_seq_length)
        except Exception:
            model_max_len = 1000

    data_loader, dataset = generate_test_data(args.fna_fn, args.tsv_fn, args.num_contigs)
    
     # Temperature sweep (if requested)
    sweep_best = None
    if args.t_sweep:
        try:
            s, e, st = [float(x) for x in args.t_sweep.split(':')]
            Ts = np.arange(s, e + 1e-9, st)
        except Exception as ex:
            print(f"Invalid --t-sweep '{args.t_sweep}': {ex}")
            Ts = []
        for T in Ts:
            results_T = run_predictions(model, data_loader, args.device, return_attention=False, temperature=T, log_every=100)
            brier_T = compute_brier_scores(results_T, class_weights=cw, min_weight=1.0, event_only=True)
            print(f"T={T:.3f}  Brier={brier_T.get('brier', 0.0):.4f}")
            by_cls_T = brier_T.get('brier_by_class', {})
            if by_cls_T:
                parts = []
                for cls_idx in sorted(by_cls_T.keys()):
                    name = GenePredictionClass.idx_to_cls.get(int(cls_idx), str(int(cls_idx)))
                    parts.append(f"{name}={float(by_cls_T[cls_idx]):.4f}")
                print("  " + " ".join(parts))
            if (sweep_best is None) or (brier_T.get('brier', 0.0) < sweep_best[0]):
                sweep_best = (float(brier_T.get('brier', 0.0)), float(T))
        if sweep_best is not None:
            print(f"Best T by Brier: {sweep_best[1]:.3f} (Brier={sweep_best[0]:.4f})")
        # Fall through to run with requested --temperature (or best-T, if not given)
        if args.temperature is None and sweep_best is not None:
            args.temperature = sweep_best[1]

    # Run predictions (with attention if requested) using final temperature
    results = run_predictions(
        model,
        data_loader,
        args.device,
        return_attention=True,
        temperature=args.temperature,
        blending_window_margin_bp=int(args.blending_window_margin_bp),
        aggregator=str(args.aggregator),
        random_prefix_ns=bool(args.random_prefix_ns),
        random_prefix_min=int(args.random_prefix_min),
        random_prefix_max=int(args.random_prefix_max),
    )
    
    # Compute metrics and motif-span prediction events for visualization (single call)
    generic, events = calculate_generic_metrics_and_predictions(results, class_weights=cw, min_weight=1.0)
    
    # Optionally compute loss components using the model (no recomputation here)
    if args.report_loss_components:
        try:
            # Use module method for consistent behavior
            model.eval()
            total = 0.0
            ce_total = 0.0
            ent_total = 0.0
            fp_total = 0.0
            count = 0
            # Per-class CE aggregates across all sequences (weighted sums)
            ce_weighted_sum_by_class: Dict[int, float] = {}
            weight_sum_by_class: Dict[int, float] = {}
            total_weighted_ce_sum: float = 0.0
            with torch.no_grad():
                for r in results:
                    logits_np = r.get('logits_raw')
                    targets_np = r.get('targets')
                    if logits_np is None or targets_np is None:
                        continue
                    logits = torch.from_numpy(logits_np).unsqueeze(0).to(args.device)
                    targets = torch.from_numpy(np.array(targets_np, dtype=np.int64)).unsqueeze(0).to(args.device)

                    # Ask model to emit components directly
                    comp: Dict[str, Any] = {}
                    _ = model._compute_adjusted_loss(logits, targets, components_out=comp)
                    total += float(comp.get('total', 0.0))
                    ce_total += float(comp.get('ce', 0.0))
                    ent_total += float(comp.get('entropy', 0.0))
                    fp_total += float(comp.get('fp_penalty', 0.0))

                    # Aggregate per-class CE weighted sums across sequences
                    ce_ws = comp.get('ce_weighted_sum_by_class', {}) or {}
                    wt_ws = comp.get('weight_sum_by_class', {}) or {}
                    total_weighted_ce_sum += float(comp.get('total_weighted_ce_sum', 0.0))
                    for k, num_k in ce_ws.items():
                        ce_weighted_sum_by_class[int(k)] = ce_weighted_sum_by_class.get(int(k), 0.0) + float(num_k)
                    for k, den_k in wt_ws.items():
                        weight_sum_by_class[int(k)] = weight_sum_by_class.get(int(k), 0.0) + float(den_k)
                    count += 1
            if count > 0:
                print("Adjusted loss components (means across sequences):")
                print(f"  total={total / count:.4f}  CE={ce_total / count:.4f}  entropy={ent_total / count:.4f}  fp_penalty={fp_total / count:.4f}")
                # Per-class CE means and shares (aggregated across all sequences)
                if weight_sum_by_class:
                    print("CE per class (weighted means) and share of total CE:")
                    for k in sorted(ce_weighted_sum_by_class.keys()):
                        denom = weight_sum_by_class.get(k, 0.0)
                        mean_k = (ce_weighted_sum_by_class[k] / denom) if denom > 0 else float('nan')
                        share_k = (ce_weighted_sum_by_class[k] / total_weighted_ce_sum) if total_weighted_ce_sum > 0 else 0.0
                        try:
                            name = GenePredictionClass.idx_to_cls.get(int(k), str(int(k)))
                        except Exception:
                            name = str(int(k))
                        print(f"  {name:>10s}: CE_mean={mean_k:.4f}  CE_share={share_k:.2%}")
        except Exception as ex:
            print(f"[warn] unable to compute loss components: {ex}")

    # Brier score on final results
    brier = compute_brier_scores(results, class_weights=cw, min_weight=1.0, event_only=True)
    print(f"Brier (overall): {brier.get('brier', 0.0):.4f}")
    by_cls = brier.get('brier_by_class', {})
    if by_cls:
        print("Brier by class:")
        for cls_idx in sorted(by_cls.keys()):
            name = GenePredictionClass.idx_to_cls.get(int(cls_idx), str(int(cls_idx)))
            print(f"  {name:>10s}: {float(by_cls[cls_idx]):.4f}")

    # Save results (FASTA + per-contig colored report)
    base_name = save_analysis_results(results, output_dir, class_weights=cw, line_width=args.line_width, ansi_colors=args.ansi_colors, events=events)

    # Optionally write decoder input pickle
    if args.write_decoder_input_pkl:
        class_order = [GenePredictionClass.idx_to_cls[i] for i in sorted(GenePredictionClass.idx_to_cls.keys())]
        items = []
        for r in results:
            seq = convert_tokens_to_sequence(r['sequence_tokens'])
            items.append(PredictedSequence(
                sequence_index=r['sequence_index'],
                sequence=seq,
                probabilities=r['probabilities'],
                class_order=class_order,
            ))
        pkl_path = Path(f"{base_name}_decoder.pickle")
        if not pkl_path.is_absolute():
            pkl_path = output_dir / pkl_path
        with open(pkl_path, 'wb') as f:
            pickle.dump(items, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"✓ Decoder input pickle written to: {pkl_path}")

    # Dump attention fragments to FASTA
    attn_fa = output_dir / f"{base_name}_attn.fa"
    dump_attention_fragments(results, events, attn_fa, k=args.dump_attention_k, window=args.dump_attention_window)
    print(f"✓ Attention fragments written to: {attn_fa}")
    
    # Print generic per-class metrics (for classes selected above)
    if generic:
        print("\nPer-class metrics:")
        for cls_idx in sorted(generic.keys()):
            name = GenePredictionClass.idx_to_cls.get(int(cls_idx), str(cls_idx))
            m = generic[cls_idx]
            print(f"  {name:>10s}  TP={m['tp']} FP={m['fp']} FN={m['fn']}  "
                  f"Sensitivity={m['sensitivity']:.1%} Precision={m['precision']:.1%} Specificity={m['specificity']:.1%}")


if __name__ == "__main__":
    main()

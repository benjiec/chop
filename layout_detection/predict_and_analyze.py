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

from utils.dataset import (
    GenomicSyntheticTestingDataset,
    RandomBasesGenerator,
    RandomUTR5Generator,
    AddATGGenerator,
)
from utils.sequences import KOZAK_SEQUENCES, UTR5_REAL_SEQUENCES, IRES_SEQUENCES
from utils.constants import GenePredictionClass as P
from gene_predictor.model import GenePredictorModule as ModelModule
from torch.utils.data import DataLoader

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

def generate_test_data(num_sequences: int, max_seq_length: int, layouts_per_contig: int = 1):
    """Generate fresh test data aligned to the model's max sequence length."""
    print(f"Generating {num_sequences} test sequences...")

    # Choose background length conservatively so total contig length stays ≤ max_seq_length
    # The layout uses four background half-segments around UTR/ATG; cap by model limit
    background_len = min(450, max_seq_length // 2)
    utr_choices = KOZAK_SEQUENCES + UTR5_REAL_SEQUENCES + IRES_SEQUENCES
    layouts = [
        [
            RandomBasesGenerator(length=background_len // 2, target=P.INTERGENIC, random_min_length=background_len // 4),
            RandomBasesGenerator(length=background_len // 2, target=P.INTERGENIC, random_min_length=background_len // 4, avoid="ATG"),
            RandomUTR5Generator(choices=utr_choices, target=P.UTR5, mutation_prob=0.1),
            AddATGGenerator(),
            RandomBasesGenerator(length=background_len // 2, target=P.INTERGENIC, random_min_length=background_len // 4, avoid="ATG"),
            RandomBasesGenerator(length=background_len // 2, target=P.INTERGENIC, decoy="ATG", max_decoy=3, random_min_length=background_len // 4),
        ],
        # adding a second layout that are just negatives with decoys to test FP
        [ 
            RandomBasesGenerator(length=background_len // 2, target=P.INTERGENIC, random_min_length=background_len // 4),
            RandomBasesGenerator(length=background_len, target=P.INTERGENIC, decoy="ATG", max_decoy=3),
            RandomBasesGenerator(length=background_len // 2, target=P.INTERGENIC, random_min_length=background_len // 4),
        ]
    ]

    dataset = GenomicSyntheticTestingDataset(
        max_sequence_length=max_seq_length,
        num_contigs=num_sequences,
        layouts_per_contig=1,
        layouts=layouts,
    )

    # Sanity check 
    for contig_idx in range(dataset.num_contigs):
        full_sequence = dataset.contigs[contig_idx]
        full_targets = dataset.contig_targets[contig_idx]
        
        utr5_positions = np.sum(full_targets == 1)
        total_atgs = 0
        real_start_atgs = 0
        real_start_coords = []
        for i in range(len(full_sequence) - 2):
            if full_sequence[i:i+3] == 'ATG':
                total_atgs += 1
                if full_targets[i] == 2:  # Check if this ATG is labeled as START
                    if i > 0 and full_targets[i-1] != 2:
                        real_start_coords.append(i)
                    real_start_atgs += 1
        
        print(f"contig {contig_idx}: {real_start_atgs} real START ATGs ({real_start_coords}), {utr5_positions} UTR5 positions, {total_atgs} total ATGs, {len(full_sequence)} bps")
        assert real_start_atgs == layouts_per_contig or real_start_atgs == 0

    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    print(f"✓ Generated {len(dataset)} test windows")
    return data_loader, dataset

def run_predictions(model, data_loader, device='cpu'):
    """Run predictions on test data."""
    
    print("Running predictions on test data...")
    
    all_results = []
    
    with torch.no_grad():
        for batch_idx, (sequences, targets) in enumerate(data_loader):
            sequences = sequences.to(device)
            targets = targets.to(device)
            
            # Run model
            outputs = model(sequences)
            
            # Handle both dictionary and tensor outputs
            if isinstance(outputs, dict):
                gene_boundaries = outputs['gene_boundaries']
            else:
                # Model returns tensor directly
                gene_boundaries = outputs
            
            predictions = torch.argmax(gene_boundaries, dim=-1)
            probabilities = torch.softmax(gene_boundaries, dim=-1)
            
            # Convert to numpy for analysis
            seq_np = sequences[0].cpu().numpy()
            target_np = targets[0].cpu().numpy()
            pred_np = predictions[0].cpu().numpy()
            prob_np = probabilities[0].cpu().numpy()
            
            all_results.append({
                'sequence_index': batch_idx,
                'sequence_tokens': seq_np,
                'targets': target_np,
                'predictions': pred_np,
                'probabilities': prob_np
            })
            
            if (batch_idx + 1) % 10 == 0:
                print(f"  Processed {batch_idx + 1} sequences...")
    
    print(f"✓ Completed predictions for {len(all_results)} sequences")
    return all_results

def convert_tokens_to_sequence(tokens):
    """Convert token indices back to DNA sequence."""
    idx_to_nucleotide = {0: 'A', 1: 'T', 2: 'G', 3: 'C', 4: 'N'}
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
                target_name = {0: 'INTERGENIC', 1: 'UTR5', 2: 'START', -1: 'UNKNOWN'}[int(target_class)]

                prediction_data = {
                    'sequence_index': seq_idx,
                    'region_positions': [pos],
                    'atg_position': pos,
                    'actual_codon': 'ATG',
                    'is_true_positive': (classification == 'TP'),
                    'classification': classification,
                    'target_class': target_name,
                    'start_probability': float(prob_pos),
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
            prob = tp['start_probability']
            
            # Get full context from original sequence
            for result in results_data:
                if result['sequence_index'] == seq_idx:
                    sequence = convert_tokens_to_sequence(result['sequence_tokens'])
                    upstream_60 = sequence[max(0, pos-60):pos]
                    codon = sequence[pos:pos+3]
                    downstream_20 = sequence[pos+3:pos+23]
                    
                    # Create header with sequence name and position
                    header = f">sequence_{seq_idx}@{pos}"
                    
                    # Create visual line
                    context_line = upstream_60 + codon + downstream_20
                    marker_line = " " * len(upstream_60) + f"^^^ TP {prob:.2f}"

                    # Per-position targets/predictions for window [-60..+20]
                    start_idx = max(0, pos - 60)
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
            prob = fp['start_probability']
            
            # Get full context from original sequence
            for result in results_data:
                if result['sequence_index'] == seq_idx:
                    sequence = convert_tokens_to_sequence(result['sequence_tokens'])
                    upstream_60 = sequence[max(0, pos-60):pos]
                    codon = sequence[pos:pos+3]
                    downstream_20 = sequence[pos+3:pos+23]
                    
                    # Create header with sequence name and position
                    header = f">sequence_{seq_idx}@{pos}"
                    
                    # Create visual line
                    context_line = upstream_60 + codon + downstream_20
                    marker_line = " " * len(upstream_60) + f"^^^ FP {prob:.2f}"

                    # Per-position targets/predictions for window [-60..+20]
                    start_idx = max(0, pos - 60)
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
            prob = fn['start_probability']
            
            # Get full context from original sequence
            for result in results_data:
                if result['sequence_index'] == seq_idx:
                    sequence = convert_tokens_to_sequence(result['sequence_tokens'])
                    upstream_60 = sequence[max(0, pos-60):pos]
                    codon = sequence[pos:pos+3]
                    downstream_20 = sequence[pos+3:pos+23]
                    
                    # Create header with sequence name and position
                    header = f">sequence_{seq_idx}@{pos}"
                    
                    # Create visual line
                    context_line = upstream_60 + codon + downstream_20
                    marker_line = " " * len(upstream_60) + f"^^^ FN {prob:.2f}"

                    # Per-position targets/predictions for window [-60..+20]
                    start_idx = max(0, pos - 60)
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


# -----------------------------
# Breakdown (integrated)
# -----------------------------

def _build_parent_catalog() -> List[Dict]:
    catalog: List[Dict] = []
    for cat, seqs in (
        ('KOZAK', KOZAK_SEQUENCES),
        ('UTR5', UTR5_REAL_SEQUENCES),
        ('IRES', IRES_SEQUENCES),
    ):
        for idx, parent in enumerate(seqs):
            parent_up = parent[:-3] if parent.endswith('ATG') else parent
            catalog.append({
                'category': cat,
                'parent_index': idx,
                'parent_sequence': parent,
                'parent_upstream': parent_up,
                'parent_len': len(parent),
            })
    return catalog


def _hamming_distance(a: str, b: str) -> int:
    assert len(a) == len(b)
    return sum(1 for x, y in zip(a, b) if x != y)


def _match_upstream_to_parent(upstream_60: str, catalog: List[Dict]) -> Tuple[Optional[Dict], float, int]:
    best = None
    best_rate = 1.0
    best_len = 0
    for entry in catalog:
        parent_up = entry['parent_upstream']
        if not parent_up:
            continue
        L = min(len(parent_up), len(upstream_60), 60)
        if L == 0:
            continue
        a = upstream_60[-L:]
        b = parent_up[-L:]
        dist = _hamming_distance(a, b)
        rate = dist / L
        if rate < best_rate or (rate == best_rate and L > best_len):
            best = entry
            best_rate = rate
            best_len = L
    return best, best_rate, best_len


def _write_breakdown_tsv(output_path: Path, rows: List[Dict]):
    header = [
        'category', 'parent_index', 'parent_sequence', 'parent_len',
        'match_len', 'mismatch_rate', 'tp_count', 'fn_count', 'tp_mutated', 'fn_mutated'
    ]
    with open(output_path, 'w') as f:
        f.write('\t'.join(header) + '\n')
        for r in rows:
            f.write('\t'.join([
                str(r['category']),
                str(r['parent_index']),
                str(r['parent_sequence']),
                str(r['parent_len']),
                str(r['match_len']),
                f"{r['mismatch_rate']:.3f}",
                str(r['tp_count']),
                str(r['fn_count']),
                str(r['tp_mutated']),
                str(r['fn_mutated']),
            ]) + '\n')


def _build_entries_from_predictions(predictions: List[Dict], results_data: List[Dict]) -> List[Dict]:
    # Map sequence indices to raw sequence strings for quick slicing
    seq_map = {r['sequence_index']: convert_tokens_to_sequence(r['sequence_tokens']) for r in results_data}
    entries: List[Dict] = []
    for p in predictions:
        if p.get('classification') not in ('TP', 'FN'):
            continue
        seq_idx = p['sequence_index']
        pos = p['atg_position']
        sequence = seq_map[seq_idx]
        upstream_60 = sequence[max(0, pos-60):pos]
        codon = sequence[pos:pos+3]
        downstream_20 = sequence[pos+3:pos+23]
        entries.append({
            'sequence_id': f'sequence_{seq_idx}',
            'position': str(pos),
            'classification': p['classification'],
            'upstream_60': upstream_60,
            'codon': codon,
            'downstream_20': downstream_20,
        })
    return entries


def generate_breakdown_tsv(predictions: List[Dict], results_data: List[Dict], output_report_path: Path, mismatch_threshold: float = 0.15) -> Path:
    """Create breakdown TSV next to the report using prediction data directly."""
    entries = _build_entries_from_predictions(predictions, results_data)
    catalog = _build_parent_catalog()
    aggregates: Dict[Tuple[str, int], Dict] = {}

    for e in entries:
        best, rate, match_len = _match_upstream_to_parent(e['upstream_60'], catalog)
        if best is None or rate > mismatch_threshold:
            key = ('Unknown', -1)
            if key not in aggregates:
                aggregates[key] = {
                    'category': 'Unknown',
                    'parent_index': -1,
                    'parent_sequence': '',
                    'parent_len': 0,
                    'match_len': 0,
                    'mismatch_rate': 1.0,
                    'tp_count': 0,
                    'fn_count': 0,
                    'tp_mutated': 0,
                    'fn_mutated': 0,
                }
            agg = aggregates[key]
        else:
            key = (best['category'], best['parent_index'])
            if key not in aggregates:
                aggregates[key] = {
                    'category': best['category'],
                    'parent_index': best['parent_index'],
                    'parent_sequence': best['parent_sequence'],
                    'parent_len': best['parent_len'],
                    'match_len': match_len,
                    'mismatch_rate': rate,
                    'tp_count': 0,
                    'fn_count': 0,
                    'tp_mutated': 0,
                    'fn_mutated': 0,
                }
            agg = aggregates[key]
            if rate < agg['mismatch_rate'] or (rate == agg['mismatch_rate'] and match_len > agg['match_len']):
                agg['mismatch_rate'] = rate
                agg['match_len'] = match_len

        if e['classification'] == 'TP':
            agg['tp_count'] += 1
            if best is None or rate > 0:
                agg['tp_mutated'] += 1
        elif e['classification'] == 'FN':
            agg['fn_count'] += 1
            if best is None or rate > 0:
                agg['fn_mutated'] += 1

    rows = list(aggregates.values())
    rows.sort(key=lambda r: (r['category'], -(r['tp_count'] + r['fn_count'])))

    out_path = output_report_path.with_name(output_report_path.stem + '_breakdown.tsv')
    _write_breakdown_tsv(out_path, rows)
    print(f"✓ Breakdown written to: {out_path}")
    return out_path

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

def print_summary(predictions: List[Dict], metrics: Dict):
    """Print concise summary of position-level metrics only."""
    print(f"\n{'='*60}")
    print("COMPREHENSIVE START PREDICTION ANALYSIS")
    print(f"{'='*60}")
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
    parser.add_argument('--mismatch-threshold', type=float, default=0.15,
                       help='Max mismatch rate (0..1) for parent assignment in breakdown TSV')
    
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir) if args.output_dir else run_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("🧬 START Prediction Analysis")
    print("=" * 50)
    
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
    data_loader, dataset = generate_test_data(args.num_sequences, model_max_len)
    
    # Run predictions
    results = run_predictions(model, data_loader, args.device)
    
    # Calculate position-level metrics
    metrics = calculate_metrics(results)
    
    # Analyze predictions
    predictions = analyze_all_predictions(results)
    
    # Save results (FASTA + visual report)
    save_analysis_results(predictions, metrics, results, output_dir)
    # Generate breakdown TSV next to the report
    # Use the last generated report path (deterministic naming inside save function)
    # Recompute the same base name to know the report location
    # Note: generate_breakdown_tsv expects the report path to add _breakdown.tsv
    # We reuse the most recent report by listing matching files
    reports = sorted(output_dir.glob('prediction_*[0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f].txt'))
    if reports:
        last_report = reports[-1]
        generate_breakdown_tsv(predictions, results, last_report, mismatch_threshold=args.mismatch_threshold)
    
    # Print summary
    print_summary(predictions, metrics)
    
    print(f"\n✅ Analysis complete! Results saved to: {output_dir}")

if __name__ == "__main__":
    main()

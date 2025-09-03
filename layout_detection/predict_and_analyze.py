#!/usr/bin/env python3
"""
Fresh START prediction analysis with new test data.

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
from typing import List, Dict, Tuple
from datetime import datetime
import hashlib

# Add project paths
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "layout_detection"))

from layout_detection.utr_start_dataset import UTRStartDataset
from layout_detection.layout_model import LayoutDetectionModule
from torch.utils.data import DataLoader

def load_trained_model(model_path: Path, device='cpu'):
    """Load the trained model from checkpoint, restoring exact architecture."""
    print(f"Loading model from: {model_path}")

    # First, try Lightning's native loader which restores saved hyperparameters
    try:
        model = LayoutDetectionModule.load_from_checkpoint(model_path, map_location=device)
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

        model = LayoutDetectionModule(config)

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

def generate_test_data(num_sequences: int = 50, layouts_per_contig: int = 2):
    """Generate fresh test data for analysis."""
    
    print(f"Generating {num_sequences} test sequences...")
    
    # Create dataset
    dataset = UTRStartDataset(
        num_contigs=num_sequences,
        layouts_per_contig=layouts_per_contig,
        background_length=500,
        window_size=2000,
        window_stride=2000
    )
    
    # Create data loader
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
    """Calculate sensitivity, precision, and specificity for ATG-based START detection."""
    
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
            target_class = targets[pos]
            pred_class = predictions[pos]
            
            # Check if this position actually contains ATG
            if pos + 2 < len(sequence):
                codon = sequence[pos:pos+3]
                is_atg_position = codon == 'ATG'
                
                if is_atg_position:
                    # This is an actual ATG position
                    if pred_class == 2 and target_class == 2:  # Predicted START, actually START
                        tp_positions += 1
                    elif pred_class == 2 and target_class != 2:  # Predicted START, not actually START
                        fp_positions += 1
                    elif pred_class != 2 and target_class == 2:  # Didn't predict START, but actually START
                        fn_positions += 1
                    elif pred_class != 2 and target_class != 2:  # Didn't predict START, not actually START
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
    """Analyze all START predictions with sequence context using direct ATG-based approach."""
    
    print("Analyzing all START predictions...")
    
    all_predictions = []
    
    for result in results_data:
        seq_idx = result['sequence_index']
        sequence = convert_tokens_to_sequence(result['sequence_tokens'])
        targets = result['targets']
        predictions = result['predictions']
        probabilities = result['probabilities']
        
        print(f"Sequence {seq_idx}: length {len(sequence)}")
        
        # Find all ATG positions and analyze each one
        atg_count = 0
        tp_count = 0
        fp_count = 0
        fn_count = 0
        seen_positions = set()
        
        for pos in range(len(sequence) - 2):
            if sequence[pos:pos+3] == 'ATG':
                atg_count += 1
                
                # Check target and prediction at this position
                target_class = targets[pos] if pos < len(targets) else -1
                pred_class = predictions[pos] if pos < len(predictions) else -1
                start_prob = probabilities[pos, 2] if pos < len(probabilities) else 0.0
                
                # Classify this ATG
                is_target_start = target_class == 2
                is_predicted_start = pred_class == 2
                
                # Only include ATGs that were either predicted as START or should be START
                if is_predicted_start or is_target_start:
                    # Ensure uniqueness per sequence/position
                    key = (seq_idx, pos)
                    if key in seen_positions:
                        raise ValueError(f"Duplicate ATG position detected for sequence {seq_idx} at position {pos}")
                    seen_positions.add(key)

                    is_tp = is_predicted_start and is_target_start
                    
                    if is_tp:
                        classification = "TP"
                        tp_count += 1
                    elif is_predicted_start and not is_target_start:
                        classification = "FP" 
                        fp_count += 1
                    elif not is_predicted_start and is_target_start:
                        classification = "FN"
                        fn_count += 1
                    else:
                        continue  # TN - skip
                    
                    # Get context
                    upstream = sequence[max(0, pos-20):pos]
                    downstream = sequence[pos+3:pos+23]
                    
                    # Analyze Kozak pattern
                    kozak_analysis = analyze_kozak_pattern(upstream, 'ATG')
                    
                    # Determine target class name
                    target_name = {0: 'INTERGENIC', 1: 'UTR5', 2: 'START', -1: 'UNKNOWN'}[target_class]
                    
                    prediction_data = {
                        'sequence_index': seq_idx,
                        'region_positions': [pos],  # Single position for ATG
                        'atg_position': pos,
                        'actual_codon': 'ATG',
                        'is_true_positive': is_tp,
                        'classification': classification,
                        'target_class': target_name,
                        'start_probability': float(start_prob),
                        'upstream_20': upstream,
                        'downstream_20': downstream,
                        'kozak_score': kozak_analysis['score'],
                        'kozak_features': kozak_analysis['features'],
                        'region_length': 1
                    }
                    
                    all_predictions.append(prediction_data)
                    
                    print(f"    ATG at {pos}: {classification}, target: {target_name}, prob: {start_prob:.3f}")
        
        print(f"  Total ATGs: {atg_count}, Analyzed: TP={tp_count}, FP={fp_count}, FN={fn_count}")
    
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
                    
                    f.write(f"{header}\n")
                    f.write(f"{context_line}\n")
                    f.write(f"{marker_line}\n\n")
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
                    
                    f.write(f"{header}\n")
                    f.write(f"{context_line}\n")
                    f.write(f"{marker_line}\n\n")
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
                    
                    f.write(f"{header}\n")
                    f.write(f"{context_line}\n")
                    f.write(f"{marker_line}\n\n")
                    break
    
    print(f"✓ Visual analysis saved to: {output_path}")

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

def main():
    parser = argparse.ArgumentParser(description='Fresh START prediction analysis')
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Output directory for analysis results')
    parser.add_argument('--num-sequences', type=int, default=50,
                       help='Number of test sequences to generate')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device to run on (cpu/cuda)')
    
    args = parser.parse_args()
    
    model_path = Path(args.model_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("🧬 Fresh START Prediction Analysis")
    print("=" * 50)
    
    # Load model
    model = load_trained_model(model_path, args.device)
    if model is None:
        return
    
    # Generate test data
    data_loader, dataset = generate_test_data(args.num_sequences)
    
    # Run predictions
    results = run_predictions(model, data_loader, args.device)
    
    # Calculate position-level metrics
    metrics = calculate_metrics(results)
    
    # Analyze predictions
    predictions = analyze_all_predictions(results)
    
    # Save results
    save_analysis_results(predictions, metrics, results, output_dir)
    
    # Print summary
    print_summary(predictions, metrics)
    
    print(f"\n✅ Analysis complete! Results saved to: {output_dir}")

if __name__ == "__main__":
    main()

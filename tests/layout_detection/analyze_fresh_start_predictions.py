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

# Add project paths
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "tests" / "layout_detection"))

from tests.layout_detection.utr_start_dataset import UTRStartDataset
from tests.layout_detection.layout_model import LayoutDetectionModule
from torch.utils.data import DataLoader

def load_trained_model(model_path: Path, device='cpu'):
    """Load the trained model from checkpoint."""
    
    print(f"Loading model from: {model_path}")
    
    try:
        # Load the checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        
        # Create model instance matching the saved configuration
        from tests.layout_detection.test_utr_start_controlled import create_utr_start_config
        config = create_utr_start_config(
            d_model=504, n_layers=4, n_heads=6,
            attention_masks={0: 4, 1: (20, 5), 2: (50, 0)},
            kmer_size=0  # No k-mer convolution based on hparams
        )
        
        model = LayoutDetectionModule(config)
        
        # Load state dict, filtering out non-model parameters
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # Filter out non-model keys (like criterion weights)
        model_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('model.'):
                model_state_dict[key] = value
            elif not key.startswith('criterion'):
                model_state_dict[key] = value
        
        model.load_state_dict(model_state_dict)
        
        model.eval()
        model = model.to(device)
        
        print("✓ Model loaded successfully")
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

def find_start_regions(predictions):
    """Group consecutive START predictions into regions."""
    
    start_regions = []
    
    # Find all positions predicted as START (class 2)
    start_positions = [i for i, pred in enumerate(predictions) if pred == 2]
    
    if not start_positions:
        return start_regions
    
    # Group consecutive positions
    current_region = [start_positions[0]]
    
    for pos in start_positions[1:]:
        if pos == current_region[-1] + 1:
            current_region.append(pos)
        else:
            # End current region, start new one
            start_regions.append(current_region)
            current_region = [pos]
    
    # Add the last region
    start_regions.append(current_region)
    
    return start_regions

def calculate_metrics(all_predictions):
    """Calculate sensitivity, precision, and specificity."""
    
    # Count position-level metrics (not region-level)
    tp_positions = 0
    fp_positions = 0
    fn_positions = 0
    tn_positions = 0
    
    for result in all_predictions:
        targets = result['targets']
        predictions = result['predictions']
        
        for pos in range(len(targets)):
            target_class = targets[pos]
            pred_class = predictions[pos]
            
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
    """Analyze all START predictions with sequence context."""
    
    print("Analyzing all START predictions...")
    
    all_predictions = []
    
    for result in results_data:
        seq_idx = result['sequence_index']
        sequence = convert_tokens_to_sequence(result['sequence_tokens'])
        targets = result['targets']
        predictions = result['predictions']
        probabilities = result['probabilities']
        
        # Find START regions in predictions
        start_regions = find_start_regions(predictions)
        
        # Find actual START regions in targets
        target_start_regions = find_start_regions(targets)
        
        # Analyze each predicted START region
        for region_idx, region in enumerate(start_regions):
            region_start = min(region)
            region_end = max(region)
            region_center = (region_start + region_end) // 2
            
            # Determine if this is a true positive
            # Check if this region overlaps with any target START region
            is_tp = False
            overlapping_target = None
            
            for target_region in target_start_regions:
                target_start = min(target_region)
                target_end = max(target_region)
                
                # Check for overlap
                if not (region_end < target_start or region_start > target_end):
                    is_tp = True
                    overlapping_target = target_region
                    break
            
            # Find the ATG in this region or nearby
            atg_position = None
            actual_codon = None
            
            # Look for ATG within the predicted region and nearby
            search_start = max(0, region_start - 3)
            search_end = min(len(sequence) - 2, region_end + 3)
            
            for pos in range(search_start, search_end):
                if pos + 2 < len(sequence):
                    codon = sequence[pos:pos+3]
                    if codon == 'ATG':
                        atg_position = pos
                        actual_codon = codon
                        break
            
            # If no ATG found, use the center position
            if atg_position is None:
                atg_position = region_center
                if atg_position + 2 < len(sequence):
                    actual_codon = sequence[atg_position:atg_position+3]
                else:
                    actual_codon = 'N/A'
            
            # Get context around the ATG/center position
            upstream = sequence[max(0, atg_position-20):atg_position]
            downstream = sequence[atg_position+3:atg_position+23]
            
            # Analyze Kozak pattern
            kozak_analysis = analyze_kozak_pattern(upstream, actual_codon)
            
            # Note: We don't analyze downstream stop codons since we're not handling splicing yet
            
            # Determine target class at this position
            if atg_position < len(targets):
                target_class = targets[atg_position]
                target_name = {0: 'INTERGENIC', 1: 'UTR5', 2: 'START'}[target_class]
            else:
                target_class = -1
                target_name = 'UNKNOWN'
            
            # Get START probability
            if atg_position < len(probabilities):
                start_prob = probabilities[atg_position, 2]  # START class probability
            else:
                start_prob = 0.0
            
            prediction_data = {
                'sequence_index': seq_idx,
                'region_positions': region,
                'atg_position': atg_position,
                'actual_codon': actual_codon,
                'is_true_positive': is_tp,
                'target_class': target_name,
                'start_probability': float(start_prob),
                'upstream_20': upstream,
                'downstream_20': downstream,
                'kozak_score': kozak_analysis['score'],
                'kozak_features': kozak_analysis['features'],
                'region_length': len(region)
            }
            
            all_predictions.append(prediction_data)
    
    return all_predictions

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
    
    # Collect all predictions and missed targets
    tps = [p for p in predictions if p['is_true_positive']]
    fps = [p for p in predictions if not p['is_true_positive']]
    
    # Find false negatives (missed STARTs)
    fns = []
    for result in results_data:
        seq_idx = result['sequence_index']
        sequence = convert_tokens_to_sequence(result['sequence_tokens'])
        targets = result['targets']
        predictions_array = result['predictions']
        probabilities = result['probabilities']
        
        # Find actual START positions that were missed
        for pos in range(len(targets)):
            if targets[pos] == 2 and predictions_array[pos] != 2:  # Actual START, not predicted
                # Get context
                upstream = sequence[max(0, pos-60):pos]
                downstream = sequence[pos+3:pos+23]
                
                # Get START probability
                start_prob = probabilities[pos, 2]
                
                fns.append({
                    'sequence_index': seq_idx,
                    'position': pos,
                    'sequence': sequence,
                    'start_probability': float(start_prob),
                    'upstream_60': upstream,
                    'downstream_20': downstream
                })
    
    with open(output_path, 'w') as f:
        f.write("START Prediction Visual Analysis\n")
        f.write("=" * 80 + "\n\n")
        
        # True Positives
        f.write("TRUE POSITIVES:\n")
        f.write("-" * 40 + "\n")
        for i, tp in enumerate(tps[:10]):  # Show first 10
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
                    
                    # Create visual line
                    context_line = upstream_60 + codon + downstream_20
                    marker_line = " " * len(upstream_60) + f"^^^ TP {prob:.2f}"
                    
                    f.write(f"{context_line}\n")
                    f.write(f"{marker_line}\n\n")
                    break
        
        # False Positives
        f.write("\nFALSE POSITIVES:\n")
        f.write("-" * 40 + "\n")
        for i, fp in enumerate(fps[:10]):  # Show first 10
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
                    
                    # Create visual line
                    context_line = upstream_60 + codon + downstream_20
                    marker_line = " " * len(upstream_60) + f"^^^ FP {prob:.2f}"
                    
                    f.write(f"{context_line}\n")
                    f.write(f"{marker_line}\n\n")
                    break
        
        # False Negatives
        f.write("\nFALSE NEGATIVES (Missed STARTs):\n")
        f.write("-" * 40 + "\n")
        for i, fn in enumerate(fns[:10]):  # Show first 10
            pos = fn['position']
            prob = fn['start_probability']
            sequence = fn['sequence']
            
            upstream_60 = sequence[max(0, pos-60):pos]
            codon = sequence[pos:pos+3]
            downstream_20 = sequence[pos+3:pos+23]
            
            # Create visual line
            context_line = upstream_60 + codon + downstream_20
            marker_line = " " * len(upstream_60) + f"^^^ FN {prob:.2f}"
            
            f.write(f"{context_line}\n")
            f.write(f"{marker_line}\n\n")
    
    print(f"✓ Visual analysis saved to: {output_path}")

def save_analysis_results(predictions: List[Dict], metrics: Dict, results_data: List[Dict], output_dir: Path):
    """Save analysis results with timestamped filenames."""
    
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save input sequences as FASTA
    fasta_output = output_dir / f"prediction_{timestamp}_input.fa"
    save_input_sequences_fasta(results_data, fasta_output)
    
    # Generate visual output
    report_output = output_dir / f"prediction_{timestamp}_report.txt"
    generate_visual_output(predictions, results_data, report_output)

def print_summary(predictions: List[Dict], metrics: Dict):
    """Print comprehensive summary of results."""
    
    total_preds = len(predictions)
    tps = [p for p in predictions if p['is_true_positive']]
    fps = [p for p in predictions if not p['is_true_positive']]
    
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
    
    print(f"\nREGION-LEVEL RESULTS:")
    print(f"  Total START regions predicted: {total_preds}")
    print(f"  True positive regions: {len(tps)}")
    print(f"  False positive regions: {len(fps)}")
    print(f"  Region precision: {len(tps)/total_preds:.1%}")
    
    if fps:
        print(f"\nFALSE POSITIVE ANALYSIS:")
        
        # By target class
        fp_by_target = {}
        for fp in fps:
            target = fp['target_class']
            if target not in fp_by_target:
                fp_by_target[target] = []
            fp_by_target[target].append(fp)
        
        for target, target_fps in fp_by_target.items():
            print(f"  {target}: {len(target_fps)} false positives")
        
        # Codon analysis
        atg_fps = [fp for fp in fps if fp['actual_codon'] == 'ATG']
        non_atg_fps = [fp for fp in fps if fp['actual_codon'] != 'ATG']
        
        print(f"\n  CODON ANALYSIS:")
        print(f"    ATG codons: {len(atg_fps)} ({len(atg_fps)/len(fps):.1%})")
        print(f"    Non-ATG codons: {len(non_atg_fps)} ({len(non_atg_fps)/len(fps):.1%})")
        
        if non_atg_fps:
            non_atg_codons = [fp['actual_codon'] for fp in non_atg_fps]
            unique_codons = sorted(list(set(non_atg_codons)))
            print(f"      Non-ATG codons: {unique_codons}")
        
        # Kozak analysis
        kozak_fps = [fp for fp in fps if fp['kozak_features'].get('minus3_purine', False)]
        print(f"\n  KOZAK ANALYSIS:")
        print(f"    With purine at -3: {len(kozak_fps)} ({len(kozak_fps)/len(fps):.1%})")
        
        avg_kozak_fp = sum(fp['kozak_score'] for fp in fps) / len(fps)
        avg_kozak_tp = sum(tp['kozak_score'] for tp in tps) / len(tps) if tps else 0
        
        print(f"    Average Kozak score (FP): {avg_kozak_fp:.2f}")
        print(f"    Average Kozak score (TP): {avg_kozak_tp:.2f}")
        
        # Note: Downstream stop analysis removed since we're not handling splicing
        
        # Show examples
        print(f"\n  EXAMPLE FALSE POSITIVES:")
        for i, fp in enumerate(fps[:5]):
            print(f"    FP {i+1}: pos {fp['atg_position']}, codon {fp['actual_codon']}, target {fp['target_class']}")
            print(f"      Context: {fp['upstream_20'][-10:]}[{fp['actual_codon']}]{fp['downstream_20'][:10]}")
            print(f"      Kozak score: {fp['kozak_score']}, START prob: {fp['start_probability']:.4f}")
    
    if tps:
        print(f"\n  TRUE POSITIVE EXAMPLES:")
        for i, tp in enumerate(tps[:3]):
            print(f"    TP {i+1}: pos {tp['atg_position']}, codon {tp['actual_codon']}")
            print(f"      Context: {tp['upstream_20'][-10:]}[{tp['actual_codon']}]{tp['downstream_20'][:10]}")
            print(f"      Kozak score: {tp['kozak_score']}, START prob: {tp['start_probability']:.4f}")

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

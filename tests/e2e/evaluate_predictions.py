#!/usr/bin/env python3
"""
Evaluate gene prediction accuracy by comparing predictions with ground truth annotations.

Calculates sensitivity and specificity for both exons and genes.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Set
from collections import defaultdict
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def parse_gff_annotations(gff_file: str) -> Dict[str, Dict]:
    """Parse GFF file to extract gene and exon annotations."""
    annotations = defaultdict(lambda: {'genes': [], 'exons': []})
    
    with open(gff_file, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
                
            fields = line.strip().split('\t')
            if len(fields) < 9:
                continue
                
            seq_id, source, feature_type, start, end, score, strand, phase, attributes = fields
            start, end = int(start) - 1, int(end)  # Convert to 0-based coordinates
            
            if feature_type == 'gene':
                annotations[seq_id]['genes'].append({
                    'start': start,
                    'end': end,
                    'strand': strand,
                    'id': attributes.split(';')[0].split('=')[1]
                })
            elif feature_type == 'CDS':
                annotations[seq_id]['exons'].append({
                    'start': start,
                    'end': end,
                    'strand': strand,
                    'parent': attributes.split(';')[1].split('=')[1]
                })
    
    return dict(annotations)


def load_predictions(predictions_file: str) -> List[Dict]:
    """Load predictions from JSON file."""
    with open(predictions_file, 'r') as f:
        return json.load(f)


def calculate_overlap(range1: Tuple[int, int], range2: Tuple[int, int]) -> float:
    """Calculate overlap between two ranges as fraction of the smaller range."""
    start1, end1 = range1
    start2, end2 = range2
    
    overlap_start = max(start1, start2)
    overlap_end = min(end1, end2)
    overlap_length = max(0, overlap_end - overlap_start)
    
    range1_length = end1 - start1
    range2_length = end2 - start2
    
    if range1_length == 0 and range2_length == 0:
        return 1.0 if start1 == start2 else 0.0
    elif range1_length == 0 or range2_length == 0:
        return 0.0
    else:
        return overlap_length / min(range1_length, range2_length)


def evaluate_exons(true_exons: List[Dict], pred_exons: List[Dict], min_overlap: float = 0.5) -> Dict:
    """Evaluate exon predictions."""
    # Convert to simple ranges for easier comparison
    true_ranges = [(ex['start'], ex['end']) for ex in true_exons]
    pred_ranges = [(ex['start'], ex['end']) for ex in pred_exons]
    
    # Find matches
    matched_true = set()
    matched_pred = set()
    
    for i, true_range in enumerate(true_ranges):
        for j, pred_range in enumerate(pred_ranges):
            if j in matched_pred:  # Already matched
                continue
                
            overlap = calculate_overlap(true_range, pred_range)
            if overlap >= min_overlap:
                matched_true.add(i)
                matched_pred.add(j)
                break  # Each true exon matches at most one predicted exon
    
    true_positives = len(matched_true)
    false_negatives = len(true_ranges) - true_positives
    false_positives = len(pred_ranges) - len(matched_pred)
    
    sensitivity = true_positives / len(true_ranges) if true_ranges else 0.0
    specificity = true_positives / len(pred_ranges) if pred_ranges else 0.0
    
    return {
        'true_positives': true_positives,
        'false_negatives': false_negatives,
        'false_positives': false_positives,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'total_true': len(true_ranges),
        'total_predicted': len(pred_ranges)
    }


def evaluate_genes(true_genes: List[Dict], pred_genes: List[Dict], min_overlap: float = 0.5) -> Dict:
    """Evaluate gene predictions."""
    # Convert to simple ranges
    true_ranges = [(gene['start'], gene['end']) for gene in true_genes]
    pred_ranges = [(gene['start'], gene['end']) for gene in pred_genes]
    
    # Find matches
    matched_true = set()
    matched_pred = set()
    
    for i, true_range in enumerate(true_ranges):
        for j, pred_range in enumerate(pred_ranges):
            if j in matched_pred:  # Already matched
                continue
                
            overlap = calculate_overlap(true_range, pred_range)
            if overlap >= min_overlap:
                matched_true.add(i)
                matched_pred.add(j)
                break
    
    true_positives = len(matched_true)
    false_negatives = len(true_ranges) - true_positives
    false_positives = len(pred_ranges) - len(matched_pred)
    
    sensitivity = true_positives / len(true_ranges) if true_ranges else 0.0
    specificity = true_positives / len(pred_ranges) if pred_ranges else 0.0
    
    return {
        'true_positives': true_positives,
        'false_negatives': false_negatives,
        'false_positives': false_positives,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'total_true': len(true_ranges),
        'total_predicted': len(pred_ranges)
    }


def analyze_prediction_sizes(predictions: List[Dict]) -> Dict:
    """Analyze the distribution of prediction sizes."""
    exon_sizes = []
    gene_sizes = []
    
    for seq_pred in predictions:
        for exon in seq_pred.get('predictions', {}).get('exons', []):
            size = exon['end'] - exon['start']
            exon_sizes.append(size)
        
        for gene in seq_pred.get('predictions', {}).get('genes', []):
            size = gene['end'] - gene['start']
            gene_sizes.append(size)
    
    def get_stats(sizes):
        if not sizes:
            return {'count': 0, 'min': 0, 'max': 0, 'mean': 0, 'median': 0}
        sizes_sorted = sorted(sizes)
        return {
            'count': len(sizes),
            'min': min(sizes),
            'max': max(sizes),
            'mean': sum(sizes) / len(sizes),
            'median': sizes_sorted[len(sizes_sorted) // 2]
        }
    
    return {
        'exons': get_stats(exon_sizes),
        'genes': get_stats(gene_sizes)
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate gene prediction accuracy")
    parser.add_argument("--predictions", required=True, help="Path to predictions JSON file")
    parser.add_argument("--annotations", required=True, help="Path to ground truth GFF file")
    parser.add_argument("--min-overlap", type=float, default=0.5, help="Minimum overlap for match")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Load data
    print("Loading annotations and predictions...")
    annotations = parse_gff_annotations(args.annotations)
    predictions = load_predictions(args.predictions)
    
    # Analyze prediction sizes
    print("\n" + "="*60)
    print("PREDICTION SIZE ANALYSIS")
    print("="*60)
    
    size_stats = analyze_prediction_sizes(predictions)
    
    print(f"Exon predictions: {size_stats['exons']['count']:,} total")
    if size_stats['exons']['count'] > 0:
        print(f"  Size range: {size_stats['exons']['min']}-{size_stats['exons']['max']} bp")
        print(f"  Mean size: {size_stats['exons']['mean']:.1f} bp")
        print(f"  Median size: {size_stats['exons']['median']} bp")
    
    print(f"\nGene predictions: {size_stats['genes']['count']:,} total")
    if size_stats['genes']['count'] > 0:
        print(f"  Size range: {size_stats['genes']['min']}-{size_stats['genes']['max']} bp")
        print(f"  Mean size: {size_stats['genes']['mean']:.1f} bp")
        print(f"  Median size: {size_stats['genes']['median']} bp")
    
    # Count 1bp predictions
    total_exons = 0
    bp1_exons = 0
    for seq_pred in predictions:
        for exon in seq_pred.get('predictions', {}).get('exons', []):
            total_exons += 1
            if exon['end'] - exon['start'] == 1:
                bp1_exons += 1
    
    if total_exons > 0:
        print(f"\n⚠️  1bp exons: {bp1_exons:,} / {total_exons:,} ({bp1_exons/total_exons*100:.1f}%)")
    
    # Evaluate predictions for each sequence
    print("\n" + "="*60)
    print("ACCURACY EVALUATION")
    print("="*60)
    
    total_exon_metrics = {
        'true_positives': 0, 'false_negatives': 0, 'false_positives': 0,
        'total_true': 0, 'total_predicted': 0
    }
    
    total_gene_metrics = {
        'true_positives': 0, 'false_negatives': 0, 'false_positives': 0,
        'total_true': 0, 'total_predicted': 0
    }
    
    for seq_pred in predictions:
        seq_id = seq_pred['sequence_id']
        
        if seq_id not in annotations:
            print(f"Warning: No annotations found for sequence {seq_id}")
            continue
        
        # Get true annotations
        true_exons = annotations[seq_id]['exons']
        true_genes = annotations[seq_id]['genes']
        
        # Get predictions (handle nested structure)
        pred_data = seq_pred.get('predictions', seq_pred)
        pred_exons = pred_data.get('exons', [])
        pred_genes = pred_data.get('genes', [])
        
        # Evaluate exons
        exon_metrics = evaluate_exons(true_exons, pred_exons, args.min_overlap)
        
        # Evaluate genes  
        gene_metrics = evaluate_genes(true_genes, pred_genes, args.min_overlap)
        
        # Accumulate totals
        for key in total_exon_metrics:
            total_exon_metrics[key] += exon_metrics[key]
            
        for key in total_gene_metrics:
            total_gene_metrics[key] += gene_metrics[key]
        
        if args.verbose:
            print(f"\nSequence: {seq_id}")
            print(f"  Exons - True: {len(true_exons)}, Predicted: {len(pred_exons)}")
            print(f"    Sensitivity: {exon_metrics['sensitivity']:.3f}, Specificity: {exon_metrics['specificity']:.3f}")
            print(f"  Genes - True: {len(true_genes)}, Predicted: {len(pred_genes)}")
            print(f"    Sensitivity: {gene_metrics['sensitivity']:.3f}, Specificity: {gene_metrics['specificity']:.3f}")
    
    # Calculate overall metrics
    overall_exon_sensitivity = (total_exon_metrics['true_positives'] / 
                               total_exon_metrics['total_true'] if total_exon_metrics['total_true'] > 0 else 0)
    overall_exon_specificity = (total_exon_metrics['true_positives'] / 
                               total_exon_metrics['total_predicted'] if total_exon_metrics['total_predicted'] > 0 else 0)
    
    overall_gene_sensitivity = (total_gene_metrics['true_positives'] / 
                               total_gene_metrics['total_true'] if total_gene_metrics['total_true'] > 0 else 0)
    overall_gene_specificity = (total_gene_metrics['true_positives'] / 
                               total_gene_metrics['total_predicted'] if total_gene_metrics['total_predicted'] > 0 else 0)
    
    # Print summary
    print(f"\nOVERALL RESULTS (min overlap: {args.min_overlap})")
    print("-" * 50)
    
    print(f"EXONS:")
    print(f"  Total annotated: {total_exon_metrics['total_true']:,}")
    print(f"  Total predicted: {total_exon_metrics['total_predicted']:,}")
    print(f"  True positives: {total_exon_metrics['true_positives']:,}")
    print(f"  Sensitivity: {overall_exon_sensitivity:.3f} ({overall_exon_sensitivity*100:.1f}%)")
    print(f"  Specificity: {overall_exon_specificity:.3f} ({overall_exon_specificity*100:.1f}%)")
    
    print(f"\nGENES:")
    print(f"  Total annotated: {total_gene_metrics['total_true']:,}")
    print(f"  Total predicted: {total_gene_metrics['total_predicted']:,}")
    print(f"  True positives: {total_gene_metrics['true_positives']:,}")
    print(f"  Sensitivity: {overall_gene_sensitivity:.3f} ({overall_gene_sensitivity*100:.1f}%)")
    print(f"  Specificity: {overall_gene_specificity:.3f} ({overall_gene_specificity*100:.1f}%)")


if __name__ == "__main__":
    main()

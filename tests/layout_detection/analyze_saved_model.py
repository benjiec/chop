#!/usr/bin/env python3
"""
Analyze a saved UTR-START model checkpoint.

This script loads a saved model and runs comprehensive layer analysis
without needing to retrain.

Usage:
    python tests/layout_detection/analyze_saved_model.py --model-path tests/layout_detection/utr_start_test_run_20250831_064626/checkpoints/last.ckpt
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

import torch
from torch.utils.data import DataLoader, random_split
import argparse

from tests.layout_detection.utr_start_dataset import UTRStartDataset
from tests.layout_detection.layout_model import LayoutDetectionModule
from tests.layout_detection.layer_analysis import LayerAnalyzer

def analyze_saved_model(model_path: str, output_dir: str, 
                       num_contigs: int = 1000, layouts_per_contig: int = 1,
                       max_samples: int = 20):
    """Analyze a saved model checkpoint."""
    
    print(f"=" * 70)
    print(f"ANALYZING SAVED UTR-START MODEL")
    print(f"Model: {model_path}")
    print(f"Output: {output_dir}")
    print(f"=" * 70)
    
    # Check if we can use existing saved predictions
    model_dir = Path(model_path).parent.parent
    existing_predictions = model_dir / "detailed_predictions.json"
    existing_sequences = model_dir / "sample_data.json"
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if existing_predictions.exists() and existing_sequences.exists():
        print("\n1. Using existing saved predictions...")
        print(f"  Found: {existing_predictions}")
        print(f"  Found: {existing_sequences}")
        
        # Load existing data
        import json
        with open(existing_predictions) as f:
            predictions_data = json.load(f)
        with open(existing_sequences) as f:
            sequences_data = json.load(f)
        
        print(f"  Loaded {len(predictions_data)} windows with predictions")
        
        # Convert to analysis format
        analysis_data = convert_saved_data_to_analysis_format(predictions_data, sequences_data)
        
        # Load model for any model-specific analysis
        print("\n2. Loading model for feature analysis...")
        model = LayoutDetectionModule.load_from_checkpoint(model_path)
        model.eval()
        
        # Run analysis on existing data
        print("\n3. Running analysis on saved predictions...")
        run_analysis_on_saved_data(model, analysis_data, output_path, max_samples)
        
    else:
        print("\n1. No existing predictions found - recreating dataset...")
        print("  (This will take longer and may not match original training data)")
        
        # Fall back to original approach
        model = LayoutDetectionModule.load_from_checkpoint(model_path)
        model.eval()
        
        dataset = UTRStartDataset(
            num_contigs=num_contigs,
            layouts_per_contig=layouts_per_contig,
            background_length=500,
            window_size=1100,
            window_stride=1100
        )
        
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=0)
        
        analyzer = LayerAnalyzer(model)
        analyzer.analyze_all(val_loader, output_path, max_samples=max_samples)
    
    print(f"\n✅ Analysis complete! Results saved to: {output_path}")


def convert_saved_data_to_analysis_format(predictions_data, sequences_data):
    """Convert saved prediction data to analysis format."""
    
    analysis_data = []
    
    # Use the saved predictions and sequences
    for i, pred_window in enumerate(predictions_data[:20]):  # First 20 windows
        
        # Convert predictions and targets back to proper format
        predictions = pred_window['predictions']
        targets = pred_window['targets']
        
        # We need to reconstruct the sequence from the sample data
        # This is a limitation - we'd need to save sequences in the detailed predictions
        
        analysis_data.append({
            'window_index': i,
            'predictions': predictions,
            'targets': targets,
            'window_accuracy': pred_window['accuracy'],
            'start_predictions': pred_window.get('start_predictions', []),
            'start_targets': pred_window.get('start_targets', [])
        })
    
    return analysis_data


def run_analysis_on_saved_data(model, analysis_data, output_path, max_samples):
    """Run analysis using saved prediction data."""
    
    # Create summary from saved data
    total_samples = len(analysis_data)
    correct_predictions = sum(1 for item in analysis_data if item['window_accuracy'] > 0.9)
    
    # Save basic analysis
    summary = {
        'total_samples_analyzed': total_samples,
        'high_accuracy_windows': correct_predictions,
        'note': 'Analysis based on saved predictions - limited feature analysis available'
    }
    
    with open(output_path / "analysis_summary.json", 'w') as f:
        import json
        json.dump(summary, f, indent=2)
    
    # Copy existing predictions to analysis directory
    print(f"  Saved analysis summary")
    print(f"  Note: Limited analysis available from saved predictions")
    print(f"  For full layer analysis, run during training or with recreated dataset")


def main():
    parser = argparse.ArgumentParser(description="Analyze saved UTR-START model")
    parser.add_argument('--model-path', required=True, help='Path to saved model checkpoint')
    parser.add_argument('--output-dir', help='Output directory (default: model_path/../analysis)')
    parser.add_argument('--contigs', type=int, default=1000, help='Number of contigs (must match original)')
    parser.add_argument('--layouts', type=int, default=1, help='Layouts per contig (must match original)')
    parser.add_argument('--max-samples', type=int, default=20, help='Number of samples to analyze')
    
    args = parser.parse_args()
    
    # Default output directory
    if args.output_dir is None:
        from datetime import datetime
        model_dir = Path(args.model_path).parent.parent
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = str(model_dir / f"analysis_{timestamp}")
    
    # Run analysis
    analyze_saved_model(
        model_path=args.model_path,
        output_dir=args.output_dir,
        num_contigs=args.contigs,
        layouts_per_contig=args.layouts,
        max_samples=args.max_samples
    )


def main():
    parser = argparse.ArgumentParser(description="Analyze saved UTR-START model")
    parser.add_argument('--model-path', required=True, help='Path to saved model checkpoint')
    parser.add_argument('--output-dir', help='Output directory (default: model_path/../analysis)')
    parser.add_argument('--contigs', type=int, default=1000, help='Number of contigs (must match original)')
    parser.add_argument('--layouts', type=int, default=1, help='Layouts per contig (must match original)')
    parser.add_argument('--max-samples', type=int, default=20, help='Number of samples to analyze')
    
    args = parser.parse_args()
    
    # Default output directory
    if args.output_dir is None:
        from datetime import datetime
        model_dir = Path(args.model_path).parent.parent
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = str(model_dir / f"analysis_{timestamp}")
    
    # Run analysis
    analyze_saved_model(
        model_path=args.model_path,
        output_dir=args.output_dir,
        num_contigs=args.contigs,
        layouts_per_contig=args.layouts,
        max_samples=args.max_samples
    )


if __name__ == "__main__":
    main()

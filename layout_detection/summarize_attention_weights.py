#!/usr/bin/env python3
"""
Extract attention range data from training runs into TSV format.

Analyzes attention patterns across layers, heads, and epochs, generating:
- Structured TSV data with attention ranges and weights
- Dynamic range detection based on actual attention patterns
- Clustered position ranges for final model analysis
- Architecture-agnostic processing (auto-detects layers/heads)
"""

import json
import numpy as np
from pathlib import Path
import argparse
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

def _select_checkpoint(run_dir: Path) -> Optional[Path]:
    ckpt_dir = run_dir / 'checkpoints'
    if not ckpt_dir.exists():
        return None
    best = ckpt_dir / 'best.ckpt'
    if best.exists():
        return best
    # try lowest val encoded at end
    candidates = list(ckpt_dir.glob('*.ckpt'))
    best_path = None
    best_val = None
    import re
    for p in candidates:
        m = re.search(r'([0-9]+(?:\.[0-9]+)?)\.ckpt$', p.name)
        if not m:
            continue
        try:
            v = float(m.group(1))
        except ValueError:
            continue
        if best_val is None or v < best_val:
            best_val = v
            best_path = p
    if best_path is not None:
        return best_path
    last = ckpt_dir / 'last.ckpt'
    return last if last.exists() else None

def _load_attention_masks_from_ckpt(run_dir: Path) -> Dict[int, Any]:
    """Load per-head attention mask config from checkpoint hyperparameters, if present."""
    try:
        ckpt = _select_checkpoint(run_dir)
        if ckpt is None:
            return {}
        import torch
        data = torch.load(ckpt, map_location='cpu')
        for hp_key in ('hyper_parameters', 'hparams'):
            if hp_key in data:
                hp = data[hp_key]
                cfg = hp.get('config', hp) if isinstance(hp, dict) else None
                if isinstance(cfg, dict):
                    model_cfg = cfg.get('model', {})
                    am = model_cfg.get('attention_masks', {})
                    if isinstance(am, dict):
                        # keys may be strings
                        masks: Dict[int, Any] = {}
                        for k, v in am.items():
                            try:
                                ki = int(k)
                            except Exception:
                                continue
                            masks[ki] = tuple(v) if isinstance(v, list) else v
                        return masks
        return {}
    except Exception:
        return {}

def _mask_windows(mask_cfg: Any) -> Dict[str, Optional[Tuple[int, int]]]:
    """Map a head's mask config to upstream/local/downstream relative windows (inclusive, relative to query pos)."""
    if isinstance(mask_cfg, int):
        w = int(mask_cfg)
        return { 'upstream': None, 'local': (-w, w), 'downstream': None }
    if isinstance(mask_cfg, tuple) and len(mask_cfg) == 2:
        before, after = int(mask_cfg[0]), int(mask_cfg[1])
        up = (-before, 0) if before > 0 else None
        dn = (0, after) if after > 0 else None
        return { 'upstream': up, 'local': None, 'downstream': dn }
    if isinstance(mask_cfg, tuple) and len(mask_cfg) == 3:
        before, gap, after = int(mask_cfg[0]), int(mask_cfg[1]), int(mask_cfg[2])
        up = (-before, -(max(0, gap) + 1)) if before > 0 else None
        dn_start = max(1, gap)
        dn = (dn_start, after) if after > 0 and after >= dn_start else None
        return { 'upstream': up, 'local': None, 'downstream': dn }
    return { 'upstream': None, 'local': None, 'downstream': None }

def load_training_data(base_dir: Path):
    """Load training dynamics and final attention data."""
    
    dynamics_file = base_dir / "training_dynamics" / "training_dynamics.json"
    attention_file = base_dir / "layer_analysis" / "attention_weights.json"
    
    dynamics_data = []
    if dynamics_file.exists():
        with open(dynamics_file, 'r') as f:
            dynamics_data = json.load(f)
    
    attention_data = None
    if attention_file.exists():
        # For the large file, we'll extract summary statistics
        attention_data = extract_attention_summary(attention_file)
    
    return dynamics_data, attention_data

def extract_attention_summary(attention_file: Path, max_samples: int = 10):
    """Extract summary statistics from the large attention file."""
    
    import ijson
    
    summary = {
        'layer_head_patterns': {},
        'position_ranges': {},
        'global_ranges': {
            'local_boundary': 10,  # Will be dynamically determined
            'upstream_extent': (-50, -10),  # Will be dynamically determined  
            'downstream_extent': (10, 50)   # Will be dynamically determined
        }
    }
    
    # Collect all relative positions to determine dynamic boundaries
    all_relative_positions = []
    
    try:
        with open(attention_file, 'rb') as file:
            sample_count = 0
            
            for item in ijson.items(file, 'start_position_attention.item'):
                if sample_count >= max_samples:
                    break
                    
                if 'attention_patterns' not in item:
                    continue
                
                position = item['position']
                attn_patterns = item['attention_patterns']
                
                for layer_name, layer_data in attn_patterns.items():
                    if layer_name not in summary['layer_head_patterns']:
                        summary['layer_head_patterns'][layer_name] = {}
                        summary['position_ranges'][layer_name] = {}
                    
                    for head_name, head_data in layer_data.items():
                        if head_name not in summary['layer_head_patterns'][layer_name]:
                            summary['layer_head_patterns'][layer_name][head_name] = {
                                'upstream_scores': [],
                                'local_scores': [],
                                'downstream_scores': [],
                                'position_ranges': []
                            }
                        
                        # Store attention scores
                        head_stats = summary['layer_head_patterns'][layer_name][head_name]
                        head_stats['upstream_scores'].append(head_data['upstream_attention'])
                        head_stats['local_scores'].append(head_data['local_attention'])
                        head_stats['downstream_scores'].append(head_data['downstream_attention'])
                        
                        # Calculate position ranges
                        top_positions = head_data['top_attended_positions']
                        relative_positions = [pos - position for pos in top_positions]
                        all_relative_positions.extend(relative_positions)
                        
                        # Store all positions for dynamic boundary calculation
                        head_stats['position_ranges'].append({
                            'all_positions': relative_positions
                        })
                
                sample_count += 1
    
    except Exception as e:
        print(f"Error extracting attention summary: {e}")
        return {}
    
    # Calculate dynamic boundaries based on actual data
    if all_relative_positions:
        all_relative_positions = sorted(all_relative_positions)
        
        # Define local boundary as the range that captures the central attention cluster
        # Use percentiles to find natural breakpoints
        local_positions = [p for p in all_relative_positions if abs(p) <= 20]  # Initial broad local range
        if local_positions:
            # Local boundary: contains ~80% of positions closest to 0
            local_boundary = max(5, min(15, int(np.percentile([abs(p) for p in local_positions], 80))))
        else:
            local_boundary = 10  # fallback
        
        # Recategorize with dynamic boundary
        upstream_positions = [p for p in all_relative_positions if p < -local_boundary]
        downstream_positions = [p for p in all_relative_positions if p > local_boundary]
        
        # Calculate upstream extent (5th to 95th percentile to avoid outliers)
        if upstream_positions:
            upstream_min = int(np.percentile(upstream_positions, 5))
            upstream_max = -local_boundary
            upstream_extent = (upstream_min, upstream_max)
        else:
            upstream_extent = (-50, -local_boundary)
        
        # Calculate downstream extent (5th to 95th percentile to avoid outliers)  
        if downstream_positions:
            downstream_min = local_boundary
            downstream_max = int(np.percentile(downstream_positions, 95))
            downstream_extent = (downstream_min, downstream_max)
        else:
            downstream_extent = (local_boundary, 50)
        
        summary['global_ranges'] = {
            'local_boundary': local_boundary,
            'upstream_extent': upstream_extent,
            'downstream_extent': downstream_extent
        }
        
        print(f"Dynamic ranges determined:")
        print(f"  Local boundary: ±{local_boundary} bases")
        print(f"  Upstream extent: {upstream_extent[0]} to {upstream_extent[1]} bases")
        print(f"  Downstream extent: {downstream_extent[0]} to {downstream_extent[1]} bases")
        
        # Now recalculate position ranges with dynamic boundaries
        for layer_name in summary['layer_head_patterns']:
            for head_name in summary['layer_head_patterns'][layer_name]:
                head_stats = summary['layer_head_patterns'][layer_name][head_name]
                updated_ranges = []
                
                for range_data in head_stats['position_ranges']:
                    relative_positions = range_data['all_positions']
                    
                    # Categorize positions with dynamic boundaries
                    local_pos = [p for p in relative_positions if abs(p) <= local_boundary]
                    upstream_pos = [p for p in relative_positions if p < -local_boundary]
                    downstream_pos = [p for p in relative_positions if p > local_boundary]
                    
                    updated_ranges.append({
                        'local': local_pos,
                        'upstream': upstream_pos,
                        'downstream': downstream_pos,
                        'upstream_range': (min(upstream_pos), max(upstream_pos)) if upstream_pos else None,
                        'downstream_range': (min(downstream_pos), max(downstream_pos)) if downstream_pos else None
                    })
                
                head_stats['position_ranges'] = updated_ranges
    
    return summary

def determine_head_specialization(head_data):
    """Determine the primary specialization of an attention head."""
    
    upstream_avg = np.mean(head_data['upstream_scores'])
    local_avg = np.mean(head_data['local_scores'])
    downstream_avg = np.mean(head_data['downstream_scores'])
    
    # Determine primary focus
    if local_avg > upstream_avg and local_avg > downstream_avg:
        primary = "local"
    elif upstream_avg > downstream_avg:
        primary = "upstream"
    else:
        primary = "downstream"
    
    # Calculate position ranges
    all_ranges = head_data['position_ranges']
    upstream_ranges = [r['upstream_range'] for r in all_ranges if r['upstream_range']]
    downstream_ranges = [r['downstream_range'] for r in all_ranges if r['downstream_range']]
    
    upstream_extent = None
    downstream_extent = None
    
    if upstream_ranges:
        all_upstream_min = min(r[0] for r in upstream_ranges)
        all_upstream_max = max(r[1] for r in upstream_ranges)
        upstream_extent = (all_upstream_min, all_upstream_max)
    
    if downstream_ranges:
        all_downstream_min = min(r[0] for r in downstream_ranges)
        all_downstream_max = max(r[1] for r in downstream_ranges)
        downstream_extent = (all_downstream_min, all_downstream_max)
    
    return {
        'primary_focus': primary,
        'attention_scores': {
            'upstream': upstream_avg,
            'local': local_avg,
            'downstream': downstream_avg
        },
        'position_extents': {
            'upstream': upstream_extent,
            'downstream': downstream_extent
        }
    }

def normalize_epoch_data(epoch_data, num_layers: int, num_heads: int, is_final_model: bool = False, global_ranges: dict = None):
    """Convert epoch data to a normalized format for drawing."""
    
    # Default ranges if not provided
    if global_ranges is None:
        global_ranges = {
            'local_boundary': 10,
            'upstream_extent': (-50, -10),
            'downstream_extent': (10, 50)
        }
    
    normalized_data = {
        'epoch_label': 'Final Model' if is_final_model else f"Epoch {epoch_data.get('epoch', 'Unknown')}",
        'layers': {},
        'global_ranges': global_ranges
    }
    
    if is_final_model:
        # Handle final model attention data
        if 'layer_head_patterns' in epoch_data:
            layer_patterns = epoch_data['layer_head_patterns']
            
            for layer_idx in range(num_layers):
                layer_name = f'layer_{layer_idx}'
                normalized_data['layers'][layer_idx] = {'heads': {}}
                
                if layer_name in layer_patterns:
                    layer_data = layer_patterns[layer_name]
                    
                    for head_idx in range(num_heads):
                        head_name = f'head_{head_idx}'
                        
                        if head_name in layer_data:
                            head_spec = determine_head_specialization(layer_data[head_name])
                            scores = head_spec['attention_scores']
                            
                            normalized_data['layers'][layer_idx]['heads'][head_idx] = {
                                'upstream_focus': float(scores['upstream']),
                                'local_focus': float(scores['local']),
                                'downstream_focus': float(scores['downstream']),
                                'upstream_extent': head_spec['position_extents']['upstream'],
                                'downstream_extent': head_spec['position_extents']['downstream'],
                                'use_actual_ranges': True
                            }
    else:
        # Handle training epoch data
        if 'attention_focus_evolution' in epoch_data:
            focus_data = epoch_data['attention_focus_evolution']
            
            for layer_idx in range(num_layers):
                layer_name = f'layer_{layer_idx}'
                normalized_data['layers'][layer_idx] = {'heads': {}}
                
                if layer_name in focus_data:
                    layer_data = focus_data[layer_name]
                    
                    for head_idx in range(num_heads):
                        head_name = f'head_{head_idx}'
                        
                        if head_name in layer_data:
                            head_data = layer_data[head_name]
                            
                            normalized_data['layers'][layer_idx]['heads'][head_idx] = {
                                'upstream_focus': float(head_data['avg_upstream_focus']),
                                'local_focus': float(head_data['avg_local_focus']),
                                'downstream_focus': float(head_data['avg_downstream_focus']),
                                'upstream_extent': None,  # Will use estimated ranges
                                'downstream_extent': None,  # Will use estimated ranges
                                'use_actual_ranges': False
                            }
    
    return normalized_data





def detect_model_architecture(dynamics_data, attention_data):
    """Detect the number of layers and heads from the training data."""
    
    num_layers = 0
    num_heads = 0
    
    # Try to get architecture info from dynamics data first
    if dynamics_data:
        for epoch_data in dynamics_data:
            if 'attention_focus_evolution' in epoch_data:
                focus_data = epoch_data['attention_focus_evolution']
                
                # Count layers
                layer_names = [name for name in focus_data.keys() if name.startswith('layer_')]
                if layer_names:
                    layer_indices = [int(name.split('_')[1]) for name in layer_names]
                    num_layers = max(layer_indices) + 1
                
                # Count heads from first layer
                if layer_names:
                    first_layer_data = focus_data[layer_names[0]]
                    head_names = [name for name in first_layer_data.keys() if name.startswith('head_')]
                    if head_names:
                        head_indices = [int(name.split('_')[1]) for name in head_names]
                        num_heads = max(head_indices) + 1
                
                if num_layers > 0 and num_heads > 0:
                    break
    
    # If not found in dynamics data, try attention data
    if (num_layers == 0 or num_heads == 0) and attention_data and 'layer_head_patterns' in attention_data:
        layer_patterns = attention_data['layer_head_patterns']
        
        # Count layers
        if not num_layers:
            layer_names = [name for name in layer_patterns.keys() if name.startswith('layer_')]
            if layer_names:
                layer_indices = [int(name.split('_')[1]) for name in layer_names]
                num_layers = max(layer_indices) + 1
        
        # Count heads from first layer
        if not num_heads and layer_names:
            first_layer_data = layer_patterns[layer_names[0]]
            head_names = [name for name in first_layer_data.keys() if name.startswith('head_')]
            if head_names:
                head_indices = [int(name.split('_')[1]) for name in head_names]
                num_heads = max(head_indices) + 1
    
    # Fallback to defaults if detection failed
    if num_layers == 0:
        print("Warning: Could not detect number of layers, using default of 4")
        num_layers = 4
    if num_heads == 0:
        print("Warning: Could not detect number of heads, using default of 6")
        num_heads = 6
    
    print(f"Detected model architecture: {num_layers} layers, {num_heads} heads per layer")
    return num_layers, num_heads

def cluster_positions(positions, min_cluster_size=3, max_gap=20):
    """Cluster attention positions into ranges."""
    if not positions:
        return []
    
    positions = sorted(positions)
    clusters = []
    current_cluster = [positions[0]]
    
    for pos in positions[1:]:
        if pos - current_cluster[-1] <= max_gap:
            current_cluster.append(pos)
        else:
            if len(current_cluster) >= min_cluster_size:
                clusters.append((min(current_cluster), max(current_cluster)))
            current_cluster = [pos]
    
    # Add the last cluster
    if len(current_cluster) >= min_cluster_size:
        clusters.append((min(current_cluster), max(current_cluster)))
    
    return clusters

def extract_attention_ranges_to_tsv(dynamics_data, attention_data, output_path: Path, num_layers: int, num_heads: int):
    """Extract attention ranges to TSV format."""
    
    import csv
    
    # Get global ranges and per-head masks (if available)
    global_ranges = None
    if attention_data and 'global_ranges' in attention_data:
        global_ranges = attention_data['global_ranges']
        local_boundary = global_ranges['local_boundary']
    else:
        local_boundary = 10
    # Attempt to load mask config from the run dir inferred from output_path
    run_dir = output_path.parent
    attention_masks = _load_attention_masks_from_ckpt(run_dir)
    
    rows = []
    
    # Process training epochs (deduplicate by taking the last entry for each epoch)
    # Note: Some epochs (e.g., epoch 0) may have multiple entries in the training dynamics data.
    # This can occur when there are multiple validation runs or initialization states recorded.
    # We take the last entry for each epoch, which typically represents the final state
    # after any initialization or early training steps within that epoch.
    epoch_data_map = {}
    for data in dynamics_data:
        epoch = data['epoch']
        epoch_data_map[epoch] = data  # This will keep the last entry for each epoch
    
    for epoch, epoch_data in sorted(epoch_data_map.items()):
        
        if 'attention_focus_evolution' in epoch_data:
            focus_data = epoch_data['attention_focus_evolution']
            
            for layer_idx in range(num_layers):
                layer_name = f'layer_{layer_idx}'
                
                if layer_name in focus_data:
                    layer_data = focus_data[layer_name]
                    
                    for head_idx in range(num_heads):
                        head_name = f'head_{head_idx}'
                        
                        if head_name in layer_data:
                            head_data = layer_data[head_name]
                            
                            upstream_focus = float(head_data['avg_upstream_focus'])
                            local_focus = float(head_data['avg_local_focus'])
                            downstream_focus = float(head_data['avg_downstream_focus'])
                            
                            # Add ranges using per-head mask windows when available; fallback to dynamic boundaries
                            range_idx = 0  # Range index counter for this head
                            
                            mw = _mask_windows(attention_masks.get(head_idx)) if attention_masks else {'upstream': None, 'local': None, 'downstream': None}
                            if upstream_focus > 0.0001 and mw['upstream'] is not None:
                                if mw['upstream'] is not None:
                                    start_pos, end_pos = mw['upstream']
                                else:
                                    pass
                                
                                rows.append({
                                    'epoch': epoch,
                                    'layer': layer_idx,
                                    'head': head_idx,
                                    'range_index': range_idx,
                                    'range_start': start_pos,
                                    'range_end': end_pos,
                                    'average_attention_weight': upstream_focus
                                })
                                range_idx += 1
                            
                            if local_focus > 0.0001 and mw['local'] is not None:
                                if mw['local'] is not None:
                                    ls, le = mw['local']
                                else:
                                    ls, le = -local_boundary, local_boundary
                                rows.append({
                                    'epoch': epoch,
                                    'layer': layer_idx,
                                    'head': head_idx,
                                    'range_index': range_idx,
                                    'range_start': ls,
                                    'range_end': le,
                                    'average_attention_weight': local_focus
                                })
                                range_idx += 1
                            
                            if downstream_focus > 0.0001 and mw['downstream'] is not None:
                                if mw['downstream'] is not None:
                                    start_pos, end_pos = mw['downstream']
                                else:
                                    pass
                                
                                rows.append({
                                    'epoch': epoch,
                                    'layer': layer_idx,
                                    'head': head_idx,
                                    'range_index': range_idx,
                                    'range_start': start_pos,
                                    'range_end': end_pos,
                                    'average_attention_weight': downstream_focus
                                })
                                range_idx += 1
    
    # Process final model with actual position data
    if attention_data and 'layer_head_patterns' in attention_data:
        layer_patterns = attention_data['layer_head_patterns']
        
        for layer_idx in range(num_layers):
            layer_name = f'layer_{layer_idx}'
            
            if layer_name in layer_patterns:
                layer_data = layer_patterns[layer_name]
                
                for head_idx in range(num_heads):
                    head_name = f'head_{head_idx}'
                    
                    if head_name in layer_data:
                        head_spec = determine_head_specialization(layer_data[head_name])
                        scores = head_spec['attention_scores']
                        
                        upstream_score = float(scores['upstream'])
                        local_score = float(scores['local'])
                        downstream_score = float(scores['downstream'])
                        
                        # Get all positions from position ranges
                        all_positions = []
                        for range_data in layer_data[head_name]['position_ranges']:
                            # Collect positions from all categories
                            all_positions.extend(range_data.get('upstream', []))
                            all_positions.extend(range_data.get('local', []))
                            all_positions.extend(range_data.get('downstream', []))
                        
                        if all_positions:
                            # Categorize positions
                            upstream_positions = [p for p in all_positions if p < -local_boundary]
                            local_positions = [p for p in all_positions if abs(p) <= local_boundary]
                            downstream_positions = [p for p in all_positions if p > local_boundary]
                            
                            # Range index counter for this head
                            range_idx = 0
                            
                            # Cluster upstream positions (only if mask allows upstream)
                            mw = _mask_windows(attention_masks.get(head_idx)) if attention_masks else {'upstream': None, 'local': None, 'downstream': None}
                            if mw.get('upstream') is not None and upstream_positions and upstream_score > 0.0001:
                                upstream_clusters = cluster_positions(upstream_positions)
                                if not upstream_clusters:  # If no clusters, use full range
                                    upstream_clusters = [(min(upstream_positions), max(upstream_positions))]
                                
                                for start_pos, end_pos in upstream_clusters:
                                    rows.append({
                                        'epoch': 'final',
                                        'layer': layer_idx,
                                        'head': head_idx,
                                        'range_index': range_idx,
                                        'range_start': start_pos,
                                        'range_end': end_pos,
                                        'average_attention_weight': upstream_score / len(upstream_clusters)
                                    })
                                    range_idx += 1
                            
                            # Local range (use mask if present)
                            if mw.get('local') is not None and local_score > 0.0001:
                                ls, le = (mw['local'] if mw.get('local') is not None else (-local_boundary, local_boundary))
                                rows.append({
                                    'epoch': 'final',
                                    'layer': layer_idx,
                                    'head': head_idx,
                                    'range_index': range_idx,
                                    'range_start': ls,
                                    'range_end': le,
                                    'average_attention_weight': local_score
                                })
                                range_idx += 1
                            
                            # Cluster downstream positions (only if mask allows downstream)
                            if mw.get('downstream') is not None and downstream_positions and downstream_score > 0.0001:
                                downstream_clusters = cluster_positions(downstream_positions)
                                if not downstream_clusters:  # If no clusters, use full range
                                    downstream_clusters = [(min(downstream_positions), max(downstream_positions))]
                                
                                for start_pos, end_pos in downstream_clusters:
                                    rows.append({
                                        'epoch': 'final',
                                        'layer': layer_idx,
                                        'head': head_idx,
                                        'range_index': range_idx,
                                        'range_start': start_pos,
                                        'range_end': end_pos,
                                        'average_attention_weight': downstream_score / len(downstream_clusters)
                                    })
                                    range_idx += 1
    
    # Write to TSV
    with open(output_path, 'w', newline='') as tsvfile:
        fieldnames = ['epoch', 'layer', 'head', 'range_index', 'range_start', 'range_end', 'average_attention_weight']
        writer = csv.DictWriter(tsvfile, fieldnames=fieldnames, delimiter='\t')
        writer.writeheader()
        
        # Sort rows by epoch, layer, head for better organization
        rows.sort(key=lambda x: (x['epoch'] if x['epoch'] != 'final' else 999, x['layer'], x['head'], x['range_start']))
        writer.writerows(rows)
    
    print(f"Attention ranges extracted to TSV: {output_path}")
    print(f"Total rows: {len(rows)}")

def main():
    parser = argparse.ArgumentParser(description='Extract attention range data to TSV format')
    parser.add_argument('--run-dir', type=str, required=True,
                       help='Path to training run directory')
    parser.add_argument('--output-dir', type=str, 
                       help='Output directory (defaults to run directory)')
    
    args = parser.parse_args()
    
    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir) if args.output_dir else run_dir
    
    if not run_dir.exists():
        print(f"Error: Run directory not found: {run_dir}")
        return
    
    print(f"Loading training data from: {run_dir}")
    dynamics_data, attention_data = load_training_data(run_dir)
    
    if not dynamics_data:
        print("Error: No training dynamics data found")
        return
    
    print(f"Found {len(dynamics_data)} epochs of training data")
    
    # Detect model architecture from data
    num_layers, num_heads = detect_model_architecture(dynamics_data, attention_data)
    
    # Generate TSV data
    tsv_output = output_dir / "attention_ranges.tsv"
    extract_attention_ranges_to_tsv(dynamics_data, attention_data, tsv_output, num_layers, num_heads)
    
    print("Attention range extraction complete!")

if __name__ == "__main__":
    main()

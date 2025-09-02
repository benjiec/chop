#!/usr/bin/env python3
"""
Training Dynamics Analysis Callback.

This callback tracks how attention patterns and layer specialization
evolve during training, saving summary statistics at each epoch.

Captures:
- Attention focus patterns (upstream/local/downstream) per head
- Layer specialization metrics over time  
- START prediction emergence
- Learning phase transitions

Storage: ~100MB total (summary stats only, no full attention matrices)
"""

import torch
import pytorch_lightning as pl
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Any

class TrainingDynamicsCallback(pl.Callback):
    """Callback to track training dynamics and attention evolution."""
    
    def __init__(self, val_loader, output_dir: Path, analysis_frequency: int = 5):
        """
        Args:
            val_loader: Validation data loader for analysis
            output_dir: Directory to save training dynamics data
            analysis_frequency: Analyze every N epochs (default: 5)
        """
        self.val_loader = val_loader
        self.output_dir = Path(output_dir)
        self.analysis_frequency = analysis_frequency
        
        # Storage for training dynamics
        self.epoch_data = []
        
        # Convert indices back to nucleotides
        self.idx_to_nucleotide = {0: 'A', 1: 'T', 2: 'G', 3: 'C', 4: 'N'}
        
        # Dynamic layer count (will be set when first model is analyzed)
        self.num_layers = None
        
    def on_validation_epoch_end(self, trainer, pl_module):
        """Analyze model state at the end of each validation epoch."""
        
        current_epoch = trainer.current_epoch
        
        # Only analyze every N epochs to save time
        if current_epoch % self.analysis_frequency != 0 and current_epoch != trainer.max_epochs - 1:
            return
        
        print(f"\n  📊 Analyzing training dynamics at epoch {current_epoch}...")
        
        # Analyze current model state
        epoch_analysis = self._analyze_epoch_state(pl_module, current_epoch)
        self.epoch_data.append(epoch_analysis)
        
        # Save incremental data
        self._save_training_dynamics()
    
    def _analyze_epoch_state(self, model, epoch: int) -> Dict[str, Any]:
        """Analyze model state at current epoch."""
        
        model.eval()
        device = next(model.parameters()).device
        
        # Set layer count dynamically on first analysis
        if self.num_layers is None:
            self.num_layers = len(model.model.transformer_layers)
            print(f"    Detected {self.num_layers} layers in model")
        
        # Collect data from a few validation samples
        attention_summaries = []
        start_predictions = []
        layer_activations = []
        
        sample_count = 0
        max_samples = 5  # Analyze 5 samples per epoch
        
        with torch.no_grad():
            for batch in self.val_loader:
                if sample_count >= max_samples:
                    break
                    
                sequences, targets = batch
                sequences = sequences.to(device)
                targets = targets.to(device)
                
                # Get predictions and attention weights
                try:
                    logits, attention_weights = model.model(sequences, return_attention=True)
                    predictions = torch.argmax(logits, dim=-1)
                    probabilities = torch.softmax(logits, dim=-1)
                    
                    # Process each sequence in batch
                    for i in range(min(sequences.size(0), max_samples - sample_count)):
                        seq_tensor = sequences[i]
                        target_tensor = targets[i]
                        pred_tensor = predictions[i]
                        prob_tensor = probabilities[i]
                        
                        # Convert sequence for analysis
                        sequence_str = ''.join([self.idx_to_nucleotide[idx.item()] for idx in seq_tensor])
                        
                        # Find START positions and analyze attention
                        for pos in range(len(sequence_str) - 2):
                            if sequence_str[pos:pos+3] == 'ATG' and target_tensor[pos] == 2:  # Real START
                                
                                # Analyze attention patterns at this START position
                                start_attention_summary = {}
                                for layer_name, layer_attention in attention_weights.items():
                                    if layer_attention is not None:
                                        layer_head_summary = {}
                                        
                                        # Handle different attention tensor formats
                                        if layer_attention.dim() == 4:
                                            # Format: (batch, heads, seq_len, seq_len)
                                            num_heads = layer_attention.size(1)
                                            
                                            # Analyze all heads in this layer
                                            for head_idx in range(num_heads):
                                                head_attention = layer_attention[i, head_idx, pos, :].cpu().numpy()
                                                
                                                # Calculate attention focus regions for this head
                                                upstream_attn = head_attention[max(0, pos-100):pos].mean() if pos >= 100 else head_attention[:pos].mean()
                                                downstream_attn = head_attention[pos+3:pos+53].mean() if pos+53 < len(head_attention) else head_attention[pos+3:].mean()
                                                local_attn = head_attention[max(0, pos-5):pos+8].mean()
                                                
                                                layer_head_summary[f'head_{head_idx}'] = {
                                                    'upstream_focus': float(upstream_attn),
                                                    'local_focus': float(local_attn),
                                                    'downstream_focus': float(downstream_attn)
                                                }
                                        
                                        elif layer_attention.dim() == 3:
                                            # Format: (batch, seq_len, seq_len) - averaged across heads
                                            pos_attention = layer_attention[i, pos, :].cpu().numpy()
                                            
                                            # Calculate attention focus regions
                                            upstream_attn = pos_attention[max(0, pos-100):pos].mean() if pos >= 100 else pos_attention[:pos].mean()
                                            downstream_attn = pos_attention[pos+3:pos+53].mean() if pos+53 < len(pos_attention) else pos_attention[pos+3:].mean()
                                            local_attn = pos_attention[max(0, pos-5):pos+8].mean()
                                            
                                            layer_head_summary['averaged_heads'] = {
                                                'upstream_focus': float(upstream_attn),
                                                'local_focus': float(local_attn),
                                                'downstream_focus': float(downstream_attn)
                                            }
                                        
                                        start_attention_summary[layer_name] = layer_head_summary
                                
                                start_predictions.append({
                                    'epoch': epoch,
                                    'sample_index': sample_count,
                                    'position': pos,
                                    'start_probability': prob_tensor[pos, 2].item(),
                                    'predicted_correctly': pred_tensor[pos] == 2,
                                    'attention_summary': start_attention_summary
                                })
                        
                        sample_count += 1
                        if sample_count >= max_samples:
                            break
                
                except Exception as e:
                    print(f"    Warning: Could not extract attention weights at epoch {epoch}: {e}")
                    break
        
        # Calculate epoch-level summaries
        epoch_summary = {
            'epoch': epoch,
            'total_start_positions_analyzed': len(start_predictions),
            'start_prediction_accuracy': sum(1 for sp in start_predictions if sp['predicted_correctly']) / len(start_predictions) if start_predictions else 0,
            'average_start_probability': sum(sp['start_probability'] for sp in start_predictions) / len(start_predictions) if start_predictions else 0,
            'attention_focus_evolution': self._summarize_attention_evolution(start_predictions),
            'layer_specialization_metrics': self._calculate_layer_specialization(start_predictions)
        }
        
        return epoch_summary
    
    def _summarize_attention_evolution(self, start_predictions: List[Dict]) -> Dict:
        """Summarize how attention focus evolves, preserving per-head data."""
        
        if not start_predictions:
            return {}
        
        # Calculate attention focus by layer, preserving per-head information
        layer_focus = {}
        for layer_idx in range(self.num_layers if self.num_layers else 4):  # Dynamic layer count
            layer_name = f'layer_{layer_idx}'
            
            # Collect per-head data across all samples
            head_data_collection = {}
            
            for sp in start_predictions:
                if layer_name in sp['attention_summary']:
                    layer_data = sp['attention_summary'][layer_name]
                    
                    # Handle new multi-head format
                    if isinstance(layer_data, dict):
                        # Check if this is the old single-head format or new multi-head format
                        if 'upstream_focus' in layer_data:
                            # Old format - single values (treat as head_0)
                            if 'head_0' not in head_data_collection:
                                head_data_collection['head_0'] = {'upstream': [], 'local': [], 'downstream': []}
                            head_data_collection['head_0']['upstream'].append(layer_data['upstream_focus'])
                            head_data_collection['head_0']['local'].append(layer_data['local_focus'])
                            head_data_collection['head_0']['downstream'].append(layer_data['downstream_focus'])
                        else:
                            # New format - multiple heads, preserve per-head data
                            for head_name, head_data in layer_data.items():
                                if isinstance(head_data, dict) and 'upstream_focus' in head_data:
                                    if head_name not in head_data_collection:
                                        head_data_collection[head_name] = {'upstream': [], 'local': [], 'downstream': []}
                                    head_data_collection[head_name]['upstream'].append(head_data['upstream_focus'])
                                    head_data_collection[head_name]['local'].append(head_data['local_focus'])
                                    head_data_collection[head_name]['downstream'].append(head_data['downstream_focus'])
            
            # Calculate per-head statistics
            if head_data_collection:
                layer_focus[layer_name] = {}
                
                # Store per-head data
                for head_name, head_scores in head_data_collection.items():
                    if head_scores['upstream']:  # Only if we have data
                        layer_focus[layer_name][head_name] = {
                            'avg_upstream_focus': float(np.mean(head_scores['upstream'])),
                            'avg_local_focus': float(np.mean(head_scores['local'])),
                            'avg_downstream_focus': float(np.mean(head_scores['downstream'])),
                            'upstream_std': float(np.std(head_scores['upstream'])),
                            'local_std': float(np.std(head_scores['local'])),
                            'downstream_std': float(np.std(head_scores['downstream']))
                        }
                
                # Also calculate layer averages for backward compatibility
                all_upstream = []
                all_local = []
                all_downstream = []
                for head_scores in head_data_collection.values():
                    all_upstream.extend(head_scores['upstream'])
                    all_local.extend(head_scores['local'])
                    all_downstream.extend(head_scores['downstream'])
                
                if all_upstream:
                    layer_focus[layer_name]['layer_average'] = {
                        'avg_upstream_focus': float(np.mean(all_upstream)),
                        'avg_local_focus': float(np.mean(all_local)),
                        'avg_downstream_focus': float(np.mean(all_downstream)),
                        'upstream_std': float(np.std(all_upstream)),
                        'local_std': float(np.std(all_local)),
                        'downstream_std': float(np.std(all_downstream))
                    }
        
        return layer_focus
    
    def _calculate_layer_specialization(self, start_predictions: List[Dict]) -> Dict:
        """Calculate metrics showing how layers specialize."""
        
        if not start_predictions:
            return {}
        
        # Calculate which layers focus where
        specialization = {}
        
        for layer_idx in range(self.num_layers if self.num_layers else 4):  # Dynamic layer count
            layer_name = f'layer_{layer_idx}'
            
            # Collect focus scores for this layer
            focus_scores = []
            for sp in start_predictions:
                if layer_name in sp['attention_summary']:
                    layer_data = sp['attention_summary'][layer_name]
                    
                    # Handle both old and new attention summary formats
                    if isinstance(layer_data, dict):
                        if 'upstream_focus' in layer_data:
                            # Old format - single values
                            focus_scores.append({
                                'upstream': layer_data['upstream_focus'],
                                'local': layer_data['local_focus'], 
                                'downstream': layer_data['downstream_focus']
                            })
                        else:
                            # New format - multiple heads, average across heads
                            head_upstream = []
                            head_local = []
                            head_downstream = []
                            
                            for head_name, head_data in layer_data.items():
                                if isinstance(head_data, dict) and 'upstream_focus' in head_data:
                                    head_upstream.append(head_data['upstream_focus'])
                                    head_local.append(head_data['local_focus'])
                                    head_downstream.append(head_data['downstream_focus'])
                            
                            if head_upstream:  # Only if we found head data
                                focus_scores.append({
                                    'upstream': sum(head_upstream) / len(head_upstream),
                                    'local': sum(head_local) / len(head_local),
                                    'downstream': sum(head_downstream) / len(head_downstream)
                                })
            
            if focus_scores:
                # Calculate which region this layer focuses on most
                avg_upstream = np.mean([fs['upstream'] for fs in focus_scores])
                avg_local = np.mean([fs['local'] for fs in focus_scores])
                avg_downstream = np.mean([fs['downstream'] for fs in focus_scores])
                
                # Determine primary focus
                max_focus = max(avg_upstream, avg_local, avg_downstream)
                if max_focus == avg_upstream:
                    primary_focus = 'upstream'
                elif max_focus == avg_local:
                    primary_focus = 'local'
                else:
                    primary_focus = 'downstream'
                
                specialization[layer_name] = {
                    'primary_focus': primary_focus,
                    'focus_strength': float(max_focus),
                    'focus_distribution': {
                        'upstream': float(avg_upstream),
                        'local': float(avg_local),
                        'downstream': float(avg_downstream)
                    }
                }
        
        return specialization
    
    def _save_training_dynamics(self):
        """Save training dynamics data incrementally."""
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save epoch data
        dynamics_file = self.output_dir / "training_dynamics.json"
        with open(dynamics_file, 'w') as f:
            json.dump(self.epoch_data, f, indent=2)
        
        # Create summary visualization data
        if len(self.epoch_data) >= 2:  # Need at least 2 epochs for trends
            self._create_dynamics_summary()
    
    def _create_dynamics_summary(self):
        """Create summary of training dynamics trends."""
        
        epochs = [data['epoch'] for data in self.epoch_data]
        start_accuracies = [data['start_prediction_accuracy'] for data in self.epoch_data]
        start_probabilities = [data['average_start_probability'] for data in self.epoch_data]
        
        # Track layer specialization over time
        layer_trends = {}
        # Generate layer names dynamically
        layer_names = [f'layer_{i}' for i in range(self.num_layers if self.num_layers else 4)]
        for layer_name in layer_names:
            upstream_trends = []
            local_trends = []
            downstream_trends = []
            
            for data in self.epoch_data:
                if layer_name in data.get('layer_specialization_metrics', {}):
                    focus = data['layer_specialization_metrics'][layer_name]['focus_distribution']
                    upstream_trends.append(focus['upstream'])
                    local_trends.append(focus['local'])
                    downstream_trends.append(focus['downstream'])
            
            if upstream_trends:
                layer_trends[layer_name] = {
                    'epochs': epochs[:len(upstream_trends)],
                    'upstream_focus_trend': upstream_trends,
                    'local_focus_trend': local_trends,
                    'downstream_focus_trend': downstream_trends
                }
        
        # Save summary
        summary = {
            'training_epochs_analyzed': epochs,
            'start_prediction_accuracy_trend': start_accuracies,
            'start_probability_trend': start_probabilities,
            'layer_specialization_trends': layer_trends,
            'learning_insights': self._generate_learning_insights()
        }
        
        summary_file = self.output_dir / "training_dynamics_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
    
    def _generate_learning_insights(self) -> Dict[str, str]:
        """Generate insights about learning dynamics."""
        
        insights = {}
        
        if len(self.epoch_data) >= 3:
            # Check if START accuracy is improving
            start_accs = [data['start_prediction_accuracy'] for data in self.epoch_data]
            if start_accs[-1] > start_accs[0]:
                insights['start_learning'] = 'START prediction accuracy improved during training'
            else:
                insights['start_learning'] = 'START prediction accuracy did not improve significantly'
            
            # Check layer specialization
            first_epoch = self.epoch_data[0]
            last_epoch = self.epoch_data[-1]
            
            for layer_name in ['layer_0', 'layer_1', 'layer_2']:
                if (layer_name in first_epoch.get('layer_specialization_metrics', {}) and 
                    layer_name in last_epoch.get('layer_specialization_metrics', {})):
                    
                    first_focus = first_epoch['layer_specialization_metrics'][layer_name]['primary_focus']
                    last_focus = last_epoch['layer_specialization_metrics'][layer_name]['primary_focus']
                    
                    if first_focus != last_focus:
                        insights[f'{layer_name}_specialization'] = f'Changed from {first_focus} to {last_focus} focus'
                    else:
                        insights[f'{layer_name}_specialization'] = f'Consistently focused on {last_focus}'
        
        return insights

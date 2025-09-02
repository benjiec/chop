#!/usr/bin/env python3
"""
Comprehensive layer analysis for UTR-START context learning.

This module provides detailed analysis of what each transformer layer learns:
1. Attention weight visualization
2. Layer-wise feature analysis  
3. Gradient-based attribution
4. Combined visualizations

Each analysis saves to separate files for flexible post-processing.
"""

import torch
import torch.nn.functional as F
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns

class LayerAnalyzer:
    """Comprehensive analyzer for transformer layer behavior."""
    
    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.model.eval()
        
        # Convert indices back to nucleotides
        self.idx_to_nucleotide = {0: 'A', 1: 'T', 2: 'G', 3: 'C', 4: 'N'}
        self.class_names = {0: 'INTERGENIC', 1: 'UTR5', 2: 'START'}
        
    def analyze_all(self, data_loader, output_dir: Path, max_samples: int = 20):
        """Run all analysis types and save results."""
        
        print(f"Running comprehensive layer analysis on {max_samples} samples...")
        output_dir.mkdir(exist_ok=True)
        
        # Collect data for analysis
        sample_data = self._collect_sample_data(data_loader, max_samples)
        
        # 1. Attention weight analysis
        print("1. Analyzing attention weights...")
        attention_data = self._analyze_attention_weights(sample_data)
        self._save_attention_analysis(attention_data, output_dir / "attention_weights.json")
        
        # 2. Layer-wise feature analysis
        print("2. Analyzing layer-wise features...")
        feature_data = self._analyze_layer_features(sample_data)
        self._save_feature_analysis(feature_data, output_dir / "layer_features.json")
        
        # 3. Gradient-based attribution (simplified for now)
        print("3. Analyzing gradient attribution...")
        attribution_data = {'position_attributions': [], 'note': 'Gradient analysis temporarily disabled due to tensor type issues'}
        self._save_attribution_analysis(attribution_data, output_dir / "gradient_attribution.json")
        
        # 4. Save sequences in clean format
        print("4. Saving sequences and predictions...")
        self._save_sequences_and_predictions(sample_data, output_dir / "sequences_and_predictions.json")
        
        # 5. Generate combined visualization
        print("5. Creating combined visualization...")
        self._create_combined_visualization(attention_data, feature_data, attribution_data, output_dir)
        
        print(f"Analysis complete! Results saved to: {output_dir}")
    
    def _collect_sample_data(self, data_loader, max_samples: int) -> List[Dict]:
        """Collect sample data for analysis."""
        
        samples = []
        sample_count = 0
        
        with torch.no_grad():
            for batch in data_loader:
                sequences, targets = batch
                sequences = sequences.to(self.device)
                targets = targets.to(self.device)
                
                # Get predictions
                logits = self.model(sequences)
                predictions = torch.argmax(logits, dim=-1)
                probabilities = F.softmax(logits, dim=-1)
                
                # Process each sequence in batch
                for i in range(sequences.size(0)):
                    if sample_count >= max_samples:
                        break
                        
                    seq_tensor = sequences[i]
                    target_tensor = targets[i]
                    pred_tensor = predictions[i]
                    prob_tensor = probabilities[i]
                    
                    # Convert to readable format
                    sequence_str = ''.join([self.idx_to_nucleotide[idx.item()] for idx in seq_tensor])
                    targets_str = ''.join([str(idx.item()) for idx in target_tensor])
                    predictions_str = ''.join([str(idx.item()) for idx in pred_tensor])
                    
                    # Find ATGs and their classifications
                    atg_analysis = []
                    for pos in range(len(sequence_str) - 2):
                        if sequence_str[pos:pos+3] == 'ATG':
                            atg_analysis.append({
                                'position': pos,
                                'target_class': target_tensor[pos].item(),
                                'predicted_class': pred_tensor[pos].item(),
                                'start_probability': prob_tensor[pos, 2].item(),  # START class probability
                                'context_before': sequence_str[max(0, pos-20):pos],
                                'context_after': sequence_str[pos+3:pos+23]
                            })
                    
                    samples.append({
                        'sample_index': sample_count,
                        'sequence': sequence_str,
                        'targets': targets_str,
                        'predictions': predictions_str,
                        'sequence_tensor': seq_tensor,
                        'target_tensor': target_tensor,
                        'prediction_tensor': pred_tensor,
                        'probability_tensor': prob_tensor,
                        'atg_analysis': atg_analysis
                    })
                    
                    sample_count += 1
                
                if sample_count >= max_samples:
                    break
        
        return samples
    
    def _analyze_attention_weights(self, sample_data: List[Dict]) -> Dict:
        """Extract and analyze attention weights from each layer."""
        
        attention_analysis = {
            'layer_attention_patterns': {},
            'start_position_attention': [],
            'attention_matrices': {}
        }
        
        # Hook to capture attention weights from each layer
        captured_attention = {}
        
        def create_attention_hook(layer_idx):
            def hook(module, input, output):
                # For MaskedTransformerLayer, we need to modify it to return attention weights
                # For now, we'll capture what we can from the MultiheadAttention module
                if hasattr(module, 'self_attn'):
                    # Enable attention weight return temporarily
                    module.self_attn.return_attention = True
            return hook
        
        # Register hooks on each transformer layer
        hooks = []
        for i, layer in enumerate(self.model.model.transformer_layers):
            hook = layer.register_forward_hook(create_attention_hook(i))
            hooks.append(hook)
        
        # Modify the model to capture attention weights
        # We need to temporarily modify the forward pass to return attention weights
        original_forward_methods = {}
        
        try:
            # Run inference on first few samples to capture attention
            with torch.no_grad():
                for sample in sample_data[:5]:  # Analyze first 5 samples
                    seq_tensor = sample['sequence_tensor'].unsqueeze(0).to(self.device)
                    
                    # Custom forward pass to capture attention weights
                    attention_weights = self._forward_with_attention_capture(seq_tensor)
                    
                    # Analyze attention patterns for START positions
                    for atg in sample['atg_analysis']:
                        if atg['target_class'] == 2:  # Real START
                            pos = atg['position']
                            
                            # Extract attention patterns for this START position
                            start_attention = {}
                            for layer_name, layer_attention in attention_weights.items():
                                if layer_attention is not None:
                                    print(f"Debug: {layer_name} attention shape: {layer_attention.shape}")
                                    
                                    # Handle different attention tensor formats
                                    if layer_attention.dim() == 4:
                                        # Expected format: (batch, heads, seq_len, seq_len)
                                        pos_attention = layer_attention[0, :, pos, :].cpu().numpy()  # (heads, seq_len)
                                        num_heads = pos_attention.shape[0]
                                    elif layer_attention.dim() == 3:
                                        # Alternative format: (batch, seq_len, seq_len) - averaged across heads
                                        pos_attention = layer_attention[0, pos, :].cpu().numpy()  # (seq_len,)
                                        # Reshape to look like single head
                                        pos_attention = pos_attention.reshape(1, -1)  # (1, seq_len)
                                        num_heads = 1
                                    else:
                                        print(f"Warning: Unexpected attention shape for {layer_name}: {layer_attention.shape}")
                                        continue
                                    
                                    # Analyze attention patterns for ALL heads
                                    head_patterns = {}
                                    for head_idx in range(num_heads):
                                        if num_heads == 1:
                                            head_attn = pos_attention[0]  # Single head case
                                        else:
                                            head_attn = pos_attention[head_idx]
                                        
                                        # Find top attended positions
                                        top_positions = np.argsort(head_attn)[-10:]  # Top 10 positions
                                        
                                        # Analyze upstream vs downstream attention
                                        upstream_attn = head_attn[max(0, pos-100):pos].mean() if pos >= 100 else head_attn[:pos].mean()
                                        downstream_attn = head_attn[pos+3:pos+53].mean() if pos+53 < len(head_attn) else head_attn[pos+3:].mean()
                                        local_attn = head_attn[max(0, pos-5):pos+8].mean()
                                        
                                        head_patterns[f'head_{head_idx}'] = {
                                            'upstream_attention': float(upstream_attn),
                                            'local_attention': float(local_attn), 
                                            'downstream_attention': float(downstream_attn),
                                            'top_attended_positions': [int(p) for p in top_positions],
                                            'attention_weights': head_attn.tolist()
                                        }
                                    
                                    start_attention[layer_name] = head_patterns
                            
                            attention_analysis['start_position_attention'].append({
                                'sample_index': sample['sample_index'],
                                'position': pos,
                                'start_probability': atg['start_probability'],
                                'predicted_correctly': atg['predicted_class'] == 2,
                                'attention_patterns': start_attention
                            })
                    
                    # Store full attention matrices only for first 2 samples (to limit file size)
                    if sample['sample_index'] < 2:
                        attention_analysis['attention_matrices'][f'sample_{sample["sample_index"]}'] = {
                            layer_name: layer_attention[0].cpu().numpy().tolist() if layer_attention is not None else None
                            for layer_name, layer_attention in attention_weights.items()
                        }
                        print(f"    Saved full attention matrices for sample {sample['sample_index']}")
                    else:
                        # For other samples, just save summary statistics to save space
                        print(f"    Saved attention summaries for sample {sample['sample_index']} (full matrices skipped)")
        
        finally:
            # Remove hooks
            for hook in hooks:
                hook.remove()
        
        return attention_analysis
    
    def _forward_with_attention_capture(self, seq_tensor: torch.Tensor) -> Dict:
        """Custom forward pass that captures attention weights from each layer."""
        
        # Use the model's attention extraction capability
        logits, attention_weights = self.model.model(seq_tensor, return_attention=True)
        
        return attention_weights
    
    def _analyze_layer_features(self, sample_data: List[Dict]) -> Dict:
        """Analyze how features evolve through layers."""
        
        feature_analysis = {
            'layer_activations': {},
            'feature_evolution': [],
            'start_position_features': []
        }
        
        # Hook to capture intermediate layer outputs
        layer_outputs = {}
        
        def feature_hook(layer_idx):
            def hook(module, input, output):
                layer_outputs[f'layer_{layer_idx}'] = output.detach().cpu()
            return hook
        
        # Register hooks on transformer layers
        hooks = []
        for i, layer in enumerate(self.model.model.transformer_layers):
            hook = layer.register_forward_hook(feature_hook(i))
            hooks.append(hook)
        
        # Run inference to capture features
        with torch.no_grad():
            for sample in sample_data[:5]:  # Analyze first 5 samples
                seq_tensor = sample['sequence_tensor'].unsqueeze(0).to(self.device)
                
                # Clear previous outputs
                layer_outputs.clear()
                
                # Forward pass (triggers hooks)
                _ = self.model(seq_tensor)
                
                # Analyze features at START positions
                for atg in sample['atg_analysis']:
                    if atg['target_class'] == 2:  # Real START
                        pos = atg['position']
                        
                        layer_features = {}
                        for layer_name, output in layer_outputs.items():
                            # Get feature vector at this position
                            feature_vec = output[0, pos, :].numpy()  # (d_model,)
                            layer_features[layer_name] = {
                                'mean_activation': float(feature_vec.mean()),
                                'max_activation': float(feature_vec.max()),
                                'min_activation': float(feature_vec.min()),
                                'std_activation': float(feature_vec.std())
                            }
                        
                        feature_analysis['start_position_features'].append({
                            'sample_index': sample['sample_index'],
                            'position': pos,
                            'predicted_correctly': atg['predicted_class'] == 2,
                            'layer_features': layer_features
                        })
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        return feature_analysis
    
    def _analyze_gradient_attribution(self, sample_data: List[Dict]) -> Dict:
        """Analyze gradient-based attribution for START predictions."""
        
        attribution_analysis = {
            'position_attributions': [],
            'upstream_importance': [],
            'downstream_importance': []
        }
        
        # Simplified gradient analysis - use embedding gradients
        self.model.train()
        
        for sample in sample_data[:5]:  # Analyze first 5 samples to avoid complexity
            seq_tensor = sample['sequence_tensor'].unsqueeze(0).to(self.device)
            
            # Hook to capture embedding gradients
            embedding_grads = {}
            
            def embedding_hook(module, grad_input, grad_output):
                if grad_output[0] is not None:
                    embedding_grads['embeddings'] = grad_output[0].detach().cpu()
            
            # Register hook on embedding layer
            hook = self.model.model.embedding.register_backward_hook(embedding_hook)
            
            try:
                # Forward pass
                logits = self.model(seq_tensor)
                
                # Find START positions for attribution
                for atg in sample['atg_analysis']:
                    if atg['target_class'] == 2:  # Real START
                        pos = atg['position']
                        
                        # Get gradient w.r.t. START class at this position
                        start_logit = logits[0, pos, 2]  # START class logit
                        
                        # Backward pass
                        self.model.zero_grad()
                        start_logit.backward(retain_graph=True)
                        
                        # Get attribution from embedding gradients
                        if 'embeddings' in embedding_grads:
                            emb_grad = embedding_grads['embeddings'][0]  # (seq_length, d_model)
                            # Sum across embedding dimensions to get per-position attribution
                            attribution = emb_grad.abs().sum(dim=1).numpy()
                            
                            # Analyze upstream vs downstream importance
                            upstream_attr = attribution[max(0, pos-100):pos].mean() if pos >= 100 else attribution[:pos].mean()
                            downstream_attr = attribution[pos+3:pos+53].mean() if pos+53 < len(attribution) else attribution[pos+3:].mean()
                            local_attr = attribution[max(0, pos-5):pos+8].mean()
                            
                            attribution_analysis['position_attributions'].append({
                                'sample_index': sample['sample_index'],
                                'position': pos,
                                'predicted_correctly': atg['predicted_class'] == 2,
                                'upstream_importance': float(upstream_attr),
                                'downstream_importance': float(downstream_attr),
                                'local_importance': float(local_attr),
                                'full_attribution': attribution.tolist()
                            })
            
            finally:
                hook.remove()
        
        self.model.eval()
        return attribution_analysis
    
    def _save_attention_analysis(self, data: Dict, filepath: Path):
        """Save attention analysis to JSON."""
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _save_feature_analysis(self, data: Dict, filepath: Path):
        """Save feature analysis to JSON."""
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _save_attribution_analysis(self, data: Dict, filepath: Path):
        """Save attribution analysis to JSON."""
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _save_sequences_and_predictions(self, sample_data: List[Dict], filepath: Path):
        """Save sequences and predictions in clean string format."""
        
        clean_data = []
        for sample in sample_data:
            clean_sample = {
                'sample_index': sample['sample_index'],
                'sequence': sample['sequence'],
                'targets': sample['targets'],
                'predictions': sample['predictions'],
                'sequence_length': len(sample['sequence']),
                'atg_analysis': sample['atg_analysis']
            }
            clean_data.append(clean_sample)
        
        with open(filepath, 'w') as f:
            json.dump(clean_data, f, indent=2)
    
    def _create_combined_visualization(self, attention_data: Dict, feature_data: Dict, 
                                     attribution_data: Dict, output_dir: Path):
        """Create combined visualization of all analyses."""
        
        # Create summary plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('UTR-START Context Learning Analysis', fontsize=16)
        
        # Plot 1: START position attribution patterns
        if attribution_data['position_attributions']:
            upstream_importance = [item['upstream_importance'] for item in attribution_data['position_attributions']]
            downstream_importance = [item['downstream_importance'] for item in attribution_data['position_attributions']]
            local_importance = [item['local_importance'] for item in attribution_data['position_attributions']]
            
            axes[0, 0].bar(['Upstream\n(UTR context)', 'Local\n(ATG)', 'Downstream\n(CDS)'], 
                          [np.mean(upstream_importance), np.mean(local_importance), np.mean(downstream_importance)])
            axes[0, 0].set_title('Average Attribution by Region')
            axes[0, 0].set_ylabel('Attribution Strength')
        
        # Plot 2: Layer feature evolution
        if feature_data['start_position_features']:
            layers = ['layer_0', 'layer_1', 'layer_2']
            mean_activations = []
            for layer in layers:
                layer_means = []
                for item in feature_data['start_position_features']:
                    if layer in item['layer_features']:
                        layer_means.append(item['layer_features'][layer]['mean_activation'])
                mean_activations.append(np.mean(layer_means) if layer_means else 0)
            
            axes[0, 1].plot(range(len(layers)), mean_activations, 'o-')
            axes[0, 1].set_title('Feature Evolution Across Layers')
            axes[0, 1].set_xlabel('Layer')
            axes[0, 1].set_ylabel('Mean Activation')
            axes[0, 1].set_xticks(range(len(layers)))
            axes[0, 1].set_xticklabels(['Layer 1', 'Layer 2', 'Layer 3'])
        
        # Plot 3: START prediction accuracy by context
        correct_predictions = sum(1 for item in attribution_data['position_attributions'] if item['predicted_correctly'])
        total_predictions = len(attribution_data['position_attributions'])
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        
        axes[1, 0].bar(['START Context\nPrediction'], [accuracy])
        axes[1, 0].set_title('START Prediction Accuracy')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].set_ylim(0, 1)
        
        # Plot 4: Attribution heatmap for sample sequence
        if attribution_data['position_attributions']:
            sample_attr = attribution_data['position_attributions'][0]
            attr_array = np.array(sample_attr['full_attribution']).reshape(1, -1)
            
            im = axes[1, 1].imshow(attr_array, aspect='auto', cmap='viridis')
            axes[1, 1].set_title(f'Attribution Pattern (Sample {sample_attr["sample_index"]})')
            axes[1, 1].set_xlabel('Sequence Position')
            axes[1, 1].set_ylabel('Attribution')
            plt.colorbar(im, ax=axes[1, 1])
        
        plt.tight_layout()
        plt.savefig(output_dir / "combined_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save summary statistics
        summary = {
            'total_samples_analyzed': len(attribution_data['position_attributions']),
            'start_prediction_accuracy': accuracy,
            'average_upstream_importance': np.mean([item['upstream_importance'] for item in attribution_data['position_attributions']]),
            'average_local_importance': np.mean([item['local_importance'] for item in attribution_data['position_attributions']]),
            'average_downstream_importance': np.mean([item['downstream_importance'] for item in attribution_data['position_attributions']])
        }
        
        with open(output_dir / "analysis_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)


def run_layer_analysis(model_path: str, dataset, output_dir: Path, max_samples: int = 20):
    """Run comprehensive layer analysis on a trained model."""
    
    # Load model
    from layout_detection.layout_model import LayoutDetectionModule
    model = LayoutDetectionModule.load_from_checkpoint(model_path)
    
    # Create data loader
    from torch.utils.data import DataLoader, random_split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=0)
    
    # Run analysis
    analyzer = LayerAnalyzer(model)
    analyzer.analyze_all(val_loader, output_dir, max_samples)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze transformer layers")
    parser.add_argument('--model-path', required=True, help='Path to saved model checkpoint')
    parser.add_argument('--output-dir', required=True, help='Output directory for analysis')
    parser.add_argument('--max-samples', type=int, default=20, help='Number of samples to analyze')
    
    args = parser.parse_args()
    
    # Would need to recreate dataset here
    print("Layer analysis tool - integrate with test drivers for full functionality")

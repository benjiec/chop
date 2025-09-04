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
from utils.constants import GenePredictionClass as P, DNAEmbed as D


class LayerAnalyzer:
    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.model.eval()
        
        # Convert indices back to nucleotides
        self.idx_to_nucleotide = {D.A: 'A', D.T: 'T', D.G: 'G', D.C: 'C', D.N: 'N'}
        self.class_names = {P.INTERGENIC: 'INTERGENIC', P.UTR5: 'UTR5', P.START: 'START'}

    def analyze_all(self, data_loader, output_dir: Path, max_samples: int = 20):
        
        output_dir.mkdir(exist_ok=True)
        sample_data = self._collect_sample_data(data_loader, max_samples)
        attention_data = self._analyze_attention_weights(sample_data)
        self._save_attention_analysis(attention_data, output_dir / "attention_weights.json")
    
    def _collect_sample_data(self, data_loader, max_samples: int) -> List[Dict]:
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
                        if atg['target_class'] == P.START:  # Real START
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
                                'predicted_correctly': atg['predicted_class'] == P.START,
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
        # Use the model's attention extraction capability
        logits, attention_weights = self.model.model(seq_tensor, return_attention=True)
        
        return attention_weights
    
    def _save_attention_analysis(self, data: Dict, filepath: Path):
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

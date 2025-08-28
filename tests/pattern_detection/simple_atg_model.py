#!/usr/bin/env python3
"""
Simple 2-class model for ATG detection.

This is a simplified version of the gene prediction model that only outputs 2 classes:
- Class 0: INTERGENIC
- Class 1: START (ATG)
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Dict, Any
import torch.nn.functional as F

class DNAEmbedding(nn.Module):
    """DNA sequence embedding layer."""
    
    def __init__(self, vocab_size: int, d_model: int, max_seq_length: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = nn.Parameter(torch.randn(max_seq_length, d_model))
        self.d_model = d_model
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_length)
        seq_length = x.size(1)
        
        # Token embeddings
        embeddings = self.embedding(x) * (self.d_model ** 0.5)  # (batch_size, seq_length, d_model)
        
        # Add positional encoding
        embeddings = embeddings + self.positional_encoding[:seq_length].unsqueeze(0)
        
        return embeddings

class SimpleATGModel(nn.Module):
    """Simple 2-class transformer model for ATG detection."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        
        # Model parameters
        vocab_size = config.get('vocab_size', 5)  # A, C, G, T, N
        d_model = config.get('d_model', 256)
        n_layers = config.get('n_layers', 4)
        n_heads = config.get('n_heads', 4)
        dropout = config.get('dropout', 0.1)
        max_seq_length = config.get('max_seq_length', 1000)
        
        # DNA embedding
        self.embedding = DNAEmbedding(vocab_size, d_model, max_seq_length)
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                batch_first=True
            ) for _ in range(n_layers)
        ])
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(d_model)
        
        # Binary classification head (2 classes: INTERGENIC, START)
        self.classifier = nn.Linear(d_model, 2)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for ATG detection.
        
        Args:
            x: DNA sequence tensor (batch_size, seq_length)
            
        Returns:
            Classification logits (batch_size, seq_length, 2)
        """
        # Embeddings
        hidden = self.embedding(x)  # (batch_size, seq_length, d_model)
        
        # Transformer layers
        for layer in self.transformer_layers:
            hidden = layer(hidden)
        
        # Layer norm and dropout
        hidden = self.layer_norm(hidden)
        hidden = self.dropout(hidden)
        
        # Classification
        logits = self.classifier(hidden)  # (batch_size, seq_length, 2)
        
        return logits

class SimpleATGModule(pl.LightningModule):
    """PyTorch Lightning module for simple ATG detection."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.save_hyperparameters()
        
        # Create model
        self.model = SimpleATGModel(config['model'])
        
        # Learning rate
        self.learning_rate = float(config.get('training', {}).get('learning_rate', 1e-4))
        
        # Class weights (if any)
        class_weights = config.get('loss', {}).get('class_weights', None)
        if class_weights:
            self.register_buffer('class_weights', torch.tensor(class_weights, dtype=torch.float))
        else:
            self.class_weights = None
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        sequences, targets = batch
        
        # Forward pass
        logits = self(sequences)  # (batch_size, seq_length, 2)
        
        # Reshape for loss computation
        logits = logits.view(-1, 2)  # (batch_size * seq_length, 2)
        targets = targets.view(-1)   # (batch_size * seq_length,)
        
        # Compute loss
        loss = F.cross_entropy(logits, targets, weight=self.class_weights)
        
        # Compute accuracy
        predictions = torch.argmax(logits, dim=1)
        accuracy = (predictions == targets).float().mean()
        
        # Compute per-class metrics
        start_mask = targets == 1
        if start_mask.sum() > 0:
            start_accuracy = (predictions[start_mask] == targets[start_mask]).float().mean()
        else:
            start_accuracy = torch.tensor(0.0)
        
        intergenic_mask = targets == 0
        if intergenic_mask.sum() > 0:
            intergenic_accuracy = (predictions[intergenic_mask] == targets[intergenic_mask]).float().mean()
        else:
            intergenic_accuracy = torch.tensor(0.0)
        
        # Log metrics
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_accuracy', accuracy, prog_bar=True)
        self.log('train_start_accuracy', start_accuracy, prog_bar=True)
        self.log('train_intergenic_accuracy', intergenic_accuracy, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        sequences, targets = batch
        
        # Forward pass
        logits = self(sequences)
        
        # Reshape for loss computation
        logits = logits.view(-1, 2)
        targets = targets.view(-1)
        
        # Compute loss
        loss = F.cross_entropy(logits, targets, weight=self.class_weights)
        
        # Compute accuracy
        predictions = torch.argmax(logits, dim=1)
        accuracy = (predictions == targets).float().mean()
        
        # Compute per-class metrics
        start_mask = targets == 1
        if start_mask.sum() > 0:
            start_accuracy = (predictions[start_mask] == targets[start_mask]).float().mean()
        else:
            start_accuracy = torch.tensor(0.0)
        
        intergenic_mask = targets == 0
        if intergenic_mask.sum() > 0:
            intergenic_accuracy = (predictions[intergenic_mask] == targets[intergenic_mask]).float().mean()
        else:
            intergenic_accuracy = torch.tensor(0.0)
        
        # Log metrics
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_accuracy', accuracy, prog_bar=True)
        self.log('val_start_accuracy', start_accuracy, prog_bar=True)
        self.log('val_intergenic_accuracy', intergenic_accuracy, prog_bar=True)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        # Same as validation step for testing
        return self.validation_step(batch, batch_idx)
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=0.01)
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=self.trainer.max_epochs,
            eta_min=self.learning_rate / 10
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss'
            }
        }

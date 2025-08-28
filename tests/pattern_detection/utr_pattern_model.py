#!/usr/bin/env python3
"""
UTR Pattern Detection Model for testing complex pattern learning.

This is a 3-class transformer model for UTR pattern detection:
- Class 0: INTERGENIC (random DNA)
- Class 1: UTR5 (5' UTR elements)
- Class 2: UTR3 (3' UTR elements)

Uses the same transformer architecture as the simple ATG model but with 3 classes.
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
from typing import Dict, Any, Optional
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from models.gene_predictor import DNAEmbedding

class UTRPatternModel(nn.Module):
    """
    Transformer model for UTR pattern detection.
    
    Architecture:
    - DNA embedding layer (converts ATGC to vectors)
    - Positional encoding
    - Multi-layer transformer encoder
    - Classification head (3 classes)
    """
    
    def __init__(self, vocab_size: int = 5, d_model: int = 256, n_layers: int = 4, 
                 n_heads: int = 4, dropout: float = 0.1, max_seq_length: int = 1000):
        super().__init__()
        
        # DNA embedding (same as gene prediction model)
        self.embedding = DNAEmbedding(vocab_size=vocab_size, d_model=d_model, max_seq_length=max_seq_length)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Classification head for 3 classes
        self.classifier = nn.Linear(d_model, 3)  # INTERGENIC, UTR5, UTR3
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, seq_length) with DNA indices
            
        Returns:
            Logits of shape (batch_size, seq_length, 3)
        """
        # Embed DNA sequence
        x = self.embedding(x)  # (batch_size, seq_length, d_model)
        
        # Apply transformer
        x = self.transformer(x)  # (batch_size, seq_length, d_model)
        
        # Apply dropout
        x = self.dropout(x)
        
        # Classify each position
        logits = self.classifier(x)  # (batch_size, seq_length, 3)
        
        return logits

class UTRPatternModule(pl.LightningModule):
    """
    PyTorch Lightning module for UTR pattern detection.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        # Store config
        self.config = config
        self.learning_rate = config['training']['learning_rate']
        
        # Create model
        model_config = config['model']
        self.model = UTRPatternModel(
            vocab_size=model_config['vocab_size'],
            d_model=model_config['d_model'],
            n_layers=model_config['n_layers'],
            n_heads=model_config['n_heads'],
            dropout=model_config['dropout'],
            max_seq_length=model_config['max_seq_length']
        )
        
        # Loss function
        loss_config = config.get('loss', {})
        class_weights = loss_config.get('class_weights')
        if class_weights is not None:
            class_weights = torch.tensor(class_weights, dtype=torch.float32)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        # Save hyperparameters for logging
        self.save_hyperparameters(config)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    
    def _calculate_metrics(self, logits: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """Calculate accuracy metrics for each class."""
        predictions = torch.argmax(logits, dim=-1)
        
        # Overall accuracy
        correct = (predictions == targets).float()
        accuracy = correct.mean()
        
        # Per-class accuracy
        intergenic_mask = (targets == 0)
        utr5_mask = (targets == 1)
        utr3_mask = (targets == 2)
        
        intergenic_accuracy = correct[intergenic_mask].mean() if intergenic_mask.any() else torch.tensor(0.0)
        utr5_accuracy = correct[utr5_mask].mean() if utr5_mask.any() else torch.tensor(0.0)
        utr3_accuracy = correct[utr3_mask].mean() if utr3_mask.any() else torch.tensor(0.0)
        
        return {
            'accuracy': accuracy,
            'intergenic_accuracy': intergenic_accuracy,
            'utr5_accuracy': utr5_accuracy,
            'utr3_accuracy': utr3_accuracy
        }
    
    def training_step(self, batch, batch_idx):
        sequences, targets = batch
        logits = self.model(sequences)
        
        # Reshape for loss calculation
        batch_size, seq_length, num_classes = logits.shape
        logits_flat = logits.view(-1, num_classes)
        targets_flat = targets.view(-1)
        
        # Calculate loss
        loss = self.criterion(logits_flat, targets_flat)
        
        # Calculate metrics
        metrics = self._calculate_metrics(logits, targets)
        
        # Log metrics
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_accuracy', metrics['accuracy'], prog_bar=True)
        self.log('train_intergenic_accuracy', metrics['intergenic_accuracy'], prog_bar=True)
        self.log('train_utr5_accuracy', metrics['utr5_accuracy'], prog_bar=True)
        self.log('train_utr3_accuracy', metrics['utr3_accuracy'], prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        sequences, targets = batch
        logits = self.model(sequences)
        
        # Reshape for loss calculation
        batch_size, seq_length, num_classes = logits.shape
        logits_flat = logits.view(-1, num_classes)
        targets_flat = targets.view(-1)
        
        # Calculate loss
        loss = self.criterion(logits_flat, targets_flat)
        
        # Calculate metrics
        metrics = self._calculate_metrics(logits, targets)
        
        # Log metrics
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_accuracy', metrics['accuracy'], prog_bar=True)
        self.log('val_intergenic_accuracy', metrics['intergenic_accuracy'], prog_bar=True)
        self.log('val_utr5_accuracy', metrics['utr5_accuracy'], prog_bar=True)
        self.log('val_utr3_accuracy', metrics['utr3_accuracy'], prog_bar=True)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        # Same as validation step for testing
        return self.validation_step(batch, batch_idx)
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=0.01)
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=self.trainer.max_epochs,
            eta_min=self.learning_rate * 0.01
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }

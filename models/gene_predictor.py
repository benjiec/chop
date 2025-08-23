"""
Gene Prediction Model using Transformer Architecture

This module implements a transformer-based model for predicting gene structures
from DNA sequences, incorporating biological constraints and multi-task learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from typing import Dict, List, Tuple, Optional
import math
import sys
from pathlib import Path

# Add utils to path for constants
sys.path.append(str(Path(__file__).parent.parent))
from utils.constants import (
    DEFAULT_VOCAB_SIZE, DEFAULT_MAX_SEQ_LENGTH, DEFAULT_D_MODEL, 
    DEFAULT_N_LAYERS, DEFAULT_N_HEADS, DEFAULT_DROPOUT, DEFAULT_THRESHOLD
)


class DNAEmbedding(nn.Module):
    """DNA sequence embedding layer with k-mer features and positional encoding."""
    
    def __init__(self, vocab_size: int = DEFAULT_VOCAB_SIZE, d_model: int = DEFAULT_D_MODEL, max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH):
        super().__init__()
        self.d_model = d_model
        self.max_seq_length = max_seq_length
        
        # DNA token embedding (A, C, G, T, N)
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, max_seq_length, d_model))
        
        # K-mer features (3-mer context)
        self.kmer_conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, seq_length)
        batch_size, seq_length = x.shape
        
        # Token embeddings
        embeddings = self.token_embedding(x)  # (batch_size, seq_length, d_model)
        
        # Add positional encoding
        if seq_length <= self.max_seq_length:
            embeddings = embeddings + self.pos_encoding[:, :seq_length, :]
        
        # K-mer context features
        kmer_features = self.kmer_conv(embeddings.transpose(1, 2)).transpose(1, 2)
        embeddings = embeddings + kmer_features
        
        # Layer normalization
        embeddings = self.layer_norm(embeddings)
        
        return embeddings


class BiologicalAttention(nn.Module):
    """Attention mechanism with biological constraints for splice sites and coding regions."""
    
    def __init__(self, d_model: int, n_heads: int = DEFAULT_N_HEADS, dropout: float = DEFAULT_DROPOUT):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_length, _ = x.shape
        
        # Linear transformations
        Q = self.w_q(x).view(batch_size, seq_length, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_length, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_length, self.n_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
        
        # Output projection
        output = self.w_o(context)
        
        return output


class GenePredictor(nn.Module):
    """Main gene prediction model with multi-task learning."""
    
    def __init__(self, 
                 vocab_size: int = DEFAULT_VOCAB_SIZE,
                 d_model: int = DEFAULT_D_MODEL,
                 n_layers: int = DEFAULT_N_LAYERS,
                 n_heads: int = DEFAULT_N_HEADS,
                 max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
                 dropout: float = DEFAULT_DROPOUT):
        super().__init__()
        
        self.d_model = d_model
        self.max_seq_length = max_seq_length
        
        # DNA embedding layer
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
        
        # Multi-task prediction heads
        self.gene_boundary_head = nn.Linear(d_model, 3)  # No gene, start, end
        self.exon_intron_head = nn.Linear(d_model, 3)    # Exon, intron, intergenic
        # splice_site_head removed - model will discover splice patterns from exon/intron boundaries
        self.coding_potential_head = nn.Linear(d_model, 1)  # Binary: coding/non-coding
        
        # Biological feature extractors
        self.splice_motif_detector = nn.Conv1d(d_model, d_model, kernel_size=5, padding=2)
        self.start_stop_detector = nn.Conv1d(d_model, d_model, kernel_size=7, padding=3)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        # x: (batch_size, seq_length)
        batch_size, seq_length = x.shape
        
        # DNA embeddings
        embeddings = self.embedding(x)
        
        # Apply transformer layers
        hidden_states = embeddings
        for layer in self.transformer_layers:
            hidden_states = layer(hidden_states, src_key_padding_mask=mask)
        
        # Multi-task predictions
        gene_boundaries = self.gene_boundary_head(hidden_states)
        exon_intron = self.exon_intron_head(hidden_states)
        # splice_sites removed - model will discover splice patterns from exon/intron boundaries
        coding_potential_logits = self.coding_potential_head(hidden_states)
        coding_potential = torch.sigmoid(coding_potential_logits)  # For other uses
        
        # Biological feature detection
        splice_features = self.splice_motif_detector(hidden_states.transpose(1, 2)).transpose(1, 2)
        start_stop_features = self.start_stop_detector(hidden_states.transpose(1, 2)).transpose(1, 2)
        
        return {
            'gene_boundaries': gene_boundaries,
            'exon_intron': exon_intron,
            # splice_sites removed - model will discover splice patterns from exon/intron boundaries
            'coding_potential': coding_potential,
            'coding_potential_logits': coding_potential_logits,
            'splice_features': splice_features,
            'start_stop_features': start_stop_features,
            'hidden_states': hidden_states,
            'sequence_tokens': x  # Include original sequence for codon validation
        }
    
    def predict_genes(self, x: torch.Tensor, threshold: float = DEFAULT_THRESHOLD) -> Dict[str, torch.Tensor]:
        """Generate final gene predictions with post-processing."""
        outputs = self.forward(x)
        
        # Apply softmax for classification tasks
        gene_boundaries = F.softmax(outputs['gene_boundaries'], dim=-1)
        exon_intron = F.softmax(outputs['exon_intron'], dim=-1)
        # splice_sites removed - model will discover splice patterns from exon/intron boundaries
        
        # Threshold-based predictions
        coding_mask = outputs['coding_potential'] > threshold
        
        return {
            'gene_boundaries': gene_boundaries,
            'exon_intron': exon_intron,
            # splice_sites removed - model will discover splice patterns from exon/intron boundaries
            'coding_potential': outputs['coding_potential'],
            'coding_mask': coding_mask
        }


class BiologicalLoss(nn.Module):
    """Loss function incorporating biological constraints."""
    
    def __init__(self, 
                 gene_weight: float = 1.0,
                 exon_weight: float = 1.0,
                 splice_weight: float = 1.0,
                 coding_weight: float = 0.5,
                 constraint_weight: float = 0.1,
                 enforce_start_stop_codons: bool = True):
        super().__init__()
        
        self.gene_weight = gene_weight
        self.exon_weight = exon_weight
        self.splice_weight = splice_weight
        self.coding_weight = coding_weight
        self.constraint_weight = constraint_weight
        self.enforce_start_stop_codons = enforce_start_stop_codons
        
        # Standard losses
        self.ce_loss = nn.CrossEntropyLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()  # More numerically stable
        
    def forward(self, predictions: Dict[str, torch.Tensor], 
                targets: Dict[str, torch.Tensor]) -> torch.Tensor:
        
        # Handle empty targets gracefully or missing keys
        required_keys = ['gene_boundaries', 'exon_intron', 'coding_potential']
        if not targets or len(targets) == 0 or not all(key in targets for key in required_keys):
            # Return a dummy loss if no targets are available
            dummy_targets = {
                'gene_boundaries': torch.zeros(predictions['gene_boundaries'].shape[0], predictions['gene_boundaries'].shape[1], dtype=torch.long),
                'exon_intron': torch.zeros(predictions['exon_intron'].shape[0], predictions['exon_intron'].shape[1], dtype=torch.long),
                # splice_sites removed - model will discover splice patterns from exon/intron boundaries
                'coding_potential': torch.zeros(predictions['coding_potential'].shape[0], predictions['coding_potential'].shape[1], dtype=torch.float32),
                'gene_ids': torch.full((predictions['gene_boundaries'].shape[0], predictions['gene_boundaries'].shape[1]), -1, dtype=torch.long)
            }
            targets = dummy_targets
        
        # Classification losses
        gene_loss = self.ce_loss(predictions['gene_boundaries'].view(-1, 3), 
                                targets['gene_boundaries'].view(-1))
        exon_loss = self.ce_loss(predictions['exon_intron'].view(-1, 3), 
                                targets['exon_intron'].view(-1))
        # splice_loss removed - model will discover splice patterns from exon/intron boundaries
        # Use logits for BCEWithLogitsLoss
        coding_loss = self.bce_loss(predictions['coding_potential_logits'].view(-1), 
                                   targets['coding_potential'].view(-1).float())
        
        # Biological constraint losses
        constraint_loss = self._biological_constraints(predictions, targets)
        
        # Total loss
        total_loss = (self.gene_weight * gene_loss + 
                     self.exon_weight * exon_loss + 
                     # splice_loss removed - model will discover splice patterns from exon/intron boundaries
                     self.coding_weight * coding_loss + 
                     self.constraint_weight * constraint_loss)
        
        return total_loss
    
    def _check_start_stop_codons(self, sequence_tokens: torch.Tensor, 
                                 gene_boundaries: torch.Tensor) -> torch.Tensor:
        """Check if predicted gene boundaries have valid start/stop codons.
        
        Note: This function assumes forward strand genes for simplicity.
        In practice, gene prediction should consider both strands, but that
        requires strand information which isn't available in the current setup.
        """
        batch_size, seq_len = sequence_tokens.shape
        device = sequence_tokens.device
        
        # DNA vocab: A=0, C=1, G=2, T=3, N=4
        # Start codon: ATG = [0, 3, 2]
        # Stop codons: TAA=[3,0,0], TAG=[3,0,2], TGA=[3,2,0]
        start_codon = torch.tensor([0, 3, 2], device=device)  # ATG
        stop_codons = torch.tensor([[3, 0, 0], [3, 0, 2], [3, 2, 0]], device=device)  # TAA, TAG, TGA
        
        # Reverse complement codons for minus strand
        # ATG reverse complement = CAT = [1, 0, 3]
        # TAA reverse complement = TTA = [3, 3, 0]
        # TAG reverse complement = CTA = [1, 3, 0] 
        # TGA reverse complement = TCA = [3, 1, 0]
        start_codon_rc = torch.tensor([1, 0, 3], device=device)  # CAT (ATG reverse complement)
        stop_codons_rc = torch.tensor([[3, 3, 0], [1, 3, 0], [3, 1, 0]], device=device)  # TTA, CTA, TCA
        
        codon_penalty = 0.0
        
        # Get predicted gene start and end positions
        gene_starts = (gene_boundaries[:, :, 1] > 0.5)  # GeneBoundaryClass.START = 1
        gene_ends = (gene_boundaries[:, :, 2] > 0.5)    # GeneBoundaryClass.END = 2
        
        for batch_idx in range(batch_size):
            # Find start positions
            start_positions = torch.where(gene_starts[batch_idx])[0]
            for start_pos in start_positions:
                if start_pos + 2 < seq_len:  # Ensure we can read 3 nucleotides
                    predicted_codon = sequence_tokens[batch_idx, start_pos:start_pos+3]
                    # Check both forward and reverse strand start codons
                    valid_start = (torch.equal(predicted_codon, start_codon) or 
                                 torch.equal(predicted_codon, start_codon_rc))
                    if not valid_start:
                        codon_penalty += 1.0
            
            # Find end positions  
            end_positions = torch.where(gene_ends[batch_idx])[0]
            for end_pos in end_positions:
                if end_pos >= 3:  # Ensure we can read 3 nucleotides before end
                    predicted_codon = sequence_tokens[batch_idx, end_pos-3:end_pos]
                    # Check if matches any stop codon (forward or reverse)
                    is_valid_stop = False
                    for stop_codon in stop_codons:
                        if torch.equal(predicted_codon, stop_codon):
                            is_valid_stop = True
                            break
                    if not is_valid_stop:
                        for stop_codon_rc in stop_codons_rc:
                            if torch.equal(predicted_codon, stop_codon_rc):
                                is_valid_stop = True
                                break
                    if not is_valid_stop:
                        codon_penalty += 1.0
        
        return torch.tensor(codon_penalty, device=device)

    def _biological_constraints(self, predictions: Dict[str, torch.Tensor], 
                               targets: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Apply biological constraints to the loss function."""
        
        # Constraint 1: Start codons should be followed by coding regions
        start_codons = (predictions['gene_boundaries'][:, :, 1] > 0.5).float()
        coding_regions = (predictions['coding_potential'].squeeze(-1) > 0.5).float()
        
        # Constraint 2: Exons should be within coding regions
        exon_regions = (predictions['exon_intron'][:, :, 1] > 0.5).float()  # ExonIntronClass.EXON = 1
        
        # Constraint 3: Start/stop codon validation (if enabled and sequence tokens available)
        codon_constraint = torch.tensor(0.0, device=predictions['gene_boundaries'].device)
        if self.enforce_start_stop_codons and 'sequence_tokens' in predictions:
            codon_constraint = self._check_start_stop_codons(
                predictions['sequence_tokens'], 
                predictions['gene_boundaries']
            )
        
        # Simple constraint: penalize biologically impossible combinations
        constraint_loss = (torch.mean(start_codons * (1 - coding_regions)) + 
                          torch.mean(exon_regions * (1 - coding_regions)) +  # Exons should be in coding regions
                          codon_constraint * 0.01)  # Reduced weight for codon constraint
        
        return constraint_loss


def create_model(config: Dict) -> GenePredictor:
    """Factory function to create a gene prediction model."""
    return GenePredictor(
        vocab_size=config.get('vocab_size', DEFAULT_VOCAB_SIZE),
        d_model=config.get('d_model', DEFAULT_D_MODEL),
        n_layers=config.get('n_layers', DEFAULT_N_LAYERS),
        n_heads=config.get('n_heads', DEFAULT_N_HEADS),
        max_seq_length=config.get('max_seq_length', DEFAULT_MAX_SEQ_LENGTH),
        dropout=config.get('dropout', DEFAULT_DROPOUT)
    )

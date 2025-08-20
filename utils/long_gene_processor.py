"""
Long Gene Processing Utilities

Handles genes longer than the model's max_sequence_length by using
overlapping windows and merging predictions.
"""

import numpy as np
import torch
from typing import List, Dict, Tuple
from utils.dna_processor import create_sequence_windows


class LongGeneProcessor:
    """Processes long genes using overlapping windows and prediction merging."""
    
    def __init__(self, window_size: int = 4096, stride: int = 2048, min_overlap: int = 500):
        self.window_size = window_size
        self.stride = stride
        self.min_overlap = min_overlap
    
    def process_long_sequence(self, sequence: str, model, threshold: float = 0.5) -> Dict:
        """Process a long sequence using overlapping windows."""
        
        if len(sequence) <= self.window_size:
            # Short sequence - process normally
            return self._predict_single_window(sequence, model, threshold)
        
        # Create overlapping windows
        windows = create_sequence_windows(sequence, self.window_size, self.stride)
        window_predictions = []
        
        # Process each window
        for i, window in enumerate(windows):
            start_pos = i * self.stride
            pred = self._predict_single_window(window, model, threshold)
            
            # Adjust positions to global coordinates
            pred = self._adjust_positions(pred, start_pos)
            window_predictions.append(pred)
        
        # Merge overlapping predictions
        merged_predictions = self._merge_predictions(window_predictions, len(sequence))
        
        return merged_predictions
    
    def _predict_single_window(self, sequence: str, model, threshold: float) -> Dict:
        """Make predictions on a single window."""
        # This would call your existing prediction logic
        # For now, return a mock structure
        return {
            'genes': [],
            'exons': [],
            'introns': [],
            'splice_sites': {'donor_sites': [], 'acceptor_sites': []},
            'coding_regions': []
        }
    
    def _adjust_positions(self, predictions: Dict, offset: int) -> Dict:
        """Adjust prediction positions by offset."""
        adjusted = predictions.copy()
        
        # Adjust gene positions
        for gene in adjusted.get('genes', []):
            gene['start'] += offset
            gene['end'] += offset
        
        # Adjust exon positions
        for exon in adjusted.get('exons', []):
            exon['start'] += offset
            exon['end'] += offset
        
        # Adjust intron positions
        for intron in adjusted.get('introns', []):
            intron['start'] += offset
            intron['end'] += offset
        
        # Adjust splice sites
        splice_sites = adjusted.get('splice_sites', {})
        if 'donor_sites' in splice_sites:
            splice_sites['donor_sites'] = [pos + offset for pos in splice_sites['donor_sites']]
        if 'acceptor_sites' in splice_sites:
            splice_sites['acceptor_sites'] = [pos + offset for pos in splice_sites['acceptor_sites']]
        
        # Adjust coding regions
        for region in adjusted.get('coding_regions', []):
            region['start'] += offset
            region['end'] += offset
        
        return adjusted
    
    def _merge_predictions(self, predictions: List[Dict], total_length: int) -> Dict:
        """Merge predictions from overlapping windows."""
        merged = {
            'sequence_length': total_length,
            'genes': [],
            'exons': [],
            'introns': [],
            'splice_sites': {'donor_sites': [], 'acceptor_sites': []},
            'coding_regions': []
        }
        
        # Collect all predictions
        all_genes = []
        all_exons = []
        all_introns = []
        all_donors = []
        all_acceptors = []
        all_coding = []
        
        for pred in predictions:
            all_genes.extend(pred.get('genes', []))
            all_exons.extend(pred.get('exons', []))
            all_introns.extend(pred.get('introns', []))
            all_donors.extend(pred.get('splice_sites', {}).get('donor_sites', []))
            all_acceptors.extend(pred.get('splice_sites', {}).get('acceptor_sites', []))
            all_coding.extend(pred.get('coding_regions', []))
        
        # Merge overlapping regions
        merged['genes'] = self._merge_regions(all_genes)
        merged['exons'] = self._merge_regions(all_exons)
        merged['introns'] = self._merge_regions(all_introns)
        merged['coding_regions'] = self._merge_regions(all_coding)
        
        # Merge splice sites (remove duplicates within min_overlap distance)
        merged['splice_sites']['donor_sites'] = self._merge_positions(all_donors)
        merged['splice_sites']['acceptor_sites'] = self._merge_positions(all_acceptors)
        
        return merged
    
    def _merge_regions(self, regions: List[Dict]) -> List[Dict]:
        """Merge overlapping regions."""
        if not regions:
            return []
        
        # Sort by start position
        sorted_regions = sorted(regions, key=lambda x: x['start'])
        merged = []
        
        current = sorted_regions[0].copy()
        
        for region in sorted_regions[1:]:
            if region['start'] <= current['end'] + self.min_overlap:
                # Overlapping - merge
                current['end'] = max(current['end'], region['end'])
                current['length'] = current['end'] - current['start']
            else:
                # Non-overlapping - add current and start new
                merged.append(current)
                current = region.copy()
        
        merged.append(current)
        return merged
    
    def _merge_positions(self, positions: List[int]) -> List[int]:
        """Merge nearby positions (remove duplicates within min_overlap)."""
        if not positions:
            return []
        
        sorted_pos = sorted(set(positions))
        merged = []
        
        for pos in sorted_pos:
            if not merged or pos > merged[-1] + self.min_overlap:
                merged.append(pos)
        
        return merged


def reconstruct_long_genes(predictions: Dict, min_gene_length: int = 1000) -> Dict:
    """
    Post-process predictions to reconstruct long genes that might have been
    split across windows.
    """
    
    # Find potential gene fragments that could be parts of longer genes
    genes = predictions.get('genes', [])
    exons = predictions.get('exons', [])
    
    # Group nearby exons that could belong to the same gene
    reconstructed_genes = []
    
    if exons:
        # Sort exons by position
        sorted_exons = sorted(exons, key=lambda x: x['start'])
        
        current_gene_exons = [sorted_exons[0]]
        
        for exon in sorted_exons[1:]:
            # If exon is within reasonable distance, add to current gene
            if exon['start'] - current_gene_exons[-1]['end'] < 10000:  # 10kb max intron
                current_gene_exons.append(exon)
            else:
                # Start new gene
                if len(current_gene_exons) > 1:  # Multi-exon gene
                    gene_start = current_gene_exons[0]['start']
                    gene_end = current_gene_exons[-1]['end']
                    if gene_end - gene_start >= min_gene_length:
                        reconstructed_genes.append({
                            'start': gene_start,
                            'end': gene_end,
                            'length': gene_end - gene_start,
                            'exon_count': len(current_gene_exons),
                            'exons': current_gene_exons
                        })
                
                current_gene_exons = [exon]
        
        # Don't forget the last gene
        if len(current_gene_exons) > 1:
            gene_start = current_gene_exons[0]['start']
            gene_end = current_gene_exons[-1]['end']
            if gene_end - gene_start >= min_gene_length:
                reconstructed_genes.append({
                    'start': gene_start,
                    'end': gene_end,
                    'length': gene_end - gene_start,
                    'exon_count': len(current_gene_exons),
                    'exons': current_gene_exons
                })
    
    # Update predictions with reconstructed genes
    result = predictions.copy()
    result['reconstructed_genes'] = reconstructed_genes
    result['long_genes'] = [g for g in reconstructed_genes if g['length'] >= 8000]
    
    return result

"""
DNA Sequence Processing Utilities

This module provides utilities for processing DNA sequences, including
tokenization, data loading, and biological feature extraction.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from Bio import SeqIO
from Bio.Seq import Seq
import re
import csv
import os
import pickle
import hashlib
import mmap
from collections import defaultdict
from pathlib import Path

from .constants import (
    GeneBoundaryClass, ExonIntronClass, SpliceSiteClass, DNA_VOCAB,
    DEFAULT_WINDOW_SIZE, DEFAULT_STRIDE, DEFAULT_MIN_GENE_COVERAGE,
    TSV_COLUMNS, UNKNOWN_GENE_ID, MAX_GENES_PER_WINDOW,
    DEFAULT_CACHE_DIR, DEFAULT_MAX_CACHE_SIZE_GB
)

# Handle different BioPython versions for GC content calculation
try:
    from Bio.SeqUtils import GC
except ImportError:
    try:
        from Bio.SeqUtils.ProtParam import ProteinAnalysis
        # Fallback: calculate GC content manually
        def GC(sequence):
            """Calculate GC content manually if Bio.SeqUtils.GC is not available."""
            sequence = sequence.upper()
            gc_count = sequence.count('G') + sequence.count('C')
            total_count = len([base for base in sequence if base in 'ATGC'])
            return (gc_count / total_count * 100) if total_count > 0 else 0
    except ImportError:
        # Manual GC calculation as final fallback
        def GC(sequence):
            """Calculate GC content manually."""
            sequence = sequence.upper()
            gc_count = sequence.count('G') + sequence.count('C')
            total_count = len([base for base in sequence if base in 'ATGC'])
            return (gc_count / total_count * 100) if total_count > 0 else 0


class DNATokenizer:
    """Tokenizes DNA sequences for the transformer model."""
    
    def __init__(self, possible_donor_motifs=None):
        # DNA vocabulary: A, C, G, T, N (unknown/ambiguous)
        self.vocab = DNA_VOCAB
        self.vocab_size = len(self.vocab)
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
        
        # Start and stop codons
        self.start_codons = {'ATG'}  # Common start codons
        self.stop_codons = {'TAA', 'TAG', 'TGA'}   # Stop codons
        
        # Splice site motifs
        if possible_donor_motifs is None:
            self.donor_motifs = ['GT', 'GC', 'GA']
        else:
            self.donor_motifs = possible_donor_motifs
        self.acceptor_motifs = ['AG']

    def tokenize(self, sequence: str) -> torch.Tensor:
        """Convert DNA sequence string to token indices. Treats any non-ATGC bases as N."""
        # Convert to uppercase
        sequence = sequence.upper()
        
        # Convert to indices - any non-ATGC base becomes N (index 4)
        tokens = [self.vocab.get(base, 4) for base in sequence]  # Default to N for any unknown base
        return torch.tensor(tokens, dtype=torch.long)
    
    def detokenize(self, tokens: torch.Tensor) -> str:
        """Convert token indices back to DNA sequence string."""
        return ''.join([self.reverse_vocab[int(token)] for token in tokens])
    
    def find_start_codons(self, sequence: str) -> List[int]:
        """Find positions of start codons in the sequence."""
        positions = []
        for i in range(len(sequence) - 2):
            codon = sequence[i:i+3]
            if codon in self.start_codons:
                positions.append(i)
        return positions
    
    def find_stop_codons(self, sequence: str) -> List[int]:
        """Find positions of stop codons in the sequence."""
        positions = []
        for i in range(len(sequence) - 2):
            codon = sequence[i:i+3]
            if codon in self.stop_codons:
                positions.append(i)
        return positions
    
    def find_splice_sites(self, sequence: str) -> Tuple[List[int], List[int]]:
        """Find donor and acceptor splice sites."""
        donor_positions = []
        acceptor_positions = []
        
        # Find donor sites (5' splice sites)
        for i in range(len(sequence) - 1):
            dinucleotide = sequence[i:i+2]
            if dinucleotide in self.donor_motifs:
                donor_positions.append(i)
        
        # Find acceptor sites (3' splice sites)
        for i in range(len(sequence) - 1):
            dinucleotide = sequence[i:i+2]
            if dinucleotide in self.acceptor_motifs:
                acceptor_positions.append(i)
        
        return donor_positions, acceptor_positions


class BiologicalFeatureExtractor:
    """Extracts biological features from DNA sequences."""
    
    def __init__(self):
        self.tokenizer = DNATokenizer()
        
    def extract_features(self, sequence: str) -> Dict[str, np.ndarray]:
        """Extract comprehensive biological features from a DNA sequence."""
        features = {}
        
        # Basic sequence features
        features['gc_content'] = self._calculate_gc_content(sequence)
        features['length'] = len(sequence)
        
        # Codon features
        features['start_codons'] = self._find_start_codons_binary(sequence)
        features['stop_codons'] = self._find_stop_codons_binary(sequence)
        
        # Splice site features
        features['donor_sites'] = self._find_donor_sites_binary(sequence)
        features['acceptor_sites'] = self._find_acceptor_sites_binary(sequence)
        
        # Reading frame features
        features['reading_frames'] = self._analyze_reading_frames(sequence)
        
        # K-mer frequencies
        features['kmer_frequencies'] = self._calculate_kmer_frequencies(sequence, k=3)
        
        return features
    
    def _calculate_gc_content(self, sequence: str) -> float:
        """Calculate GC content of the sequence."""
        return GC(sequence)
    
    def _find_start_codons_binary(self, sequence: str) -> np.ndarray:
        """Create binary array indicating start codon positions."""
        binary = np.zeros(len(sequence), dtype=np.int8)
        positions = self.tokenizer.find_start_codons(sequence)
        for pos in positions:
            binary[pos:pos+3] = 1
        return binary
    
    def _find_stop_codons_binary(self, sequence: str) -> np.ndarray:
        """Create binary array indicating stop codon positions."""
        binary = np.zeros(len(sequence), dtype=np.int8)
        positions = self.tokenizer.find_stop_codons(sequence)
        for pos in positions:
            binary[pos:pos+3] = 1
        return binary
    
    def _find_donor_sites_binary(self, sequence: str) -> np.ndarray:
        """Create binary array indicating donor splice site positions."""
        binary = np.zeros(len(sequence), dtype=np.int8)
        donor_positions, _ = self.tokenizer.find_splice_sites(sequence)
        for pos in donor_positions:
            binary[pos:pos+2] = 1
        return binary
    
    def _find_acceptor_sites_binary(self, sequence: str) -> np.ndarray:
        """Create binary array indicating acceptor splice site positions."""
        binary = np.zeros(len(sequence), dtype=np.int8)
        _, acceptor_positions = self.tokenizer.find_splice_sites(sequence)
        for pos in acceptor_positions:
            binary[pos:pos+2] = 1
        return binary
    
    def _analyze_reading_frames(self, sequence: str) -> Dict[str, np.ndarray]:
        """Analyze coding potential in different reading frames."""
        frames = {}
        
        for frame in range(3):
            # Extract codons for this frame
            frame_sequence = sequence[frame:]
            codons = [frame_sequence[i:i+3] for i in range(0, len(frame_sequence) - 2, 3)]
            
            # Calculate coding potential (simplified)
            coding_potential = np.zeros(len(sequence), dtype=np.float32)
            
            for i, codon in enumerate(codons):
                start_pos = frame + i * 3
                if codon in self.tokenizer.start_codons:
                    coding_potential[start_pos:start_pos+3] = 1.0
                elif codon in self.tokenizer.stop_codons:
                    coding_potential[start_pos:start_pos+3] = -1.0
                else:
                    # Simple heuristic: check if codon is common in coding regions
                    coding_potential[start_pos:start_pos+3] = 0.1
            
            frames[f'frame_{frame}'] = coding_potential
        
        return frames
    
    def _calculate_kmer_frequencies(self, sequence: str, k: int = 3) -> Dict[str, float]:
        """Calculate k-mer frequencies in the sequence."""
        kmer_counts = {}
        total_kmers = len(sequence) - k + 1
        
        for i in range(total_kmers):
            kmer = sequence[i:i+k]
            kmer_counts[kmer] = kmer_counts.get(kmer, 0) + 1
        
        # Convert to frequencies
        kmer_frequencies = {kmer: count / total_kmers for kmer, count in kmer_counts.items()}
        return kmer_frequencies


class DataCache:
    """Intelligent caching system for processed sequences and annotations."""
    
    def __init__(self, cache_dir: str = DEFAULT_CACHE_DIR, max_size_gb: float = DEFAULT_MAX_CACHE_SIZE_GB):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.max_size_bytes = max_size_gb * 1024**3
        
    def _get_file_hash(self, file_path: str) -> str:
        """Generate hash of file for cache invalidation."""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def _get_cache_path(self, key: str) -> Path:
        """Get cache file path for a given key."""
        return self.cache_dir / f"{key}.pkl"
    
    def _cleanup_cache(self) -> None:
        """Remove old cache files if cache size exceeds limit."""
        cache_files = list(self.cache_dir.glob("*.pkl"))
        total_size = sum(f.stat().st_size for f in cache_files)
        
        if total_size > self.max_size_bytes:
            # Sort by modification time and remove oldest
            cache_files.sort(key=lambda f: f.stat().st_mtime)
            
            while total_size > self.max_size_bytes and cache_files:
                oldest_file = cache_files.pop(0)
                file_size = oldest_file.stat().st_size
                oldest_file.unlink()
                total_size -= file_size
                print(f"Removed old cache file: {oldest_file.name}")
    
    def get(self, file_path: str, processing_params: Dict = None) -> Optional[any]:
        """Get cached data if available and valid."""
        try:
            file_hash = self._get_file_hash(file_path)
            cache_key = f"{Path(file_path).stem}_{file_hash}"
            
            if processing_params:
                # Include processing parameters in cache key
                params_str = str(sorted(processing_params.items()))
                params_hash = hashlib.md5(params_str.encode()).hexdigest()[:8]
                cache_key += f"_{params_hash}"
            
            cache_path = self._get_cache_path(cache_key)
            
            if cache_path.exists():
                with open(cache_path, 'rb') as f:
                    cached_data = pickle.load(f)
                print(f"Loaded from cache: {cache_path.name}")
                return cached_data
                
        except Exception as e:
            print(f"Cache read error: {e}")
            
        return None
    
    def set(self, file_path: str, data: any, processing_params: Dict = None) -> None:
        """Cache processed data."""
        try:
            file_hash = self._get_file_hash(file_path)
            cache_key = f"{Path(file_path).stem}_{file_hash}"
            
            if processing_params:
                params_str = str(sorted(processing_params.items()))
                params_hash = hashlib.md5(params_str.encode()).hexdigest()[:8]
                cache_key += f"_{params_hash}"
            
            cache_path = self._get_cache_path(cache_key)
            
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
                
            print(f"Cached to: {cache_path.name}")
            
            # Cleanup if needed
            self._cleanup_cache()
            
        except Exception as e:
            print(f"Cache write error: {e}")


def load_fasta_sequences_cached(file_path: str, use_cache: bool = True) -> List[str]:
    """Load DNA sequences from a FASTA file with caching. No validation - accepts all sequences."""
    
    cache = DataCache() if use_cache else None
    
    # Try cache first
    if cache:
        cached_sequences = cache.get(file_path)
        if cached_sequences is not None:
            return cached_sequences
    
    sequences = []
    
    for record in SeqIO.parse(file_path, "fasta"):
        sequence = str(record.seq)
        sequences.append(sequence)
        
    print(f"Loaded {len(sequences)} sequences from {file_path}")
    
    # Cache the results
    if cache:
        cache.set(file_path, sequences)
    
    return sequences


class DNADataset:
    """Dataset class for DNA sequences with biological annotations."""
    
    def __init__(self, sequences: List[str], annotations: Optional[List[Dict]] = None, 
                 max_length: int = 8192, tokenizer: Optional[DNATokenizer] = None,
                 use_sliding_windows: bool = False, window_size: int = None, 
                 stride: int = None, min_gene_coverage: float = 0.5,
                 enable_augmentation: bool = False, augmentation_params: Dict = None):
        self.original_sequences = sequences
        self.original_annotations = annotations or []
        self.max_length = max_length
        self.tokenizer = tokenizer or DNATokenizer()
        self.feature_extractor = BiologicalFeatureExtractor()
        
        # Sliding window parameters
        self.use_sliding_windows = use_sliding_windows
        self.window_size = window_size or max_length
        self.stride = stride or (window_size // 2 if window_size else max_length // 2)
        self.min_gene_coverage = min_gene_coverage
        
        # Data augmentation
        self.enable_augmentation = enable_augmentation
        self.augmentation = DataAugmentation(**(augmentation_params or {})) if enable_augmentation else None
        
        # Process sequences into windows
        self.sequences = []
        self.annotations = []
        self.window_metadata = []  # Track which window belongs to which original sequence
        
        if use_sliding_windows:
            self._create_sliding_windows()
        else:
            self.sequences = self.original_sequences
            self.annotations = self.original_annotations
            self.window_metadata = [{'original_idx': i, 'window_idx': 0, 'start_pos': 0} 
                                  for i in range(len(self.original_sequences))]
        
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sequence = self.sequences[idx]
        annotation = self.annotations[idx] if idx < len(self.annotations) else {}
        
        # Apply data augmentation if enabled
        if self.enable_augmentation and self.augmentation:
            sequence, annotation = self.augmentation.augment_sequence(sequence, annotation)
        
        # Tokenize sequence
        tokens = self.tokenizer.tokenize(sequence)
        
        # Pad or truncate to max_length
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        elif len(tokens) < self.max_length:
            padding = torch.zeros(self.max_length - len(tokens), dtype=torch.long)
            tokens = torch.cat([tokens, padding])
        
        # Create attention mask
        attention_mask = torch.ones(self.max_length, dtype=torch.bool)
        if len(sequence) < self.max_length:
            attention_mask[len(sequence):] = False
        
        # Extract biological features - only return scalars and fixed-size arrays
        features = self.feature_extractor.extract_features(sequence)
        
        # Convert features to tensors and pad/truncate to max_length
        processed_features = {}
        for key, value in features.items():
            if isinstance(value, np.ndarray):
                # Pad or truncate arrays to max_length
                if len(value) > self.max_length:
                    value = value[:self.max_length]
                elif len(value) < self.max_length:
                    padding = np.zeros(self.max_length - len(value), dtype=value.dtype)
                    value = np.concatenate([value, padding])
                processed_features[key] = torch.from_numpy(value)
            elif isinstance(value, dict):
                # Skip dictionary features for now (like kmer_frequencies)
                continue
            else:
                # Scalar values
                processed_features[key] = torch.tensor(float(value))
        
        # Prepare targets if annotations exist
        targets = {}
        if annotation:
            targets = self._prepare_targets(sequence, annotation)
        else:
            # Create empty targets for sequences without annotations
            targets = {
                'gene_boundaries': torch.zeros(self.max_length, dtype=torch.long),
                'exon_intron': torch.zeros(self.max_length, dtype=torch.long),
                # splice_sites removed - model will discover from exon/intron boundaries
                'coding_potential': torch.zeros(self.max_length, dtype=torch.float32),
                'gene_ids': torch.full((self.max_length,), UNKNOWN_GENE_ID, dtype=torch.long)
            }
        
        return {
            'input_ids': tokens,
            'attention_mask': attention_mask,
            'sequence_length': torch.tensor(len(sequence)),
            'features': processed_features,
            'targets': targets
        }
    
    def _prepare_targets(self, sequence: str, annotation: Dict) -> Dict[str, torch.Tensor]:
        """Prepare target tensors from annotations with gene ID tracking."""
        targets = {}
        
        # Initialize target arrays - use max_length for consistency
        seq_length = min(len(sequence), self.max_length)
        gene_boundaries = torch.zeros(self.max_length, dtype=torch.long)  # Class indices for CrossEntropyLoss
        exon_intron = torch.zeros(self.max_length, dtype=torch.long)       # Class indices for CrossEntropyLoss
        # splice_sites removed - let model discover from exon/intron boundaries
        coding_potential = torch.zeros(self.max_length, dtype=torch.float32)  # Binary for BCELoss
        gene_ids = torch.full((self.max_length,), UNKNOWN_GENE_ID, dtype=torch.long)  # Gene ID tracking
        
        # Create gene ID mapping for this window
        gene_id_map = {}  # gene_id_string -> integer_id
        next_gene_id = 0
        
        # Fill targets based on annotation
        if 'genes' in annotation:
            for gene_idx, gene in enumerate(annotation['genes']):
                start = gene.get('start', 0)
                end = gene.get('end', seq_length)
                gene_id_str = gene.get('gene_id', f"gene_{start}_{end}")
                
                # Assign integer gene ID (use gene_idx for consistent mapping within window)
                gene_id_int = gene_idx
                
                # Gene boundaries
                if start < self.max_length:
                    gene_boundaries[start] = GeneBoundaryClass.START
                if end < self.max_length:
                    gene_boundaries[end] = GeneBoundaryClass.END
                    
                # Fill gene ID for entire gene region
                gene_start_pos = max(0, start)
                gene_end_pos = min(self.max_length, end)
                gene_ids[gene_start_pos:gene_end_pos] = gene_id_int
                
                # Exon/intron structure
                if 'exons' in gene:
                    for exon in gene['exons']:
                        exon_start = max(0, exon.get('start', start))
                        exon_end = min(self.max_length, exon.get('end', end))
                        exon_intron[exon_start:exon_end] = ExonIntronClass.EXON
                
                # Intron regions
                if 'introns' in gene:
                    for intron in gene['introns']:
                        intron_start = max(0, intron.get('start', 0))
                        intron_end = min(self.max_length, intron.get('end', 0))
                        if intron_start < intron_end:  # Valid intron
                            exon_intron[intron_start:intron_end] = ExonIntronClass.INTRON
                
                # Splice sites removed - model will discover from exon/intron boundaries
                
                # Coding potential
                coding_start = max(0, gene.get('coding_start', start))
                coding_end = min(self.max_length, gene.get('coding_end', end))
                coding_potential[coding_start:coding_end] = 1
        
        targets['gene_boundaries'] = gene_boundaries
        targets['exon_intron'] = exon_intron
        # splice_sites removed - model will discover from exon/intron boundaries
        targets['coding_potential'] = coding_potential
        targets['gene_ids'] = gene_ids
        
        return targets
    
    def _create_sliding_windows(self) -> None:
        """Create sliding windows from original sequences with annotation mapping."""
        
        for seq_idx, sequence in enumerate(self.original_sequences):
            if len(sequence) <= self.window_size:
                # Sequence fits in single window
                self.sequences.append(sequence)
                if seq_idx < len(self.original_annotations):
                    self.annotations.append(self.original_annotations[seq_idx])
                else:
                    self.annotations.append({})
                self.window_metadata.append({
                    'original_idx': seq_idx,
                    'window_idx': 0,
                    'start_pos': 0,
                    'end_pos': len(sequence)
                })
            else:
                # Create sliding windows
                window_idx = 0
                for start_pos in range(0, len(sequence) - self.window_size + 1, self.stride):
                    end_pos = start_pos + self.window_size
                    window_seq = sequence[start_pos:end_pos]
                    
                    self.sequences.append(window_seq)
                    
                    # Map annotations to this window
                    if seq_idx < len(self.original_annotations):
                        window_annotation = self._map_annotations_to_window(
                            self.original_annotations[seq_idx], start_pos, end_pos
                        )
                        self.annotations.append(window_annotation)
                    else:
                        self.annotations.append({})
                        
                    self.window_metadata.append({
                        'original_idx': seq_idx,
                        'window_idx': window_idx,
                        'start_pos': start_pos,
                        'end_pos': end_pos
                    })
                    
                    window_idx += 1
                    
                # Add final window if sequence doesn't end exactly at window boundary
                if len(sequence) > self.window_size:
                    final_start = len(sequence) - self.window_size
                    if final_start > start_pos:  # Avoid duplicate if already covered
                        final_window = sequence[final_start:]
                        self.sequences.append(final_window)
                        
                        if seq_idx < len(self.original_annotations):
                            window_annotation = self._map_annotations_to_window(
                                self.original_annotations[seq_idx], final_start, len(sequence)
                            )
                            self.annotations.append(window_annotation)
                        else:
                            self.annotations.append({})
                            
                        self.window_metadata.append({
                            'original_idx': seq_idx,
                            'window_idx': window_idx,
                            'start_pos': final_start,
                            'end_pos': len(sequence)
                        })
    
    def _map_annotations_to_window(self, annotation: Dict, window_start: int, window_end: int) -> Dict:
        """Map gene annotations to a specific window, filtering by gene coverage."""
        
        if not annotation or 'genes' not in annotation:
            return {}
            
        window_annotation = {'genes': []}
        
        for gene in annotation['genes']:
            gene_start = gene.get('start', 0)
            gene_end = gene.get('end', 0)
            
            # Calculate overlap between gene and window
            overlap_start = max(gene_start, window_start)
            overlap_end = min(gene_end, window_end)
            
            if overlap_start >= overlap_end:
                continue  # No overlap
                
            # Calculate gene coverage in this window
            gene_length = gene_end - gene_start
            overlap_length = overlap_end - overlap_start
            coverage = overlap_length / gene_length if gene_length > 0 else 0
            
            # For genes that extend beyond the sequence, calculate coverage differently
            effective_gene_end = min(gene_end, window_end)
            effective_gene_start = max(gene_start, window_start)
            effective_gene_length = max(0, effective_gene_end - effective_gene_start)
            
            if effective_gene_length == 0 or coverage < self.min_gene_coverage:
                continue  # Insufficient coverage
                
            # Adjust gene coordinates to window-relative positions
            window_gene = {
                'start': max(0, gene_start - window_start),
                'end': min(self.window_size, gene_end - window_start),
                'strand': gene.get('strand', '+'),
                'gene_id': gene.get('gene_id', f"gene_{gene_start}_{gene_end}"),
                'coverage': coverage,
                'exons': [],
                'introns': []
            }
            
            # Map exons to window coordinates
            if 'exons' in gene:
                for exon in gene['exons']:
                    exon_start = exon.get('start', gene_start)
                    exon_end = exon.get('end', gene_end)
                    
                    # Check if exon overlaps with window
                    if exon_end > window_start and exon_start < window_end:
                        window_exon = {
                            'start': max(0, exon_start - window_start),
                            'end': min(self.window_size, exon_end - window_start)
                        }
                        window_gene['exons'].append(window_exon)
                        
            # Map introns to window coordinates
            if 'introns' in gene:
                for intron in gene['introns']:
                    intron_start = intron.get('start', 0)
                    intron_end = intron.get('end', 0)
                    
                    # Check if intron overlaps with window
                    if intron_end > window_start and intron_start < window_end:
                        window_intron = {
                            'start': max(0, intron_start - window_start),
                            'end': min(self.window_size, intron_end - window_start),
                            'donor_pos': max(0, intron.get('donor_pos', intron_start) - window_start),
                            'acceptor_pos': min(self.window_size - 1, intron.get('acceptor_pos', intron_end - 1) - window_start)
                        }
                        window_gene['introns'].append(window_intron)
                        
            window_annotation['genes'].append(window_gene)
            
        return window_annotation


# Validation functions removed - data preparation is separate from training


def load_fasta_sequences_with_ids(file_path: str) -> List[Tuple[str, str]]:
    """Load DNA sequences from a FASTA file with sequence IDs. No validation - accepts all sequences."""
    sequences_with_ids = []
    
    for record in SeqIO.parse(file_path, "fasta"):
        sequence = str(record.seq)
        sequence_id = record.id
        sequences_with_ids.append((sequence_id, sequence))
        
    print(f"Loaded {len(sequences_with_ids)} sequences with IDs from {file_path}")
    
    # Cache the results
    cache = DataCache()
    cache.set(file_path, sequences_with_ids)
    
    return sequences_with_ids


def load_fasta_sequences(file_path: str) -> List[str]:
    """Load DNA sequences from a FASTA file. No validation - accepts all sequences."""
    sequences = []
    
    for record in SeqIO.parse(file_path, "fasta"):
        sequence = str(record.seq)
        sequences.append(sequence)
        
    print(f"Loaded {len(sequences)} sequences from {file_path}")
    return sequences



def load_tsv_annotations(file_path: str) -> List[Dict]:
    """Load gene annotations from TSV file in standardized format. Returns sequence objects with their genes."""
    print(f"Loading TSV annotations from {file_path}")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"TSV file not found: {file_path}")
    
    # Group exons by gene_id
    genes_data = defaultdict(lambda: {
        'sequence_id': None,
        'gene_start': float('inf'),
        'gene_end': 0,
        'strand': '+',
        'exons': []
    })
    
    with open(file_path, 'r', newline='') as f:
        reader = csv.DictReader(f, delimiter='\t')
        
        # Validate header
        if reader.fieldnames != TSV_COLUMNS:
            raise ValueError(f"Invalid TSV header. Expected: {TSV_COLUMNS}, Got: {reader.fieldnames}")
        
        for row_num, row in enumerate(reader, 2):  # Start at 2 since header is row 1
            try:
                gene_id = row['gene_id']
                sequence_id = row['sequence_id']
                gene_start = int(row['gene_start'])
                gene_end = int(row['gene_end'])
                exon_start = int(row['exon_start'])
                exon_end = int(row['exon_end'])
                strand = row['strand']
                
                # Update gene boundaries
                genes_data[gene_id]['sequence_id'] = sequence_id
                genes_data[gene_id]['gene_start'] = min(genes_data[gene_id]['gene_start'], gene_start)
                genes_data[gene_id]['gene_end'] = max(genes_data[gene_id]['gene_end'], gene_end)
                genes_data[gene_id]['strand'] = strand
                
                # Add exon
                genes_data[gene_id]['exons'].append({
                    'start': exon_start,
                    'end': exon_end
                })
                
            except (ValueError, KeyError) as e:
                print(f"Warning: Skipping malformed row {row_num}: {e}")
                continue
    
    # Group genes by sequence_id for proper annotation structure
    sequence_groups = defaultdict(list)
    
    for gene_id, gene_data in genes_data.items():
        if not gene_data['exons']:
            continue
            
        # Sort exons by position
        gene_data['exons'].sort(key=lambda x: x['start'])
        
        # Create introns (gaps between exons)
        introns = []
        if len(gene_data['exons']) > 1:  # Multi-exon gene
            for i in range(len(gene_data['exons']) - 1):
                intron_start = gene_data['exons'][i]['end']
                intron_end = gene_data['exons'][i + 1]['start']
                if intron_end > intron_start:  # Valid intron
                    introns.append({
                        'start': intron_start,
                        'end': intron_end,
                        'donor_pos': intron_start,      # 5' splice site
                        'acceptor_pos': intron_end - 1  # 3' splice site
                    })
        
        # Create gene structure
        gene_structure = {
            'start': gene_data['gene_start'],
            'end': gene_data['gene_end'],
            'strand': gene_data['strand'],
            'gene_id': gene_id,
            'coding_start': gene_data['gene_start'],
            'coding_end': gene_data['gene_end'],
            'exons': gene_data['exons'],
            'introns': introns
        }
        
        sequence_id = gene_data['sequence_id']
        sequence_groups[sequence_id].append(gene_structure)
    
    # Create sequence objects with all genes grouped by sequence_id
    sequence_objects = []
    for sequence_id, genes in sequence_groups.items():
        sequence_obj = {'genes': genes, 'sequence_id': sequence_id}
        sequence_objects.append(sequence_obj)
    
    # Statistics
    all_genes = []
    for seq_obj in sequence_objects:
        all_genes.extend(seq_obj['genes'])
    
    total_genes = len(all_genes)
    single_exon = sum(1 for gene in all_genes if len(gene['exons']) == 1)
    multi_exon = total_genes - single_exon
    total_exons = sum(len(gene['exons']) for gene in all_genes)
    
    print(f"Loaded {total_genes} gene annotations from TSV:")
    print(f"  - {single_exon} single-exon genes")
    print(f"  - {multi_exon} multi-exon genes")
    print(f"  - {total_exons} total exons")
    
    if all_genes:
        sample = all_genes[0]
        print(f"Sample gene: {sample['gene_id']} ({sample['start']}-{sample['end']}, {len(sample['exons'])} exons)")
    
    print(f"Loaded {len(sequence_objects)} annotated sequences from TSV")
    
    return sequence_objects


def reverse_complement(sequence: str) -> str:
    """Generate reverse complement of DNA sequence."""
    complement_map = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', 'N': 'N'}
    complement = ''.join(complement_map.get(base.upper(), 'N') for base in sequence)
    return complement[::-1]


def validate_start_stop_codons(sequence: str, gene_start: int, gene_end: int, strand: str = '+') -> bool:
    """
    Validate that a gene has proper start and stop codons, considering strand.
    
    Args:
        sequence: DNA sequence string
        gene_start: Gene start position (0-based)
        gene_end: Gene end position (0-based, exclusive)
        strand: '+' for forward strand, '-' for reverse strand
        
    Returns:
        True if gene has valid ATG start and TAA/TAG/TGA stop codons
    """
    valid_start_codons = {'ATG'}
    valid_stop_codons = {'TAA', 'TAG', 'TGA'}
    
    if strand == '+':
        # Forward strand: check codons directly
        # Check start codon
        if gene_start + 2 >= len(sequence):
            return False
        start_codon = sequence[gene_start:gene_start+3].upper()
        if start_codon not in valid_start_codons:
            return False
        
        # Check stop codon
        if gene_end < 3:
            return False
        stop_codon = sequence[gene_end-3:gene_end].upper()
        if stop_codon not in valid_stop_codons:
            return False
            
    else:  # strand == '-'
        # Reverse strand: check reverse complement
        # For minus strand, the "start" is actually at gene_end and "stop" at gene_start
        # But we need to look at the reverse complement
        
        # Check start codon (at gene_end on reverse complement)
        if gene_end < 3:
            return False
        # The start codon is the last 3 bases of the gene, reverse complemented
        start_region = sequence[gene_end-3:gene_end].upper()
        start_codon = reverse_complement(start_region)
        if start_codon not in valid_start_codons:
            return False
        
        # Check stop codon (at gene_start on reverse complement)  
        if gene_start + 2 >= len(sequence):
            return False
        # The stop codon is the first 3 bases of the gene, reverse complemented
        stop_region = sequence[gene_start:gene_start+3].upper()
        stop_codon = reverse_complement(stop_region)
        if stop_codon not in valid_stop_codons:
            return False
        
    return True


def load_gene_contexts_with_annotations(gene_contexts_path: str, 
                                       annotations_path: str,
                                       filter_invalid_codons: bool = True) -> Tuple[List[str], List[Dict]]:
    """
    Load pre-extracted gene contexts and their corresponding annotations.
    
    This function expects:
    - gene_contexts_path: FASTA file with gene contexts (sequence ID = gene ID)
    - annotations_path: TSV file with gene annotations
    
    Returns:
        Tuple of (sequences, annotations) where each sequence corresponds to one gene
    """
    print(f"Loading gene contexts from {gene_contexts_path}")
    
    # Load gene context sequences
    gene_contexts = load_fasta_sequences_with_ids(gene_contexts_path)
    gene_context_dict = {seq_id: sequence for seq_id, sequence in gene_contexts}
    
    print(f"Loaded {len(gene_context_dict)} gene context sequences")
    
    # Load annotations
    annotation_objects = load_tsv_annotations(annotations_path)
    
    # Create mapping from gene_id to gene annotation
    gene_annotation_map = {}
    for seq_obj in annotation_objects:
        genes = seq_obj.get('genes', [])
        for gene in genes:
            gene_id = gene['gene_id']
            gene_annotation_map[gene_id] = gene
    
    print(f"Loaded annotations for {len(gene_annotation_map)} genes")
    
    # Match gene contexts with their annotations
    matched_sequences = []
    matched_annotations = []
    flank_size = 2000  # Should match the flanking size used in extract_gene_contexts.py
    filtered_count = 0
    
    for gene_id, sequence in gene_context_dict.items():
        if gene_id in gene_annotation_map:
            # Get original gene annotation
            original_gene = gene_annotation_map[gene_id]
            
            # Adjust coordinates to be relative to gene context sequence
            # Gene context starts at (gene_start - flank_size), so gene starts at flank_size
            gene_length = original_gene['end'] - original_gene['start']
            
            # Create adjusted gene annotation for the context sequence
            adjusted_gene = original_gene.copy()
            adjusted_gene['start'] = flank_size  # Gene starts at position flank_size in context
            adjusted_gene['end'] = flank_size + gene_length  # Gene ends at flank_size + gene_length
            
            # Filter out genes with invalid start/stop codons if requested
            if filter_invalid_codons:
                strand = adjusted_gene.get('strand', '+')
                if not validate_start_stop_codons(sequence, adjusted_gene['start'], adjusted_gene['end'], strand):
                    filtered_count += 1
                    continue  # Skip this gene
            
            matched_sequences.append(sequence)
            
            # Adjust exon coordinates too
            if 'exons' in adjusted_gene:
                adjusted_exons = []
                for exon in adjusted_gene['exons']:
                    adjusted_exon = exon.copy()
                    # Adjust exon coordinates relative to gene context
                    adjusted_exon['start'] = exon['start'] - original_gene['start'] + flank_size
                    adjusted_exon['end'] = exon['end'] - original_gene['start'] + flank_size
                    adjusted_exons.append(adjusted_exon)
                adjusted_gene['exons'] = adjusted_exons
            
            # Create sequence object format expected by DNADataset
            sequence_obj = {
                'sequence_id': gene_id,  # Use gene_id as sequence_id
                'genes': [adjusted_gene]
            }
            matched_annotations.append(sequence_obj)
        else:
            print(f"Warning: No annotation found for gene context {gene_id}")
    
    if filter_invalid_codons and filtered_count > 0:
        print(f"Filtered out {filtered_count} genes with invalid start/stop codons")
    
    print(f"Matched {len(matched_sequences)} gene contexts with annotations")
    
    return matched_sequences, matched_annotations


def map_sequences_to_annotations(sequences_with_ids: List[Tuple[str, str]], 
                                sequence_objects: List[Dict]) -> Tuple[List[str], List[Dict]]:
    """Map sequences to their corresponding gene annotations using sequence IDs."""
    
    # Create mapping from sequence_id to sequence object
    sequence_map = {}
    for seq_obj in sequence_objects:
        if 'sequence_id' in seq_obj:
            sequence_map[seq_obj['sequence_id']] = seq_obj
        else:
            # For legacy format compatibility
            for gene in seq_obj.get('genes', []):
                # This shouldn't happen with the new TSV format, but handle legacy
                pass
    
    # Map sequences to their gene annotations - ONLY INCLUDE ANNOTATED SEQUENCES
    mapped_sequences = []
    mapped_annotations = []
    
    skipped_count = 0
    for sequence_id, sequence in sequences_with_ids:
        # Find matching sequence object with genes
        if sequence_id in sequence_map:
            # Only include sequences that have annotations
            mapped_sequences.append(sequence)
            mapped_annotations.append(sequence_map[sequence_id])
        else:
            # Skip sequences without annotations to speed up training
            skipped_count += 1
    
    annotated_count = len(mapped_annotations)
    total_count = len(sequences_with_ids)
    speedup = total_count / annotated_count if annotated_count > 0 else 1
    
    print(f"Filtered to {annotated_count} annotated sequences (skipped {skipped_count} unannotated)")
    print(f"Training speedup: {speedup:.1f}x faster ({annotated_count}/{total_count} sequences)")
    
    return mapped_sequences, mapped_annotations


def create_sequence_windows(sequence: str, window_size: int = 1024, 
                           stride: int = 512) -> List[str]:
    """Create overlapping windows from a long sequence."""
    windows = []
    
    for i in range(0, len(sequence) - window_size + 1, stride):
        window = sequence[i:i + window_size]
        windows.append(window)
    
    # Add the last window if it's not covered
    if len(sequence) > window_size:
        last_window = sequence[-window_size:]
        windows.append(last_window)
    
    return windows


class DataAugmentation:
    """Data augmentation utilities for DNA sequences."""
    
    def __init__(self, reverse_complement_prob: float = 0.5, 
                 masking_prob: float = 0.1, max_mask_length: int = 50):
        self.reverse_complement_prob = reverse_complement_prob
        self.masking_prob = masking_prob
        self.max_mask_length = max_mask_length
        self.complement_map = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', 'N': 'N'}
    
    def reverse_complement(self, sequence: str) -> str:
        """Generate reverse complement of DNA sequence."""
        complement = ''.join(self.complement_map.get(base, 'N') for base in sequence.upper())
        return complement[::-1]
    
    def mask_random_regions(self, sequence: str) -> str:
        """Randomly mask regions of the sequence with N."""
        if np.random.random() > self.masking_prob:
            return sequence
            
        sequence_list = list(sequence)
        seq_length = len(sequence_list)
        
        # Random mask length and position
        mask_length = np.random.randint(1, min(self.max_mask_length, seq_length // 10))
        mask_start = np.random.randint(0, max(1, seq_length - mask_length))
        
        # Apply mask
        for i in range(mask_start, min(mask_start + mask_length, seq_length)):
            sequence_list[i] = 'N'
            
        return ''.join(sequence_list)
    
    def augment_sequence(self, sequence: str, annotation: Dict = None) -> Tuple[str, Dict]:
        """Apply data augmentation to sequence and adjust annotations accordingly."""
        augmented_sequence = sequence
        augmented_annotation = annotation.copy() if annotation else {}
        
        # Apply reverse complement augmentation
        if np.random.random() < self.reverse_complement_prob:
            augmented_sequence = self.reverse_complement(augmented_sequence)
            
            # Adjust annotations for reverse complement
            if annotation and 'genes' in annotation:
                seq_length = len(sequence)
                augmented_annotation = {'genes': []}
                
                for gene in annotation['genes']:
                    # Reverse coordinates
                    new_start = seq_length - gene.get('end', 0)
                    new_end = seq_length - gene.get('start', 0)
                    new_strand = '-' if gene.get('strand', '+') == '+' else '+'
                    
                    augmented_gene = {
                        'start': new_start,
                        'end': new_end,
                        'strand': new_strand,
                        'gene_id': gene.get('gene_id', ''),
                        'exons': [],
                        'introns': []
                    }
                    
                    # Reverse exon coordinates
                    if 'exons' in gene:
                        for exon in gene['exons']:
                            augmented_gene['exons'].append({
                                'start': seq_length - exon.get('end', 0),
                                'end': seq_length - exon.get('start', 0)
                            })
                        # Sort exons by new positions
                        augmented_gene['exons'].sort(key=lambda x: x['start'])
                    
                    # Reverse intron coordinates
                    if 'introns' in gene:
                        for intron in gene['introns']:
                            augmented_gene['introns'].append({
                                'start': seq_length - intron.get('end', 0),
                                'end': seq_length - intron.get('start', 0),
                                'donor_pos': seq_length - intron.get('acceptor_pos', 0),
                                'acceptor_pos': seq_length - intron.get('donor_pos', 0)
                            })
                        # Sort introns by new positions
                        augmented_gene['introns'].sort(key=lambda x: x['start'])
                    
                    augmented_annotation['genes'].append(augmented_gene)
        
        # Apply masking (after reverse complement to avoid coordinate issues)
        augmented_sequence = self.mask_random_regions(augmented_sequence)
        
        return augmented_sequence, augmented_annotation

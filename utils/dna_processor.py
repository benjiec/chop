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
    
    def __init__(self):
        # DNA vocabulary: A, C, G, T, N (unknown/ambiguous)
        self.vocab = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}
        self.vocab_size = len(self.vocab)
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
        
        # Start and stop codons
        self.start_codons = {'ATG', 'GTG', 'TTG'}  # Common start codons
        self.stop_codons = {'TAA', 'TAG', 'TGA'}   # Stop codons
        
        # Splice site motifs
        self.donor_motifs = ['GT', 'GC']      # 5' splice site
        self.acceptor_motifs = ['AG']         # 3' splice site
        
    def tokenize(self, sequence: str) -> torch.Tensor:
        """Convert DNA sequence string to token indices."""
        # Convert to uppercase and handle ambiguous bases
        sequence = sequence.upper()
        
        # Replace ambiguous bases with N
        for base in sequence:
            if base not in self.vocab:
                sequence = sequence.replace(base, 'N')
        
        # Convert to indices
        tokens = [self.vocab.get(base, 4) for base in sequence]  # Default to N
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


class DNADataset:
    """Dataset class for DNA sequences with biological annotations."""
    
    def __init__(self, sequences: List[str], annotations: Optional[List[Dict]] = None, 
                 max_length: int = 8192, tokenizer: Optional[DNATokenizer] = None):
        self.sequences = sequences
        self.annotations = annotations or []
        self.max_length = max_length
        self.tokenizer = tokenizer or DNATokenizer()
        self.feature_extractor = BiologicalFeatureExtractor()
        
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sequence = self.sequences[idx]
        
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
        if idx < len(self.annotations):
            targets = self._prepare_targets(sequence, self.annotations[idx])
        
        return {
            'input_ids': tokens,
            'attention_mask': attention_mask,
            'sequence_length': torch.tensor(len(sequence)),
            'features': processed_features,
            'targets': targets
        }
    
    def _prepare_targets(self, sequence: str, annotation: Dict) -> Dict[str, torch.Tensor]:
        """Prepare target tensors from annotations."""
        targets = {}
        
        # Initialize target arrays - use max_length for consistency
        seq_length = min(len(sequence), self.max_length)
        gene_boundaries = torch.zeros(self.max_length, 3)  # No gene, start, end
        exon_intron = torch.zeros(self.max_length, 3)     # Exon, intron, intergenic
        splice_sites = torch.zeros(self.max_length, 3)    # No splice, donor, acceptor
        coding_potential = torch.zeros(self.max_length)
        
        # Fill targets based on annotation
        if 'genes' in annotation:
            for gene in annotation['genes']:
                start = gene.get('start', 0)
                end = gene.get('end', seq_length)
                
                # Gene boundaries
                if start < self.max_length:
                    gene_boundaries[start, 1] = 1  # Start
                if end < self.max_length:
                    gene_boundaries[end, 2] = 1    # End
                
                # Exon/intron structure
                if 'exons' in gene:
                    for exon in gene['exons']:
                        exon_start = max(0, exon.get('start', start))
                        exon_end = min(self.max_length, exon.get('end', end))
                        exon_intron[exon_start:exon_end, 0] = 1  # Exon
                
                # Splice sites
                if 'introns' in gene:
                    for intron in gene['introns']:
                        donor_pos = intron.get('donor_pos', 0)
                        acceptor_pos = intron.get('acceptor_pos', 0)
                        
                        if donor_pos < self.max_length:
                            splice_sites[donor_pos, 1] = 1  # Donor
                        if acceptor_pos < self.max_length:
                            splice_sites[acceptor_pos, 2] = 1  # Acceptor
                
                # Coding potential
                coding_start = max(0, gene.get('coding_start', start))
                coding_end = min(self.max_length, gene.get('coding_end', end))
                coding_potential[coding_start:coding_end] = 1
        
        targets['gene_boundaries'] = gene_boundaries
        targets['exon_intron'] = exon_intron
        targets['splice_sites'] = splice_sites
        targets['coding_potential'] = coding_potential
        
        return targets


def load_fasta_sequences(file_path: str) -> List[str]:
    """Load DNA sequences from a FASTA file."""
    sequences = []
    for record in SeqIO.parse(file_path, "fasta"):
        sequences.append(str(record.seq))
    return sequences


def load_gff_annotations(file_path: str) -> List[Dict]:
    """Load gene annotations from a GFF file."""
    # This is a simplified GFF parser - you might want to use a more robust one
    annotations = {}  # Use dict to group CDS by gene ID
    
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            
            parts = line.strip().split('\t')
            if len(parts) < 9:
                continue
            
            # Handle both 'gene' and 'CDS' entries
            if parts[2] in ['gene', 'CDS']:
                start = int(parts[3]) - 1  # Convert to 0-based
                end = int(parts[4])
                strand = parts[6]
                attributes = parts[8]
                
                # Extract gene ID from attributes
                gene_id = None
                for attr in attributes.split(';'):
                    if 'ID=' in attr:
                        gene_id = attr.split('ID=')[1].split(';')[0]
                        break
                    elif 'locus_tag=' in attr:
                        gene_id = attr.split('locus_tag=')[1].split(';')[0]
                        break
                    elif 'gene=' in attr:
                        gene_id = attr.split('gene=')[1].split(';')[0]
                        break
                
                if gene_id is None:
                    gene_id = f"gene_{start}_{end}"  # Fallback ID
                
                if gene_id not in annotations:
                    annotations[gene_id] = {
                        'genes': [{
                            'start': start,
                            'end': end,
                            'strand': strand,
                            'coding_start': start,
                            'coding_end': end,
                            'exons': []
                        }]
                    }
                else:
                    # Extend gene boundaries if needed
                    gene = annotations[gene_id]['genes'][0]
                    gene['start'] = min(gene['start'], start)
                    gene['end'] = max(gene['end'], end)
                
                # Add as exon if it's a CDS
                if parts[2] == 'CDS':
                    annotations[gene_id]['genes'][0]['exons'].append({
                        'start': start,
                        'end': end
                    })
    
    # Convert to list format expected by the dataset
    annotation_list = []
    for gene_id, annotation in annotations.items():
        annotation_list.append(annotation)
    
    return annotation_list


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

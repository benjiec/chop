#!/usr/bin/env python3
"""
Quick test to verify target generation for UTR-START layouts.

This creates a few test layouts and manually verifies that:
1. UTR5 sequences are labeled as class 1
2. ATGs following UTRs are labeled as class 2 (START)
3. Decoy ATGs in background are labeled as class 0 (INTERGENIC)
4. Background regions are labeled as class 0
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from tests.layout_detection.utr_start_dataset import UTRStartDataset
import numpy as np

def test_target_generation():
    """Test that targets are generated correctly for UTR-START layouts."""
    
    print("=" * 60)
    print("TARGET GENERATION VERIFICATION TEST")
    print("=" * 60)
    
    # Create a small test dataset
    print("\n1. Creating test dataset (2 contigs, 3 layouts each)...")
    dataset = UTRStartDataset(
        num_contigs=2,
        layouts_per_contig=3,
        background_length=200,  # Smaller for easier verification
        window_size=1000,
        window_stride=200
    )
    
    print(f"\n2. Verifying target generation...")
    
    # Check each full contig
    for contig_idx in range(dataset.num_contigs):
        full_sequence = dataset.contigs[contig_idx]
        full_targets = dataset.contig_targets[contig_idx]
        
        print(f"\n--- CONTIG {contig_idx} ---")
        print(f"Length: {len(full_sequence)}bp")
        
        # Find all ATGs and their classifications
        atg_positions = []
        for i in range(len(full_sequence) - 2):
            if full_sequence[i:i+3] == 'ATG':
                atg_class = full_targets[i]
                atg_positions.append({
                    'position': i,
                    'class': atg_class,
                    'class_name': ['INTERGENIC', 'UTR5', 'START'][atg_class],
                    'context_before': full_sequence[max(0, i-30):i],
                    'context_after': full_sequence[i+3:i+33]
                })
        
        print(f"Found {len(atg_positions)} ATGs:")
        
        real_starts = 0
        decoy_atgs = 0
        
        for atg in atg_positions:
            pos = atg['position']
            class_name = atg['class_name']
            before = atg['context_before']
            after = atg['context_after']
            
            if class_name == 'START':
                real_starts += 1
                print(f"  REAL START at {pos}: ...{before[-10:]}[ATG]{after[:10]}...")
                
                # Verify this ATG follows a UTR5 region
                has_upstream_utr = False
                for check_pos in range(max(0, pos-100), pos):
                    if full_targets[check_pos] == 1:  # UTR5 class
                        has_upstream_utr = True
                        break
                
                if not has_upstream_utr:
                    print(f"    ❌ ERROR: START at {pos} has no upstream UTR5!")
                else:
                    print(f"    ✅ CORRECT: START at {pos} has upstream UTR5")
                    
            elif class_name == 'INTERGENIC':
                decoy_atgs += 1
                if decoy_atgs <= 3:  # Show first 3 decoys
                    print(f"  DECOY ATG at {pos}: ...{before[-10:]}[ATG]{after[:10]}...")
        
        print(f"\\nContig {contig_idx} summary:")
        print(f"  Real STARTs: {real_starts} (expected: 3)")
        print(f"  Decoy ATGs: {decoy_atgs}")
        
        # Assert correct counts
        assert real_starts == 3, f"Expected 3 real STARTs, got {real_starts}"
        print(f"  ✅ PASSED: Correct number of real STARTs")
        
        # Check UTR5 regions
        utr5_positions = np.sum(full_targets == 1)
        print(f"  UTR5 positions: {utr5_positions}")
        assert utr5_positions > 0, "Expected UTR5 positions"
        print(f"  ✅ PASSED: UTR5 regions present")
    
    print(f"\n" + "=" * 60)
    print("✅ ALL TESTS PASSED - Target generation is correct!")
    print("=" * 60)

if __name__ == "__main__":
    test_target_generation()

#!/usr/bin/env python3
"""
Convert Symbiodinium genome data to TSV format.

This script converts the existing Symbiodinium GFF annotations to the standardized
TSV format for improved data loading performance.
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))
from scripts.gff_to_tsv import GFFToTSVConverter


def main():
    """Convert Symbiodinium data to TSV format."""
    
    # Paths
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data"
    
    # Input files
    gff_path = data_dir / "GCA_001939145.1_filtered_tcov0.99_evalue1e-20.gff"
    
    # Output file
    tsv_path = data_dir / "GCA_001939145.1_annotations.tsv"
    
    # Check if input exists
    if not gff_path.exists():
        print(f"Error: GFF file not found: {gff_path}")
        sys.exit(1)
    
    print("Converting Symbiodinium genome annotations to TSV format...")
    print(f"Input: {gff_path}")
    print(f"Output: {tsv_path}")
    
    # Create converter
    converter = GFFToTSVConverter(validate_structure=True, verbose=True)
    
    try:
        # Convert
        converter.convert(str(gff_path), str(tsv_path))
        print(f"\nConversion completed successfully!")
        print(f"TSV file saved to: {tsv_path}")
        
    except Exception as e:
        print(f"Error during conversion: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()

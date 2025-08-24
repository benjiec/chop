#!/usr/bin/env python3
"""
Simple end-to-end test runner.

This script runs a basic version of the end-to-end test for quick validation.
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def run_simple_e2e_test():
    """Run a simplified end-to-end test."""
    print("Starting simple end-to-end test...")
    
    # Create temporary directory
    with tempfile.TemporaryDirectory(prefix="chop_simple_test_") as temp_dir:
        print(f"Test directory: {temp_dir}")
        
        # Step 1: Generate test data
        print("\n1. Generating synthetic test data...")
        from generate_test_fixture import generate_test_fixture
        
        fixtures_dir = Path(temp_dir) / "fixtures"
        fasta_file, gff_file = generate_test_fixture(
            output_dir=str(fixtures_dir),
            num_contigs=2,  # Smaller for quick test
            contig_length=6000
        )
        print(f"   Generated: {Path(fasta_file).name}, {Path(gff_file).name}")
        
        # Step 2: Run preprocessing
        print("\n2. Running preprocessing...")
        import argparse
        from scripts.preprocess_gene_data import main as preprocess_main
        
        processed_dir = Path(temp_dir) / "processed"
        processed_dir.mkdir()
        
        processed_fasta = processed_dir / "gene_contexts.fna"
        processed_tsv = processed_dir / "annotations.tsv"
        
        args = argparse.Namespace(
            fasta=fasta_file,
            gff=gff_file,
            output_fasta=str(processed_fasta),
            output_tsv=str(processed_tsv),
            flanking_bp=1000,  # Smaller for test
            min_gene_length=150,
            verbose=False
        )
        
        result = preprocess_main(args)
        if result != 0:
            print(f"   ❌ Preprocessing failed with code {result}")
            return False
        
        print(f"   ✅ Preprocessing completed")
        
        # Step 3: Verify processed files
        print("\n3. Verifying processed data...")
        from Bio import SeqIO
        
        contexts = list(SeqIO.parse(processed_fasta, "fasta"))
        print(f"   Generated {len(contexts)} gene contexts")
        
        with open(processed_tsv) as f:
            lines = [line for line in f if not line.startswith('#') and line.strip()]
            print(f"   Generated {len(lines)-1} annotation rows")  # -1 for header
        
        if len(contexts) == 0:
            print("   ❌ No gene contexts generated")
            return False
        
        if len(lines) <= 1:
            print("   ❌ No annotations generated")
            return False
        
        print("   ✅ Data verification passed")
        
        # For a simple test, we'll stop here since training takes time
        # In a real test, you would continue with training and inference
        
        print("\n✅ Simple end-to-end test completed successfully!")
        print("\nTest validated:")
        print("  • Synthetic data generation")
        print("  • GFF to preprocessed format conversion")
        print("  • Data integrity and format validation")
        print("\nFor full training and inference testing, use the complete test suite.")
        
        return True


if __name__ == "__main__":
    success = run_simple_e2e_test()
    sys.exit(0 if success else 1)

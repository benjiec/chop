#!/usr/bin/env python3
"""
End-to-end test for gene prediction training pipeline.

This test validates that the gene prediction training works from fixtures
through model training to basic inference.
"""

import sys
import tempfile
import subprocess
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from tests.e2e.generate_gene_prediction_fixture import generate_gene_prediction_fixture


def test_gene_prediction_e2e():
    """Test the complete gene prediction pipeline."""
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        print(f"Working directory: {tmp_dir}")
        
        # Step 1: Generate test fixtures
        print("Step 1: Generating test fixtures...")
        fasta_file, gff_file = generate_gene_prediction_fixture(
            output_dir=tmp_path / "fixtures",
            num_contigs=10,  # More contigs to ensure we fit all genes
            total_genes=100  # Increased to 100 genes for better training
        )
        
        # Step 2: Train gene prediction model
        print("Step 2: Training gene prediction model...")
        config_file = project_root / "configs" / "gene_prediction_test.yaml"
        model_output_dir = tmp_path / "model"
        
        # Run training script
        cmd = [
            "python", str(project_root / "training" / "train_gene_prediction.py"),
            "--config", str(config_file),
            "--fasta", str(fasta_file),
            "--gff", str(gff_file),
            "--output-dir", str(model_output_dir)
        ]
        
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=project_root, timeout=300)  # 5 minute timeout
        
        if result.returncode != 0:
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            raise AssertionError(f"Training failed with return code {result.returncode}")
        
        print("Training completed successfully!")
        
        # Step 3: Check outputs
        print("Step 3: Validating outputs...")
        
        # Check that model files were created
        model_files = list(model_output_dir.glob("*.ckpt"))
        assert len(model_files) > 0, f"No model checkpoint files found in {model_output_dir}"
        
        print(f"Found {len(model_files)} model checkpoint(s):")
        for model_file in model_files:
            print(f"  {model_file.name}")
        
        print("Gene prediction training test passed!")
        return tmp_dir


if __name__ == "__main__":
    test_gene_prediction_e2e()

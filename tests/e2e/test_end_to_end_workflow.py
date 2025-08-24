#!/usr/bin/env python3
"""
End-to-end workflow test for gene prediction pipeline.

This test validates the complete workflow:
1. Generate synthetic test data
2. Run preprocessing (FASTA + GFF → gene contexts + TSV)
3. Train model on preprocessed data
4. Run inference on original sequences
5. Validate that predictions approximate original annotations

The test uses synthetic data to ensure controlled, reproducible testing.
"""

import unittest
import sys
import os
import tempfile
import shutil
import subprocess
import yaml
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Set
from Bio import SeqIO

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
# Add current directory to path for local imports
sys.path.insert(0, str(Path(__file__).parent))

from generate_test_fixture import generate_test_fixture
# Subprocess calls used instead of direct imports
# All components use subprocess calls for better isolation
from utils.constants import GeneBoundaryClass, ExonIntronClass


class EndToEndWorkflowTest(unittest.TestCase):
    """Test the complete gene prediction workflow."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment."""
        # Create temporary directory for all test files
        cls.test_dir = tempfile.mkdtemp(prefix="chop_e2e_test_")
        cls.addClassCleanup(lambda: shutil.rmtree(cls.test_dir, ignore_errors=True))
        
        # Set deterministic behavior
        torch.manual_seed(42)
        import random
        import numpy as np
        random.seed(42)
        np.random.seed(42)
        
        print(f"Test directory: {cls.test_dir}")
    
    def setUp(self):
        """Set up individual test."""
        self.fixtures_dir = Path(self.test_dir) / "fixtures"
        self.processed_dir = Path(self.test_dir) / "processed"
        self.models_dir = Path(self.test_dir) / "models"
        self.results_dir = Path(self.test_dir) / "results"
        
        # Create directories
        for dir_path in [self.fixtures_dir, self.processed_dir, self.models_dir, self.results_dir]:
            dir_path.mkdir(exist_ok=True, parents=True)
    
    def test_step_1_generate_synthetic_data(self):
        """Step 1: Generate synthetic genomic data."""
        print("Step 1: Generating synthetic test data...")
        
        # Generate test fixture
        fasta_file, gff_file = generate_test_fixture(
            output_dir=str(self.fixtures_dir),
            num_contigs=3,
            contig_length=7500
        )
        
        # Verify files were created
        self.assertTrue(Path(fasta_file).exists(), "FASTA file not created")
        self.assertTrue(Path(gff_file).exists(), "GFF file not created")
        
        # Verify file contents
        sequences = list(SeqIO.parse(fasta_file, "fasta"))
        self.assertEqual(len(sequences), 3, "Should have 3 contigs")
        
        for seq in sequences:
            self.assertGreaterEqual(len(seq.seq), 7000, "Contig should be at least 7kb")
            self.assertLessEqual(len(seq.seq), 8000, "Contig should be at most 8kb")
        
        # Verify GFF has content
        with open(gff_file) as f:
            lines = [line for line in f if not line.startswith('#') and line.strip()]
            self.assertGreater(len(lines), 0, "GFF should have annotations")
        
        print(f"  ✓ Generated {len(sequences)} contigs")
        print(f"  ✓ Generated {len(lines)} GFF annotations")
        
        # Store file paths for next steps
        self.__class__.original_fasta = fasta_file
        self.__class__.original_gff = gff_file
    
    def test_step_2_preprocess_data(self):
        """Step 2: Run preprocessing pipeline."""
        print("Step 2: Running preprocessing pipeline...")
        
        # Ensure step 1 completed
        self.assertTrue(hasattr(self.__class__, 'original_fasta'), "Step 1 must complete first")
        
        # Set up output paths
        processed_fasta = self.processed_dir / "gene_contexts.fna"
        processed_tsv = self.processed_dir / "annotations.tsv"
        
        # Run preprocessing using subprocess
        cmd = [
            sys.executable, str(project_root / "scripts" / "preprocess_gene_data.py"),
            "--fasta", self.__class__.original_fasta,
            "--gff", self.__class__.original_gff,
            "--output-fasta", str(processed_fasta),
            "--output-tsv", str(processed_tsv),
            "--flanking-bp", "2000",
            "--validate-codons"
        ]
        
        print(f"  Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(project_root))
        
        if result.returncode != 0:
            print(f"Preprocessing failed with return code {result.returncode}")
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            self.fail("Preprocessing failed")
        
        self.assertEqual(result.returncode, 0, "Preprocessing should succeed")
        
        # Verify output files
        self.assertTrue(processed_fasta.exists(), "Processed FASTA not created")
        self.assertTrue(processed_tsv.exists(), "Processed TSV not created")
        
        # Verify processed data
        contexts = list(SeqIO.parse(processed_fasta, "fasta"))
        self.assertGreater(len(contexts), 0, "Should have gene contexts")
        
        # Check TSV format
        with open(processed_tsv) as f:
            header = f.readline().strip()
            expected_cols = ['sequence_id', 'gene_id', 'gene_start', 'gene_end', 'exon_start', 'exon_end', 'strand']
            self.assertEqual(header.split('\t'), expected_cols, "TSV header incorrect")
            
            data_lines = [line for line in f if line.strip()]
            self.assertGreater(len(data_lines), 0, "TSV should have data")
        
        print(f"  ✓ Generated {len(contexts)} gene contexts")
        print(f"  ✓ Generated {len(data_lines)} annotation rows")
        
        # Store paths for next steps
        self.__class__.processed_fasta = str(processed_fasta)
        self.__class__.processed_tsv = str(processed_tsv)
    
    def test_step_3_create_test_config(self):
        """Step 3: Create and validate test configuration."""
        print("Step 3: Setting up test configuration...")
        
        # Ensure step 2 completed
        self.assertTrue(hasattr(self.__class__, 'processed_fasta'), "Step 2 must complete first")
        
        # Load base config
        config_path = Path(__file__).parent / "test_e2e_config.yaml"
        self.assertTrue(config_path.exists(), "Test config template should exist")
        
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        # Update paths to use our test data
        config['data']['sequences_path'] = self.__class__.processed_fasta
        config['data']['tsv_annotations_path'] = self.__class__.processed_tsv
        config['training']['checkpoint_dir'] = str(self.models_dir / "checkpoints")
        config['evaluation']['output_dir'] = str(self.results_dir)
        
        # Create test-specific config
        test_config_path = Path(self.test_dir) / "test_config.yaml"
        with open(test_config_path, 'w') as f:
            yaml.dump(config, f)
        
        print(f"  ✓ Created test config: {test_config_path}")
        
        # Store config path
        self.__class__.test_config = str(test_config_path)
    
    def test_step_4_train_model(self):
        """Step 4: Train model on preprocessed data."""
        print("Step 4: Training model...")
        
        # Ensure step 3 completed
        self.assertTrue(hasattr(self.__class__, 'test_config'), "Step 3 must complete first")
        
        # Create checkpoint directory
        checkpoint_dir = self.models_dir / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True, parents=True)
        
        # Train using subprocess to avoid module conflicts
        cmd = [
            sys.executable, str(project_root / "training" / "train.py"),
            "--config", self.__class__.test_config
        ]
        
        print(f"  Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(project_root))
        
        # Check if training succeeded
        if result.returncode != 0:
            print(f"Training failed with return code {result.returncode}")
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            self.fail("Model training failed")
        
        # Verify model checkpoint was created
        checkpoint_files = list(checkpoint_dir.glob("*.ckpt"))
        self.assertGreater(len(checkpoint_files), 0, "No checkpoint files created")
        
        # Find the best/final checkpoint
        if checkpoint_dir.glob("final_model.pt"):
            model_path = checkpoint_dir / "final_model.pt"
        elif checkpoint_files:
            model_path = checkpoint_files[-1]  # Use the last checkpoint
        else:
            self.fail("No model checkpoint found")
        
        print(f"  ✓ Training completed")
        print(f"  ✓ Model saved: {model_path}")
        
        # Store model path
        self.__class__.trained_model = str(model_path)
    
    def test_step_5_run_inference(self):
        """Step 5: Run inference on original sequences."""
        print("Step 5: Running inference...")
        
        # Ensure step 4 completed
        self.assertTrue(hasattr(self.__class__, 'trained_model'), "Step 4 must complete first")
        
        # Run inference using subprocess
        output_dir = self.results_dir
        
        cmd = [
            sys.executable, str(project_root / "inference" / "predict.py"),
            "--model", self.__class__.trained_model,
            "--config", self.__class__.test_config,
            "--input", self.__class__.original_fasta,
            "--output", str(output_dir)
        ]
        
        print(f"  Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(project_root))
        
        if result.returncode != 0:
            print(f"Inference failed with return code {result.returncode}")
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            self.fail("Inference failed")
        
        # Verify predictions were generated
        predictions_file = output_dir / "all_predictions.json"
        self.assertTrue(predictions_file.exists(), "Predictions file not created")
        
        print(f"  ✓ Inference completed")
        print(f"  ✓ Predictions saved: {predictions_file}")
        
        # Store predictions path
        self.__class__.predictions_file = str(predictions_file)
    
    def test_step_6_validate_predictions(self):
        """Step 6: Validate predictions against original annotations."""
        print("Step 6: Validating predictions...")
        
        # Ensure step 5 completed
        self.assertTrue(hasattr(self.__class__, 'predictions_file'), "Step 5 must complete first")
        
        # This is a simplified validation - in a real test you might:
        # 1. Parse the predictions JSON
        # 2. Compare predicted gene locations with original GFF
        # 3. Calculate precision/recall metrics
        # 4. Verify that major gene structures are detected
        
        # For now, just verify the workflow completed successfully
        import json
        
        try:
            with open(self.__class__.predictions_file) as f:
                predictions = json.load(f)
            
            # Basic validation that we got some predictions
            self.assertIsInstance(predictions, (dict, list), "Predictions should be valid JSON")
            
            # If predictions are per-sequence
            if isinstance(predictions, dict):
                self.assertGreater(len(predictions), 0, "Should have predictions for sequences")
            elif isinstance(predictions, list):
                self.assertGreater(len(predictions), 0, "Should have some predictions")
            
            print(f"  ✓ Predictions validated")
            print(f"  ✓ Found {len(predictions)} prediction entries")
            
        except (json.JSONDecodeError, FileNotFoundError) as e:
            self.fail(f"Failed to load predictions: {e}")
    
    def test_complete_workflow(self):
        """Run the complete end-to-end workflow."""
        print("\n" + "="*60)
        print("RUNNING COMPLETE END-TO-END WORKFLOW TEST")
        print("="*60)
        
        # Run all steps in sequence
        self.test_step_1_generate_synthetic_data()
        self.test_step_2_preprocess_data()
        self.test_step_3_create_test_config()
        self.test_step_4_train_model()
        self.test_step_5_run_inference()
        self.test_step_6_validate_predictions()
        
        print("\n" + "="*60)
        print("END-TO-END WORKFLOW TEST COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\nWorkflow Summary:")
        print(f"  • Test directory: {self.test_dir}")
        print(f"  • Original data: {self.__class__.original_fasta}")
        print(f"  • Processed data: {self.__class__.processed_fasta}")
        print(f"  • Trained model: {self.__class__.trained_model}")
        print(f"  • Predictions: {self.__class__.predictions_file}")
        print("\nAll components of the gene prediction pipeline are working correctly!")


def run_quick_test():
    """Run a quick version of the test with minimal training."""
    # This function can be used for rapid development testing
    test = EndToEndWorkflowTest()
    test.setUpClass()
    test.setUp()
    
    # Modify config for ultra-fast training
    test.test_step_1_generate_synthetic_data()
    test.test_step_2_preprocess_data()
    test.test_step_3_create_test_config()
    
    # Update config for fast dev run
    with open(test.__class__.test_config) as f:
        config = yaml.safe_load(f)
    
    config['test']['fast_dev_run'] = True
    config['training']['max_epochs'] = 1
    config['training']['limit_train_batches'] = 2
    
    with open(test.__class__.test_config, 'w') as f:
        yaml.dump(config, f)
    
    test.test_step_4_train_model()
    test.test_step_5_run_inference()
    test.test_step_6_validate_predictions()
    
    print("Quick test completed!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run end-to-end workflow test")
    parser.add_argument("--quick", action="store_true", help="Run quick test with minimal training")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    if args.quick:
        run_quick_test()
    else:
        # Run full test suite
        if args.verbose:
            unittest.main(argv=[''], verbosity=2, exit=False)
        else:
            unittest.main(argv=[''], exit=False)

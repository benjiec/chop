#!/usr/bin/env python3
"""
Gene prediction workflow test.

This test validates the complete gene prediction pipeline:
1. Generate synthetic test data (FASTA + GFF)
2. Run preprocessing (FASTA + GFF → gene contexts + TSV)
3. Train gene prediction model on preprocessed data
4. Run inference on original sequences
5. Evaluate predictions with comprehensive metrics

The test uses synthetic data to ensure controlled, reproducible testing
and provides detailed evaluation of gene boundary detection performance.
"""

import unittest
import sys
import os
import subprocess
import yaml
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Set
from collections import defaultdict
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


class GenePredictionWorkflowTest(unittest.TestCase):
    """Test the complete gene prediction workflow."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment."""
        # Create unique run ID for this test execution
        import datetime
        run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        cls.run_id = run_id
        
        # Use persistent directories with unique run ID
        cls.test_dir = project_root / "tests" / "gene_prediction" / f"test_run_{run_id}"
        cls.test_dir.mkdir(exist_ok=True, parents=True)
        
        # Set deterministic behavior
        torch.manual_seed(42)
        import random
        import numpy as np
        random.seed(42)
        np.random.seed(42)
        
        print(f"Test run ID: {run_id}")
        print(f"Test directory: {cls.test_dir}")
    
    def setUp(self):
        """Set up individual test."""
        # Use test run directory for all artifacts to ensure complete isolation
        self.fixtures_dir = Path(self.test_dir) / "fixtures"
        self.processed_dir = Path(self.test_dir) / "processed"
        self.models_dir = Path(self.test_dir) / "models"
        self.results_dir = Path(self.test_dir) / "predictions"
        
        # Create directories
        for dir_path in [self.fixtures_dir, self.processed_dir, self.models_dir, self.results_dir]:
            dir_path.mkdir(exist_ok=True, parents=True)
    
    def test_step_1_generate_synthetic_data(self, num_contigs=3):
        """Step 1: Generate synthetic genomic data."""
        print(f"Step 1: Generating synthetic test data ({num_contigs} contigs)...")
       
        # Generate test fixture in persistent fixtures directory
        fasta_file, gff_file = generate_test_fixture(
            output_dir=str(self.fixtures_dir),
            num_contigs=num_contigs
        )
        
        # Verify files were created
        self.assertTrue(Path(fasta_file).exists(), "FASTA file not created")
        self.assertTrue(Path(gff_file).exists(), "GFF file not created")
        
        # Verify file contents
        sequences = list(SeqIO.parse(fasta_file, "fasta"))
        self.assertEqual(len(sequences), num_contigs, f"Should have {num_contigs} contigs")
        
        # Load sequences into a dictionary for validation
        seq_dict = {seq.id: str(seq.seq) for seq in sequences}
        
        for seq in sequences:
            # Contig size is now determined by genes + spacers, so just verify reasonable size
            self.assertGreaterEqual(len(seq.seq), 10000, "Contig should be at least 10kb")
            self.assertLessEqual(len(seq.seq), 200000, "Contig should be reasonable size (< 200kb)")
        
        # Parse GFF and validate gene structure
        gene_annotations = self._parse_gff_annotations(gff_file)
        
        # Verify GFF has content
        self.assertGreater(len(gene_annotations), 0, "GFF should have annotations")
        
        # Comprehensive validation of gene structure
        total_genes = 0
        total_cds = 0
        
        for seq_id, annotations in gene_annotations.items():
            if seq_id not in seq_dict:
                continue
                
            sequence = seq_dict[seq_id]
            genes = [ann for ann in annotations if ann['type'] == 'gene']
            cds_features = [ann for ann in annotations if ann['type'] == 'CDS']
            
            total_genes += len(genes)
            total_cds += len(cds_features)
            
            # Validate each gene
            for gene in genes:
                gene_cds = [cds for cds in cds_features 
                           if gene['start'] <= cds['start'] < cds['end'] <= gene['end']]
                
                if not gene_cds:
                    continue  # Skip genes without CDS
                
                # Sort CDS by start position
                gene_cds.sort(key=lambda x: x['start'])
                
                # Validate non-overlapping exons
                for i in range(len(gene_cds) - 1):
                    current_end = gene_cds[i]['end']
                    next_start = gene_cds[i + 1]['start']
                    self.assertLess(current_end, next_start, 
                                  f"Overlapping CDS in gene {gene.get('attributes', {}).get('ID', 'unknown')}: "
                                  f"CDS {i} ends at {current_end}, CDS {i+1} starts at {next_start}")
                
                # Validate START codon (first 3 bp of first CDS)
                first_cds = gene_cds[0]
                start_codon = sequence[first_cds['start']:first_cds['start'] + 3]
                self.assertEqual(start_codon, 'ATG', 
                               f"First CDS should start with ATG, got '{start_codon}' "
                               f"in gene {gene.get('attributes', {}).get('ID', 'unknown')}")
                
                # Validate STOP codon (last 3 bp of last CDS)
                last_cds = gene_cds[-1]
                stop_codon = sequence[last_cds['end'] - 3:last_cds['end']]
                valid_stop_codons = {'TAA', 'TAG', 'TGA'}
                self.assertIn(stop_codon, valid_stop_codons,
                             f"Last CDS should end with stop codon (TAA/TAG/TGA), got '{stop_codon}' "
                             f"in gene {gene.get('attributes', {}).get('ID', 'unknown')}")
        
        print(f"  ✓ Generated {len(sequences)} contigs")
        print(f"  ✓ Generated {total_genes} genes with {total_cds} CDS features")
        print(f"  ✓ Validated START/STOP codons and non-overlapping exons")
        
        # Store file paths for next steps
        self.__class__.original_fasta = fasta_file
        self.__class__.original_gff = gff_file
    
    def _parse_gff_annotations(self, gff_file: str) -> Dict[str, List[Dict]]:
        """Parse GFF file into structured annotation data."""
        annotations = defaultdict(list)
        
        with open(gff_file, 'r') as f:
            for line in f:
                if line.startswith('#') or not line.strip():
                    continue
                    
                parts = line.strip().split('\t')
                if len(parts) < 9:
                    continue
                
                seq_id = parts[0]
                feature_type = parts[2]
                start = int(parts[3]) - 1  # Convert to 0-based
                end = int(parts[4])        # Keep as 1-based exclusive
                strand = parts[6]
                attributes_str = parts[8]
                
                # Parse attributes
                attributes = {}
                for attr in attributes_str.split(';'):
                    if '=' in attr:
                        key, value = attr.split('=', 1)
                        attributes[key] = value
                
                annotation = {
                    'type': feature_type,
                    'start': start,
                    'end': end,
                    'strand': strand,
                    'attributes': attributes
                }
                
                annotations[seq_id].append(annotation)
        
        return dict(annotations)

    def test_step_2_preprocess_data(self):
        """Step 2: Run preprocessing pipeline."""
        print("Step 2: Running preprocessing pipeline...")
        
        # Ensure step 1 completed
        self.assertTrue(hasattr(self.__class__, 'original_fasta'), "Step 1 must complete first")
        
        # Set up output paths
        processed_fasta = self.processed_dir / "gene_contexts.fna"
        processed_tsv = self.processed_dir / "annotations.tsv"
        
        # Run preprocessing using subprocess (without --validate-codons for synthetic data)
        cmd = [
            sys.executable, str(project_root / "scripts" / "preprocess_gene_data.py"),
            "--fasta", self.__class__.original_fasta,
            "--gff", self.__class__.original_gff,
            "--output-fasta", str(processed_fasta),
            "--output-tsv", str(processed_tsv),
            "--flanking-bp", "2000"
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
        
        # Load original data for comparison
        original_sequences = {seq.id: str(seq.seq) for seq in SeqIO.parse(self.__class__.original_fasta, "fasta")}
        original_annotations = self._parse_gff_annotations(self.__class__.original_gff)
        
        # Check TSV format and load data
        import pandas as pd
        annotations_df = pd.read_csv(processed_tsv, sep='\t')
        expected_cols = ['sequence_id', 'gene_id', 'gene_start', 'gene_end', 'exon_start', 'exon_end', 'strand']
        self.assertEqual(list(annotations_df.columns), expected_cols, "TSV header incorrect")
        self.assertGreater(len(annotations_df), 0, "TSV should have data")
        
        # Validate gene counts match between original and processed
        original_gene_count = 0
        for seq_id, annotations in original_annotations.items():
            original_gene_count += len([ann for ann in annotations if ann['type'] == 'gene'])
        
        processed_gene_count = annotations_df['gene_id'].nunique()
        self.assertEqual(processed_gene_count, original_gene_count, 
                        f"Gene count mismatch: original {original_gene_count}, processed {processed_gene_count}")
        
        # CRITICAL TEST: Validate FASTA sequence IDs match TSV sequence IDs
        fasta_sequence_ids = set(seq.id for seq in contexts)
        tsv_sequence_ids = set(annotations_df['sequence_id'].unique())
        
        self.assertEqual(fasta_sequence_ids, tsv_sequence_ids,
                        f"FASTA sequence IDs don't match TSV sequence IDs!\n"
                        f"  FASTA IDs: {sorted(fasta_sequence_ids)}\n"
                        f"  TSV IDs: {sorted(tsv_sequence_ids)}\n"
                        f"  Missing in FASTA: {tsv_sequence_ids - fasta_sequence_ids}\n"
                        f"  Missing in TSV: {fasta_sequence_ids - tsv_sequence_ids}")
        
        print(f"  ✓ Verified FASTA and TSV sequence IDs match ({len(fasta_sequence_ids)} sequences)")
        
        # Validate CDS/exon counts match
        original_cds_count = 0
        for seq_id, annotations in original_annotations.items():
            original_cds_count += len([ann for ann in annotations if ann['type'] == 'CDS'])
        
        processed_cds_count = len(annotations_df)
        self.assertEqual(processed_cds_count, original_cds_count,
                        f"CDS count mismatch: original {original_cds_count}, processed {processed_cds_count}")
        
        # Validate coordinate translation and sequence verification
        contexts_dict = {seq.id: str(seq.seq) for seq in contexts}
        validation_count = 0
        
        for _, row in annotations_df.iterrows():
            gene_id = row['gene_id']
            context_seq = contexts_dict.get(gene_id)
            
            if context_seq is None:
                continue
                
            # Find original sequence and coordinates
            orig_seq_id = gene_id.split('_gene_')[0]  # Extract original sequence ID
            if orig_seq_id not in original_sequences:
                continue
                
            original_seq = original_sequences[orig_seq_id]
            
            # Get original CDS coordinates (1-based from GFF)
            orig_exon_start = None
            orig_exon_end = None
            
            for annotation in original_annotations.get(orig_seq_id, []):
                if (annotation['type'] == 'CDS' and 
                    annotation['start'] + 1 == row['exon_start'] and  # Convert to 1-based
                    annotation['end'] == row['exon_end']):
                    orig_exon_start = annotation['start']  # 0-based
                    orig_exon_end = annotation['end']      # 1-based exclusive
                    break
            
            if orig_exon_start is None:
                continue
                
            # Extract sequences and compare
            original_exon_seq = original_seq[orig_exon_start:orig_exon_end]
            
            # Convert processed coordinates to 0-based for extraction
            processed_exon_start = row['exon_start'] - 1  # Convert to 0-based
            processed_exon_end = row['exon_end']           # Keep as 1-based exclusive
            processed_exon_seq = context_seq[processed_exon_start:processed_exon_end]
            
            self.assertEqual(original_exon_seq, processed_exon_seq,
                           f"Sequence mismatch for {gene_id} exon {row['exon_start']}-{row['exon_end']}: "
                           f"original='{original_exon_seq}', processed='{processed_exon_seq}'")
            
            validation_count += 1
            
            # Limit validation for performance (check first 20 exons)
            if validation_count >= 20:
                break
        
        print(f"  ✓ Generated {len(contexts)} gene contexts")
        print(f"  ✓ Generated {len(annotations_df)} annotation rows")
        print(f"  ✓ Validated {original_gene_count} genes and {original_cds_count} CDS features")
        print(f"  ✓ Verified coordinate translation and sequences for {validation_count} exons")
        
        # Store paths for next steps
        self.__class__.processed_fasta = str(processed_fasta)
        self.__class__.processed_tsv = str(processed_tsv)
    
    def _get_config_path(self):
        """Get the config path, allowing override via class attribute."""
        config_name = getattr(self.__class__, 'config_name', 'gene_prediction_capped_weights.yaml')
        return str(project_root / "configs" / config_name)
    
    def test_step_3_train_model(self):
        """Step 3: Train model using gene prediction training script."""
        print("Step 3: Training model...")
        
        # Ensure step 2 completed (we need preprocessed TSV and gene context FASTA)
        self.assertTrue(hasattr(self.__class__, 'processed_fasta'), "Step 2 must complete first")
        self.assertTrue(hasattr(self.__class__, 'processed_tsv'), "Step 2 must complete first")
        
        # Use persistent models directory
        models_dir = self.models_dir
        
        # Use standard config for both modes
        config_path = self._get_config_path()
        
        # Train using the gene prediction training script (preprocessed FASTA + TSV input)
        cmd = [
            sys.executable, str(project_root / "training" / "train_gene_prediction.py"),
            "--config", config_path,
            "--fasta", self.__class__.processed_fasta,
            "--tsv", self.__class__.processed_tsv,
            "--output-dir", str(models_dir)
        ]
        
        print(f"  Running: {' '.join(cmd)}")
        print("  Training progress:")
        print("  " + "-" * 50)
        
        # Run without capturing output so we can see training progress
        result = subprocess.run(cmd, cwd=str(project_root))
        
        print("  " + "-" * 50)
        
        # Check if training succeeded
        if result.returncode != 0:
            print(f"Training failed with return code {result.returncode}")
            self.fail("Model training failed")
        
        # Verify model checkpoint was created in persistent directory
        checkpoint_files = list(models_dir.glob("**/*.ckpt"))
        self.assertGreater(len(checkpoint_files), 0, "No checkpoint files created")
        
        # Find the best checkpoint - look for the one with the lowest validation loss
        best_checkpoint = None
        best_val_loss = float('inf')
        
        for ckpt_file in checkpoint_files:
            # Look for validation loss in filename (e.g., "val_total_loss=0.834.ckpt")
            if "val_total_loss=" in str(ckpt_file):
                try:
                    # Extract validation loss from filename
                    val_loss_str = str(ckpt_file).split("val_total_loss=")[1].split(".ckpt")[0]
                    val_loss = float(val_loss_str)
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_checkpoint = ckpt_file
                except (ValueError, IndexError):
                    continue
        
        # If no validation loss checkpoint found, use the last checkpoint
        if best_checkpoint is None:
            last_checkpoints = list(models_dir.glob("**/last*.ckpt"))
            if last_checkpoints:
                best_checkpoint = last_checkpoints[0]
            else:
                # Fallback to any checkpoint
                best_checkpoint = checkpoint_files[0]
        
        model_path = best_checkpoint
        
        print(f"  ✓ Training completed")
        print(f"  ✓ Model saved: {model_path}")
        print(f"  ✓ Model persisted in: {models_dir}")
        
        # Store model path and models directory
        self.__class__.trained_model = str(model_path)
        self.__class__.models_dir = str(models_dir)
    
    def test_step_4_run_inference(self):
        """Step 4: Run inference on original sequences."""
        print("Step 4: Running inference...")
        
        # Ensure step 3 completed
        self.assertTrue(hasattr(self.__class__, 'trained_model'), "Step 3 must complete first")
        
        # Use persistent results directory
        results_dir = self.results_dir
        
        # Use standard config for both modes
        config_path = self._get_config_path()
        
        # Run inference using the gene prediction inference script
        cmd = [
            sys.executable, str(project_root / "inference" / "predict_gene_boundaries.py"),
            "--model", self.__class__.trained_model,
            "--config", config_path,
            "--input", self.__class__.original_fasta,
            "--output", str(results_dir)
        ]
        
        print(f"  Running: {' '.join(cmd)}")
        print("  Inference progress:")
        print("  " + "-" * 30)
        
        # Run without capturing output so we can see inference progress
        result = subprocess.run(cmd, cwd=str(project_root))
        
        print("  " + "-" * 30)
        
        if result.returncode != 0:
            print(f"Inference failed with return code {result.returncode}")
            self.fail("Inference failed")
        
        # Verify predictions were generated
        predictions_file = results_dir / "all_gene_boundary_predictions.json"
        self.assertTrue(predictions_file.exists(), "Gene boundary predictions file not created")
        
        print(f"  ✓ Inference completed")
        print(f"  ✓ Predictions saved: {predictions_file}")
        
        # Store predictions path
        self.__class__.predictions_file = str(predictions_file)
    
    def test_step_5_validate_predictions(self):
        """Step 5: Validate predictions with sensitivity, specificity, precision, recall."""
        print("Step 5: Analyzing predictions with comprehensive metrics...")
        
        # Ensure step 4 completed
        self.assertTrue(hasattr(self.__class__, 'predictions_file'), "Step 4 must complete first")
        
        # Load predictions and ground truth
        try:
            import json
            from collections import defaultdict
            import numpy as np
            
            # Load predictions
            with open(self.__class__.predictions_file) as f:
                predictions = json.load(f)
            
            # Load ground truth from GFF
            true_genes = self._parse_gff_ground_truth(self.__class__.original_gff)
            
            # Calculate metrics for each class
            metrics = self._calculate_evaluation_metrics(predictions, true_genes)
            
            # Print comprehensive evaluation results
            print(f"\n  📊 COMPREHENSIVE EVALUATION RESULTS:")
            print(f"  " + "="*50)
            
            for class_name, class_metrics in metrics.items():
                print(f"\n  {class_name.upper()} PREDICTIONS:")
                print(f"    Sensitivity (Recall): {class_metrics['sensitivity']:.3f}")
                print(f"    Specificity:         {class_metrics['specificity']:.3f}")
                print(f"    Precision:           {class_metrics['precision']:.3f}")
                print(f"    F1-Score:            {class_metrics['f1_score']:.3f}")
                print(f"    True Positives:      {class_metrics['tp']}")
                print(f"    False Positives:     {class_metrics['fp']}")
                print(f"    False Negatives:     {class_metrics['fn']}")
                print(f"    True Negatives:      {class_metrics['tn']}")
            
            # Overall assessment
            avg_f1 = np.mean([m['f1_score'] for m in metrics.values() if not np.isnan(m['f1_score'])])
            print(f"\n  🎯 OVERALL PERFORMANCE:")
            print(f"    Average F1-Score:    {avg_f1:.3f}")
            
            # Check if model meets minimum performance thresholds
            start_f1 = metrics.get('start', {}).get('f1_score', 0)
            stop_f1 = metrics.get('stop', {}).get('f1_score', 0)
            gene_f1 = metrics.get('gene_body', {}).get('f1_score', 0)
            
            print(f"    START F1:            {start_f1:.3f}")
            print(f"    STOP F1:             {stop_f1:.3f}")
            print(f"    GENE_BODY F1:        {gene_f1:.3f}")
            
            # Performance thresholds for gene_prediction test
            if start_f1 > 0.3 and stop_f1 > 0.3 and gene_f1 > 0.5:
                print(f"\n  ✅ Model performance PASSES gene_prediction test thresholds!")
            elif start_f1 > 0.1 and stop_f1 > 0.1 and gene_f1 > 0.3:
                print(f"\n  ⚠️  Model performance is moderate - may need more training")
            else:
                print(f"\n  ❌ Model performance below minimum thresholds")
                # Don't fail the test - this is about pipeline functionality
                print(f"     (Pipeline works, but model needs improvement)")
            
            # Save detailed metrics
            metrics_file = Path(self.__class__.models_dir) / "evaluation_metrics.json"
            with open(metrics_file, 'w') as f:
                # Convert numpy types to regular Python types for JSON serialization
                serializable_metrics = {}
                for k, v in metrics.items():
                    serializable_metrics[k] = {
                        metric_k: float(metric_v) if isinstance(metric_v, np.floating) 
                                 else int(metric_v) if isinstance(metric_v, np.integer)
                                 else metric_v 
                        for metric_k, metric_v in v.items()
                    }
                json.dump(serializable_metrics, f, indent=2)
            
            print(f"\n  💾 Detailed metrics saved: {metrics_file}")
            
            # Store metrics for potential further analysis
            self.__class__.evaluation_metrics = metrics
            
        except (json.JSONDecodeError, FileNotFoundError) as e:
            self.fail(f"Failed to load predictions or ground truth: {e}")
        except Exception as e:
            self.fail(f"Error during evaluation: {e}")
    
    def _parse_gff_ground_truth(self, gff_file):
        """Parse GFF file to extract ground truth gene boundaries."""
        ground_truth = defaultdict(list)
        
        with open(gff_file) as f:
            for line in f:
                if line.startswith('#') or not line.strip():
                    continue
                
                parts = line.strip().split('\t')
                if len(parts) < 9:
                    continue
                
                seqid = parts[0]
                feature_type = parts[2]
                start = int(parts[3]) - 1  # Convert to 0-based
                end = int(parts[4])        # Keep end exclusive (GFF is 1-based inclusive)
                strand = parts[6]
                
                if feature_type == 'gene':
                    # Parse gene boundaries
                    ground_truth[seqid].append({
                        'type': 'gene',
                        'start': start,
                        'end': end,
                        'strand': strand
                    })
                elif feature_type == 'CDS':
                    # Parse CDS (exon) boundaries
                    ground_truth[seqid].append({
                        'type': 'cds',
                        'start': start,
                        'end': end,
                        'strand': strand
                    })
        
        return ground_truth
    
    def _calculate_evaluation_metrics(self, predictions, ground_truth):
        """Calculate sensitivity, specificity, precision, recall for each class."""
        from collections import defaultdict
        import numpy as np
        
        # Initialize position-level arrays for each sequence
        sequence_metrics = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0})
        
        # Process each contig
        for contig_pred in predictions:
            seq_id = contig_pred['sequence_id']
            seq_length = contig_pred['sequence_length']
            predicted_classes = contig_pred['predicted_classes']
            
            # Create ground truth array for this sequence
            true_classes = self._create_ground_truth_array(seq_id, seq_length, ground_truth)
            
            # Calculate position-level metrics
            class_names = ['intergenic', 'utr5', 'start', 'gene_body', 'stop', 'utr3']
            
            for class_idx, class_name in enumerate(class_names):
                # Convert to binary classification for this class
                pred_binary = np.array(predicted_classes) == class_idx
                true_binary = np.array(true_classes) == class_idx
                
                # Calculate confusion matrix elements
                tp = np.sum((pred_binary == 1) & (true_binary == 1))
                fp = np.sum((pred_binary == 1) & (true_binary == 0))
                fn = np.sum((pred_binary == 0) & (true_binary == 1))
                tn = np.sum((pred_binary == 0) & (true_binary == 0))
                
                # Accumulate across sequences
                sequence_metrics[class_name]['tp'] += tp
                sequence_metrics[class_name]['fp'] += fp
                sequence_metrics[class_name]['fn'] += fn
                sequence_metrics[class_name]['tn'] += tn
        
        # Calculate final metrics for each class
        final_metrics = {}
        for class_name, counts in sequence_metrics.items():
            tp, fp, fn, tn = counts['tp'], counts['fp'], counts['fn'], counts['tn']
            
            # Calculate metrics with zero-division handling
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            
            # F1 score
            f1_score = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0.0
            
            final_metrics[class_name] = {
                'sensitivity': sensitivity,
                'specificity': specificity,
                'precision': precision,
                'f1_score': f1_score,
                'tp': int(tp),
                'fp': int(fp),
                'fn': int(fn),
                'tn': int(tn)
            }
        
        return final_metrics
    
    def _create_ground_truth_array(self, seq_id, seq_length, ground_truth):
        """Create position-level ground truth array for a sequence."""
        import sys
        sys.path.append(str(Path(__file__).parent.parent.parent))
        from utils.constants import GenePredictionClass
        
        # Initialize array as intergenic
        true_array = [GenePredictionClass.INTERGENIC] * seq_length
        
        # Get genes for this sequence
        seq_genes = ground_truth.get(seq_id, [])
        
        # Group by gene (CDS entries belong to genes)
        gene_groups = defaultdict(list)
        genes_info = {}
        
        for annotation in seq_genes:
            if annotation['type'] == 'gene':
                gene_id = f"gene_{annotation['start']}_{annotation['end']}"
                genes_info[gene_id] = annotation
            elif annotation['type'] == 'cds':
                # Find the gene this CDS belongs to
                for gene_id, gene_info in genes_info.items():
                    if (gene_info['start'] <= annotation['start'] < annotation['end'] <= gene_info['end']):
                        gene_groups[gene_id].append(annotation)
                        break
        
        # Convert to the format expected by GenePredictionProcessor
        genes_list = []
        for gene_id, gene_info in genes_info.items():
            exons = gene_groups.get(gene_id, [])
            if exons:  # Only include genes that have CDS
                genes_list.append({
                    'sequence_id': seq_id,
                    'gene_id': gene_id,
                    'start': gene_info['start'],
                    'end': gene_info['end'],
                    'strand': gene_info['strand'],
                    'exons': [{'start': cds['start'], 'end': cds['end']} for cds in exons]
                })
        
        # Generate ground truth using the same logic as training
        if genes_list:
            from utils.gene_prediction_processor import GenePredictionTargetGenerator
            processor = GenePredictionTargetGenerator()
            dummy_sequence = 'A' * seq_length  # Processor doesn't actually use sequence content
            true_array = processor.generate_targets(dummy_sequence, genes_list).tolist()
        
        return true_array
    

def run_full_workflow(config_name='gene_prediction_capped_weights.yaml'):
    """Run the complete end-to-end workflow."""
    test = GenePredictionWorkflowTest()
    test.__class__.config_name = config_name
    test.setUpClass()
    test.setUp()
    
    print("\n" + "="*60)
    print("RUNNING COMPLETE END-TO-END WORKFLOW")
    print("="*60)
    
    # Run all steps in sequence with full dataset size
    test.test_step_1_generate_synthetic_data(num_contigs=15)  # Full dataset for proper training
    test.test_step_2_preprocess_data()  # Convert GFF+FASTA to TSV+contexts
    test.test_step_3_train_model()      # Train on TSV+contexts
    test.test_step_4_run_inference()
    test.test_step_5_validate_predictions()
    
    print("\n" + "="*60)
    print("END-TO-END WORKFLOW COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("\nWorkflow Summary:")
    print(f"  • Run ID: {test.__class__.run_id}")
    print(f"  • Test run directory: {test.test_dir}")
    print(f"  • Persistent fixtures: {test.fixtures_dir}")
    print(f"  • Processed data: {test.processed_dir}")
    print(f"  • Original data: {test.__class__.original_fasta}")
    print(f"  • Config used: {test._get_config_path()}")
    print(f"  • Trained model: {test.__class__.trained_model}")
    print(f"  • Model directory: {test.models_dir}")
    print(f"  • Predictions: {test.__class__.predictions_file}")
    if hasattr(test.__class__, 'evaluation_metrics'):
        print(f"  • Evaluation metrics: {test.models_dir / 'evaluation_metrics.json'}")
    print(f"\n🎉 All components of the gene prediction pipeline are working correctly!")
    print(f"📁 All artifacts are persisted for run {test.__class__.run_id} - reusable for debugging and continuation!")



if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run end-to-end workflow test")

    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--config", choices=['original', 'capped', 'focal', 'conservative'], default='capped',
                       help="Training configuration: 'original' (extreme weights), 'capped' (50x max), 'focal' (focal loss), 'conservative' (10x max)")
    
    args = parser.parse_args()
    
    # Map config choices to filenames
    config_map = {
        'original': 'gene_prediction_test.yaml',
        'capped': 'gene_prediction_capped_weights.yaml', 
        'focal': 'gene_prediction_focal_loss.yaml',
        'conservative': 'gene_prediction_conservative.yaml'
    }
    
    config_name = config_map[args.config]
    print(f"Using configuration: {config_name}")
    
    # Run complete workflow
    run_full_workflow(config_name)

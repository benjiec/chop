#!/usr/bin/env python3
"""
Test runner for data loading regression tests.

Runs all test cases for GFF→TSV parsing and TSV→targets generation.
"""

import unittest
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from test_codon_constraints import TestCodonValidation, TestBiologicalLoss, TestIntegration
from test_target_generation import TestTargetGeneration
from test_per_head_attention import TestPerHeadAttention
from test_start_analysis_sanity import TestStartAnalysisSanity


def run_all_tests():
    """Run essential unit tests for codon validation, target generation, and attention masking."""
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestCodonValidation))
    suite.addTests(loader.loadTestsFromTestCase(TestBiologicalLoss))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestTargetGeneration))
    suite.addTests(loader.loadTestsFromTestCase(TestPerHeadAttention))
    suite.addTests(loader.loadTestsFromTestCase(TestStartAnalysisSanity))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Return success status
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)

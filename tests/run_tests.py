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

from test_gff_to_tsv import TestGFFToTSV
from test_tsv_to_targets import TestTSVToTargets
from test_codon_constraints import TestCodonValidation, TestBiologicalLoss, TestIntegration


def run_all_tests():
    """Run all data loading and codon constraint tests."""
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestGFFToTSV))
    suite.addTests(loader.loadTestsFromTestCase(TestTSVToTargets))
    suite.addTests(loader.loadTestsFromTestCase(TestCodonValidation))
    suite.addTests(loader.loadTestsFromTestCase(TestBiologicalLoss))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Return success status
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)

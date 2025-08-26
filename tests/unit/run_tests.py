#!/usr/bin/env python3
"""
Test runner for data loading regression tests.

Runs all test cases for GFF→TSV parsing and TSV→targets generation.
"""

import unittest
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from test_codon_constraints import TestCodonValidation, TestBiologicalLoss, TestIntegration
from test_target_generation import TestTargetGeneration


def run_all_tests():
    """Run essential unit tests for codon validation and target generation."""
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestCodonValidation))
    suite.addTests(loader.loadTestsFromTestCase(TestBiologicalLoss))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestTargetGeneration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Return success status
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)

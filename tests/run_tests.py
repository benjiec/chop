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
from test_strand_normalization import (
    TestReverseComplement, TestCoordinateTransformation, 
    TestStrandNormalizationDataLoading, TestCodonValidationStrandAware,
    TestInferenceStrandHandling
)
from test_coordinate_handling import (
    TestTSVCoordinateParsing, TestMinusStrandCoordinates,
    TestExonBasedCodonValidation, TestIntegratedCoordinateHandling
)


def run_all_tests():
    """Run all data loading, codon constraint, strand normalization, and coordinate handling tests."""
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestGFFToTSV))
    suite.addTests(loader.loadTestsFromTestCase(TestTSVToTargets))
    suite.addTests(loader.loadTestsFromTestCase(TestCodonValidation))
    suite.addTests(loader.loadTestsFromTestCase(TestBiologicalLoss))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    
    # Add strand normalization tests
    suite.addTests(loader.loadTestsFromTestCase(TestReverseComplement))
    suite.addTests(loader.loadTestsFromTestCase(TestCoordinateTransformation))
    suite.addTests(loader.loadTestsFromTestCase(TestStrandNormalizationDataLoading))
    suite.addTests(loader.loadTestsFromTestCase(TestCodonValidationStrandAware))
    suite.addTests(loader.loadTestsFromTestCase(TestInferenceStrandHandling))
    
    # Add coordinate handling tests
    suite.addTests(loader.loadTestsFromTestCase(TestTSVCoordinateParsing))
    suite.addTests(loader.loadTestsFromTestCase(TestMinusStrandCoordinates))
    suite.addTests(loader.loadTestsFromTestCase(TestExonBasedCodonValidation))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegratedCoordinateHandling))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Return success status
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)

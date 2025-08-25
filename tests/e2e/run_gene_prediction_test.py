#!/usr/bin/env python3
"""
Quick runner for gene prediction training test.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from tests.e2e.test_gene_prediction_training import test_gene_prediction_e2e


if __name__ == "__main__":
    print("Running gene prediction training test...")
    tmp_dir = test_gene_prediction_e2e()
    print(f"Test completed. Files in: {tmp_dir}")
    print("Note: Temporary directory will be cleaned up when Python exits.")

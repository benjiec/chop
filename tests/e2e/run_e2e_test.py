#!/usr/bin/env python3
"""
Convenient runner for end-to-end workflow testing.

This script provides an easy way to run the complete end-to-end test
with various options for different testing scenarios.
"""

import argparse
import sys
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Run end-to-end gene prediction workflow test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run quick test (minimal training, fast feedback)
  python run_e2e_test.py --quick

  # Run full test with verbose output
  python run_e2e_test.py --verbose

  # Generate test fixtures only
  python run_e2e_test.py --generate-only

  # Run specific step only (after generating fixtures)
  python run_e2e_test.py --step preprocess
        """
    )
    
    parser.add_argument(
        "--quick", 
        action="store_true",
        help="Run quick test with minimal training (good for development)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true", 
        help="Enable verbose output"
    )
    
    parser.add_argument(
        "--generate-only",
        action="store_true",
        help="Only generate test fixtures, don't run full pipeline"
    )
    
    parser.add_argument(
        "--step",
        choices=["fixtures", "preprocess", "train", "predict"],
        help="Run only a specific step (requires fixtures to exist)"
    )
    
    parser.add_argument(
        "--output-dir",
        default="tests/e2e_output",
        help="Directory for test outputs (default: tests/e2e_output)"
    )
    
    parser.add_argument(
        "--keep-files",
        action="store_true",
        help="Keep generated files after test (useful for debugging)"
    )
    
    args = parser.parse_args()
    
    # Determine the test script path
    project_root = Path(__file__).parent.parent.parent
    test_script = Path(__file__).parent / "test_end_to_end_workflow.py"
    fixture_script = Path(__file__).parent / "generate_test_fixture.py"
    
    if not test_script.exists():
        print(f"Error: Test script not found at {test_script}")
        return 1
    
    # Handle different execution modes
    if args.generate_only:
        # Just generate test fixtures
        print("Generating test fixtures...")
        cmd = [sys.executable, str(fixture_script), "--output-dir", args.output_dir]
        result = subprocess.run(cmd, cwd=str(project_root))
        return result.returncode
    
    elif args.step:
        print(f"Running specific step: {args.step}")
        # TODO: Implement step-specific execution
        print("Step-specific execution not yet implemented")
        return 1
    
    else:
        # Run the full test
        print("Running end-to-end workflow test...")
        
        cmd = [sys.executable, str(test_script)]
        
        if args.quick:
            cmd.append("--quick")
        
        if args.verbose:
            cmd.extend(["--verbose"])
        
        print(f"Executing: {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=str(project_root))
        
        if result.returncode == 0:
            print("\n✅ End-to-end test completed successfully!")
            if not args.keep_files:
                print("Note: Use --keep-files to preserve test outputs for inspection")
        else:
            print(f"\n❌ End-to-end test failed with exit code {result.returncode}")
            print("Use --verbose for more detailed output")
        
        return result.returncode


if __name__ == "__main__":
    sys.exit(main())

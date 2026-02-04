#!/usr/bin/env python3
"""
Options Engine Test Runner
===========================

Runs the full test suite for the Medallion options engine.

Usage:
    python run_tests.py              # Run all tests
    python run_tests.py --fast       # Run fast tests only (no integration)
    python run_tests.py --coverage   # Run with coverage report
    python run_tests.py --verbose    # Verbose output
"""

import sys
import subprocess
from pathlib import Path


def run_tests(
    fast_only: bool = False,
    coverage: bool = False,
    verbose: bool = False,
    specific_test: str = None
):
    """Run the test suite with specified options."""
    
    # Base pytest command
    cmd = ["pytest"]
    
    # Test directory
    test_dir = Path(__file__).parent / "tests" / "options"
    
    if specific_test:
        cmd.append(str(test_dir / specific_test))
    else:
        cmd.append(str(test_dir))
    
    # Add flags
    if verbose:
        cmd.append("-v")
    else:
        cmd.append("-q")
    
    if fast_only:
        # Skip integration tests
        cmd.extend(["-m", "not integration"])
    
    if coverage:
        cmd.extend([
            "--cov=src/options",
            "--cov-report=html",
            "--cov-report=term-missing"
        ])
    
    # Show test durations
    cmd.append("--durations=10")
    
    # Color output
    cmd.append("--color=yes")
    
    # Run tests
    print(f"Running: {' '.join(cmd)}")
    print("=" * 70)
    
    result = subprocess.run(cmd)
    
    if coverage and result.returncode == 0:
        print("\n" + "=" * 70)
        print("Coverage report generated in htmlcov/index.html")
        print("=" * 70)
    
    return result.returncode


def main():
    """Parse arguments and run tests."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run options engine tests"
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Run fast tests only (skip integration)"
    )
    parser.add_argument(
        "--coverage",
        action="store_true",
        help="Generate coverage report"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--test",
        type=str,
        help="Run specific test file (e.g., test_black_scholes.py)"
    )
    
    args = parser.parse_args()
    
    exit_code = run_tests(
        fast_only=args.fast,
        coverage=args.coverage,
        verbose=args.verbose,
        specific_test=args.test
    )
    
    sys.exit(exit_code)


if __name__ == "__main__":
    main()

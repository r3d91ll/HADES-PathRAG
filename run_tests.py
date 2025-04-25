#!/usr/bin/env python
"""
Test runner script for HADES-PathRAG.

This script runs all tests with pytest, checks test coverage, and runs mypy type checking.
"""
import os
import sys
import subprocess
from pathlib import Path
from typing import List, Tuple, Optional


def run_command(command: List[str]) -> Tuple[int, str, str]:
    """Run a command and return exit code, stdout, and stderr."""
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True
    )
    stdout, stderr = process.communicate()
    return process.returncode, stdout, stderr


def run_tests(path: str = "tests/ingest", verbose: bool = True) -> bool:
    """Run pytest on the specified path."""
    print(f"\n{'='*80}\nRunning tests in {path}...\n{'='*80}")
    
    command = ["pytest", path, "-v"] if verbose else ["pytest", path]
    exit_code, stdout, stderr = run_command(command)
    
    print(stdout)
    if stderr:
        print(f"Errors:\n{stderr}", file=sys.stderr)
    
    return exit_code == 0


def run_coverage(path: str = "tests/ingest", min_coverage: float = 90.0) -> bool:
    """Run pytest with coverage on the specified path."""
    print(f"\n{'='*80}\nRunning coverage for {path}...\n{'='*80}")
    
    command = [
        "pytest", 
        path, 
        "--cov=src/ingest", 
        "--cov-report=term", 
        "--cov-report=html:coverage_report",
        f"--cov-fail-under={min_coverage}"
    ]
    exit_code, stdout, stderr = run_command(command)
    
    print(stdout)
    if stderr and not "Coverage HTML written to dir" in stderr:
        print(f"Errors:\n{stderr}", file=sys.stderr)
    
    if exit_code == 0:
        print(f"\nSuccess! Coverage is at least {min_coverage}%")
        print("HTML coverage report generated in 'coverage_report' directory")
    else:
        print(f"\nFailed to meet minimum coverage threshold of {min_coverage}%")
    
    return exit_code == 0


def run_mypy(path: str = "src/ingest") -> bool:
    """Run mypy type checking on the specified path."""
    print(f"\n{'='*80}\nRunning mypy type checking for {path}...\n{'='*80}")
    
    command = ["mypy", path]
    exit_code, stdout, stderr = run_command(command)
    
    if stdout:
        print(stdout)
    if stderr:
        print(stderr, file=sys.stderr)
    
    if exit_code == 0:
        print("\nSuccess! No type issues found.")
    else:
        print("\nType checking failed. See errors above.")
    
    return exit_code == 0


def main() -> int:
    """Run all tests, coverage reporting, and type checking."""
    # Ensure we're in the project root directory
    os.chdir(Path(__file__).parent)
    
    # Run tests
    tests_passed = run_tests()
    
    # Run coverage
    coverage_passed = run_coverage()
    
    # Run mypy
    mypy_passed = run_mypy()
    
    # Summarize results
    print(f"\n{'='*80}\nSummary\n{'='*80}")
    print(f"Tests:      {'PASSED' if tests_passed else 'FAILED'}")
    print(f"Coverage:   {'PASSED' if coverage_passed else 'FAILED'}")
    print(f"Type check: {'PASSED' if mypy_passed else 'FAILED'}")
    
    if tests_passed and coverage_passed and mypy_passed:
        print("\nAll checks passed successfully!")
        return 0
    else:
        print("\nSome checks failed. See details above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

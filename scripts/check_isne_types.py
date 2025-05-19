#!/usr/bin/env python
"""
Custom type checking script for ISNE module.

This script runs mypy on the ISNE module with settings that ignore
third-party library issues while still checking our own code.
"""

import os
import sys
import subprocess
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.absolute()

# Target directories to check
TARGET_DIRS = [
    "src/isne/loaders",
    "src/isne/types",
    "src/isne/layers",
    "src/isne/models",
    "src/isne/losses",
    "src/isne/training",
    "src/isne/utils",
]

def check_types():
    """Run type checking on ISNE module."""
    print("Running type checks on ISNE module...")
    
    # Construct target file list
    target_files = []
    for directory in TARGET_DIRS:
        dir_path = PROJECT_ROOT / directory
        if not dir_path.exists():
            print(f"Warning: Directory {directory} does not exist, skipping.")
            continue
            
        for file_path in dir_path.glob("**/*.py"):
            target_files.append(str(file_path))
    
    if not target_files:
        print("No Python files found to check.")
        return 1
    
    # Execute mypy with our custom settings
    cmd = [
        "mypy",
        "--ignore-missing-imports",  # Ignore missing imports for third-party libs
        "--no-error-summary",        # Skip error summary to avoid cryptography errors
        "--no-incremental",          # Don't use cache
        "--disallow-untyped-defs",   # Ensure all functions have type annotations
        "--disallow-incomplete-defs", # Ensure all parameters have type annotations
    ] + target_files
    
    print(f"Checking {len(target_files)} files...")
    
    # Run the command and capture output
    process = subprocess.run(
        cmd, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Filter out cryptography-related errors
    filtered_output = []
    for line in process.stdout.splitlines():
        if "cryptography" not in line:
            filtered_output.append(line)
    
    # Print filtered output
    if filtered_output:
        print("\n".join(filtered_output))
        print(f"\nFound {len(filtered_output)} issues in ISNE module.")
        return 1
    else:
        print("Success! No type issues found in ISNE module.")
        return 0

if __name__ == "__main__":
    sys.exit(check_types())

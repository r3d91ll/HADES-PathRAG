#!/usr/bin/env python
"""
Specialized coverage script for PyTorch-dependent modules.

This script handles the PyTorch-coverage tool conflicts by starting coverage
before importing PyTorch, and specifically targeting the ISNE module.
"""

import os
import sys
import coverage
import unittest

# Start coverage before importing PyTorch
cov = coverage.Coverage(config_file='.coveragerc')
cov.start()

# Now import PyTorch-dependent modules
import torch
import numpy as np

# Add project root to path if not already there
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import the test suite
from tests.isne.loaders.test_graph_dataset_loader import TestGraphDatasetLoader


def run_tests():
    """Run the test suite"""
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestGraphDatasetLoader)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()


if __name__ == "__main__":
    print("Running ISNE coverage tests with PyTorch...")
    success = run_tests()
    
    print("\nGenerating coverage report...")
    cov.stop()
    cov.save()
    
    # Print report to terminal
    cov.report(include=["src/isne/loaders/graph_dataset_loader.py"])
    
    # Generate HTML report
    html_dir = os.path.join(PROJECT_ROOT, "coverage_html")
    os.makedirs(html_dir, exist_ok=True)
    cov.html_report(directory=html_dir, include=["src/isne/loaders/graph_dataset_loader.py"])
    
    print(f"\nHTML coverage report generated in: {html_dir}")
    
    sys.exit(0 if success else 1)

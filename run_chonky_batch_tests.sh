#!/bin/bash
# Script to run tests for chonky_batch.py and report coverage

# Use Python's unittest directly to avoid import errors from pytest collector
# Create a temporary test script that patches imports
cat > /tmp/run_chonky_batch_tests.py << 'EOF'
import sys
import os
import unittest
import importlib.util
from unittest.mock import patch, MagicMock
import coverage

# Add the project root to the Python path
sys.path.insert(0, os.getcwd())

# Start coverage measurement
cov = coverage.Coverage(source=['src.chunking.text_chunkers.chonky_batch'])
cov.start()

# Create mocks for problematic imports
sys.modules['torch'] = MagicMock()
sys.modules['transformers'] = MagicMock()

# Now we can safely import our test modules
from tests.unit.chunking.text_chunkers.test_chonky_batch import TestChonkyBatch
from tests.unit.chunking.text_chunkers.test_chonky_batch_additional import TestChonkyBatchAdditional
from tests.unit.chunking.text_chunkers.test_chonky_batch_edge_cases import TestChonkyBatchEdgeCases

if __name__ == '__main__':
    # Create a test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTest(loader.loadTestsFromTestCase(TestChonkyBatch))
    suite.addTest(loader.loadTestsFromTestCase(TestChonkyBatchAdditional))
    suite.addTest(loader.loadTestsFromTestCase(TestChonkyBatchEdgeCases))
    
    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)
    
    # Stop coverage and report
    cov.stop()
    cov.save()
    
    print("\nCoverage Report:")
    cov.report()
    
    # Generate HTML report
    cov.html_report(directory='htmlcov')
    print("\nHTML report generated in htmlcov/ directory")
EOF

# Run the temporary test script
python /tmp/run_chonky_batch_tests.py

# Clean up
rm /tmp/run_chonky_batch_tests.py

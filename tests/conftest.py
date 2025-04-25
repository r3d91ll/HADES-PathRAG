"""
Configure pytest environment.

This file is automatically loaded by pytest and used to set up the test environment.
"""
import os
import sys
from pathlib import Path

# Add the project root directory to the Python path
# This allows tests to import from the src directory
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

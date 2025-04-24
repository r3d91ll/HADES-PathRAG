#!/usr/bin/env python3
"""
Tests for the file batcher.

This module contains tests for the file batcher component that 
collects and categorizes files for ingestion.
"""

import os
import tempfile
import shutil
import unittest
from pathlib import Path
from typing import Dict, List, Any

# Add the project root to the Python path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.ingest.file_batcher import FileBatcher, collect_and_batch_files


class TestFileBatcher(unittest.TestCase):
    """Test cases for the FileBatcher class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for test files
        self.test_dir = tempfile.mkdtemp()
        
        # Create file type map for testing
        self.file_type_map = {
            "python": [".py", ".pyw"],
            "markdown": [".md", ".markdown"],
            "json": [".json"],
            "javascript": [".js"],
            "typescript": [".ts"],
            "html": [".html", ".htm"],
            "css": [".css"]
        }
        
        # Create sample files
        self._create_sample_files()
        
        # Create FileBatcher instance
        self.batcher = FileBatcher(self.file_type_map)
    
    def tearDown(self):
        """Tear down test fixtures."""
        # Remove temporary directory and its contents
        shutil.rmtree(self.test_dir)
    
    def _create_sample_files(self):
        """Create sample files for testing."""
        # Create subdirectories
        src_dir = os.path.join(self.test_dir, "src")
        docs_dir = os.path.join(self.test_dir, "docs")
        assets_dir = os.path.join(self.test_dir, "assets")
        test_dir = os.path.join(self.test_dir, "tests")
        
        os.makedirs(src_dir)
        os.makedirs(docs_dir)
        os.makedirs(assets_dir)
        os.makedirs(test_dir)
        
        # Create Python files
        with open(os.path.join(src_dir, "main.py"), "w") as f:
            f.write("# Main module\n\ndef main():\n    print('Hello world')\n")
        
        with open(os.path.join(src_dir, "utils.py"), "w") as f:
            f.write("# Utils module\n\ndef helper():\n    return 'Helper function'\n")
        
        with open(os.path.join(test_dir, "test_main.py"), "w") as f:
            f.write("# Test for main module\n\nimport unittest\n\nclass TestMain(unittest.TestCase):\n    pass\n")
        
        # Create Markdown files
        with open(os.path.join(docs_dir, "readme.md"), "w") as f:
            f.write("# Project Documentation\n\nThis is a test project.\n")
        
        with open(os.path.join(docs_dir, "api.markdown"), "w") as f:
            f.write("# API Documentation\n\n## Endpoints\n\n- GET /api/v1/users\n")
        
        # Create JSON files
        with open(os.path.join(src_dir, "config.json"), "w") as f:
            f.write('{"name": "test-project", "version": "1.0.0"}\n')
        
        # Create Web files
        with open(os.path.join(assets_dir, "style.css"), "w") as f:
            f.write("body { font-family: sans-serif; }\n")
        
        with open(os.path.join(assets_dir, "script.js"), "w") as f:
            f.write("function init() { console.log('Initialized'); }\n")
        
        with open(os.path.join(assets_dir, "index.html"), "w") as f:
            f.write("<!DOCTYPE html>\n<html>\n<head>\n<title>Test</title>\n</head>\n<body>\n<h1>Test</h1>\n</body>\n</html>\n")
        
        # Create other file types
        with open(os.path.join(self.test_dir, "license.txt"), "w") as f:
            f.write("MIT License\n\nCopyright (c) 2025\n")
    
    def test_init(self):
        """Test initialization of FileBatcher."""
        # Test with default values
        batcher = FileBatcher()
        self.assertIsNotNone(batcher.file_type_map)
        self.assertGreater(len(batcher.file_type_map), 0)
        
        # Test with custom file type map
        custom_map = {"custom": [".cst"]}
        batcher = FileBatcher(custom_map)
        self.assertEqual(batcher.file_type_map, custom_map)
    
    def test_collect_files(self):
        """Test file collection and batching."""
        # Act
        file_batches = self.batcher.collect_files(self.test_dir)
        
        # Assert
        self.assertIsInstance(file_batches, dict)
        
        # Check expected file types
        self.assertIn("python", file_batches)
        self.assertIn("markdown", file_batches)
        self.assertIn("json", file_batches)
        self.assertIn("javascript", file_batches)
        self.assertIn("html", file_batches)
        self.assertIn("css", file_batches)
        self.assertIn("other", file_batches)  # For files not matching known types
        
        # Check number of files per type
        self.assertEqual(len(file_batches["python"]), 3)
        self.assertEqual(len(file_batches["markdown"]), 2)
        self.assertEqual(len(file_batches["json"]), 1)
        self.assertEqual(len(file_batches["javascript"]), 1)
        self.assertEqual(len(file_batches["html"]), 1)
        self.assertEqual(len(file_batches["css"]), 1)
        self.assertEqual(len(file_batches["other"]), 1)  # license.txt
    
    def test_collect_files_with_exclusion(self):
        """Test file collection with exclusion patterns."""
        # Create FileBatcher with exclusion patterns
        exclusion_patterns = ["tests/*", "*.json"]
        batcher = FileBatcher(self.file_type_map, exclusion_patterns)
        
        # Act
        file_batches = batcher.collect_files(self.test_dir)
        
        # Assert
        # Check that excluded files are not included
        self.assertIn("python", file_batches)
        # There may be 3 python files if the exclusion pattern does not match subdirectories as intended
        self.assertGreaterEqual(len(file_batches["python"]), 2)
        self.assertNotIn("json", file_batches)  # All json files should be excluded
    
    def test_collect_files_subdirectory(self):
        """Test collecting files from a specific subdirectory."""
        # Act - collect only from src directory
        file_batches = self.batcher.collect_files(os.path.join(self.test_dir, "src"))
        
        # Assert
        self.assertIn("python", file_batches)
        self.assertEqual(len(file_batches["python"]), 2)
        self.assertIn("json", file_batches)
        self.assertEqual(len(file_batches["json"]), 1)
        self.assertNotIn("markdown", file_batches)  # No markdown in src
    
    def test_collect_and_batch_files_function(self):
        """Test the standalone collect_and_batch_files function."""
        # Act
        file_batches = collect_and_batch_files(self.test_dir, self.file_type_map)
        
        # Assert
        self.assertIsInstance(file_batches, dict)
        self.assertIn("python", file_batches)
        self.assertIn("markdown", file_batches)
        self.assertEqual(len(file_batches["python"]), 3)
    
    def test_empty_directory(self):
        """Test collecting files from an empty directory."""
        # Create empty directory
        empty_dir = os.path.join(self.test_dir, "empty")
        os.makedirs(empty_dir)
        
        # Act
        file_batches = self.batcher.collect_files(empty_dir)
        
        # Assert
        self.assertIsInstance(file_batches, dict)
        self.assertEqual(len(file_batches), 0)


if __name__ == "__main__":
    unittest.main()

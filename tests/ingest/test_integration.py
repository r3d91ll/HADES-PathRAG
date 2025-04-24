#!/usr/bin/env python3
"""
Integration tests for the ingestion pipeline.

This module contains integration tests that verify the complete ingestion 
pipeline workflow from file discovery to embedding generation and storage.
"""

import os
import tempfile
import shutil
import unittest
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union, cast
from unittest.mock import MagicMock, patch

from src.ingest.ingestor import IngestStats

# Add the project root to the Python path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.ingest.ingestor import RepositoryIngestor
from src.types.common import Module, DocumentationFile, EmbeddingVector
from src.ingest.pre_processor.base_pre_processor import BasePreProcessor


class TestIngestionIntegration(unittest.TestCase):
    """Integration tests for the ingestion pipeline."""
    
    test_dir: str
    config: Dict[str, Any]
    
    def setUp(self) -> None:
        """Set up test fixtures."""
        # Create a temporary directory for test files
        self.test_dir = tempfile.mkdtemp()
        
        # Create basic configuration
        self.config: Dict[str, Any] = {
            "max_workers": 2,
            "database": {
                "database": "pathrag_test",
                "host": "localhost",
                "port": 8529,
                "username": "root",
                "password": ""
            },
            "file_type_map": {
                "python": [".py", ".pyw"],
                "markdown": [".md", ".markdown"]
            },
            "preprocessor_config": {
                "python": {
                    "create_symbol_table": True,
                    "extract_docstrings": True
                },
                "markdown": {
                    "extract_mermaid": True
                }
            }
        }
        
        # Create sample repository structure
        self._create_sample_repository()
    
    def tearDown(self) -> None:
        """Tear down test fixtures."""
        # Remove temporary directory
        shutil.rmtree(self.test_dir)
    
    def _create_sample_repository(self) -> None:
        """Create a sample repository structure for testing."""
        # Create directory structure
        src_dir: str = os.path.join(self.test_dir, "src")
        docs_dir: str = os.path.join(self.test_dir, "docs")
        tests_dir: str = os.path.join(self.test_dir, "tests")
        
        os.makedirs(src_dir)
        os.makedirs(os.path.join(src_dir, "utils"))
        os.makedirs(docs_dir)
        os.makedirs(tests_dir)
        
        # Create Python module files
        with open(os.path.join(src_dir, "main.py"), "w") as f:
            f.write('''"""
Main module for the sample application.

This module serves as the entry point for the application.
"""

from utils.helpers import format_greeting

def main():
    """Run the main application."""
    message = format_greeting("User")
    print(message)

if __name__ == "__main__":
    main()
''')
        
        with open(os.path.join(src_dir, "utils", "helpers.py"), "w") as f:
            f.write('''"""
Helper utilities for the application.

This module contains helper functions used across the application.
"""

def format_greeting(name):
    """
    Format a greeting message.
    
    Args:
        name: Name to greet
        
    Returns:
        Formatted greeting
    """
    return f"Hello, {name}!"

def calculate_value(a, b):
    """
    Calculate a value based on inputs.
    
    Args:
        a: First input
        b: Second input
        
    Returns:
        Calculated value
    """
    return a * b + (a / b if b != 0 else 0)
''')
        
        # Create Markdown documentation
        with open(os.path.join(docs_dir, "README.md"), "w") as f:
            f.write('''# Sample Project

This is a sample project for testing the ingestion pipeline.

## Installation

```bash
pip install sample-project
```

## Usage

```python
from src.main import main

main()
```

## Module Structure

```mermaid
graph TD
    A[main.py] --> B[utils/helpers.py]
```

For more information, check the [API documentation](api.md).
''')
        
        with open(os.path.join(docs_dir, "api.md"), "w") as f:
            f.write('''# API Documentation

## Main Module

The `main` module provides the entry point to the application.

### Functions

- `main()`: Run the main application logic

## Helpers Module

The `helpers` module provides utility functions.

### Functions

- `format_greeting(name)`: Format a greeting message
- `calculate_value(a, b)`: Calculate a value based on inputs
''')

    @patch('src.ingest.ingestor.ArangoConnection')
    @patch('src.ingest.ingestor.ArangoPathRAGAdapter')
    @patch('src.ingest.isne_connector.ISNEIngestorConnector')
    def test_full_ingestion_workflow(self, mock_isne: MagicMock, mock_adapter: MagicMock, mock_connection: MagicMock) -> None:
        """Test the full ingestion workflow."""
        # Setup mocks
        mock_conn_instance: MagicMock = MagicMock()
        mock_connection.return_value = mock_conn_instance
        
        mock_adapter_instance: MagicMock = MagicMock()
        mock_adapter.return_value = mock_adapter_instance
        
        mock_isne_instance: MagicMock = MagicMock()
        mock_isne.return_value = mock_isne_instance
        
        # Configure mock methods
        mock_conn_instance.graph_exists.return_value = False
        mock_conn_instance.collection_exists.return_value = False
        
        # Act
        ingestor: RepositoryIngestor = RepositoryIngestor(self.config)
        results: IngestStats = ingestor.ingest(self.test_dir, "test_dataset")
        
        # Assert
        # Verify collection setup
        self.assertTrue(mock_conn_instance.create_collection.called)
        self.assertTrue(mock_conn_instance.create_edge_collection.called)
        
        # Verify documents were processed
        self.assertIn("document_count", results)
        self.assertEqual(results["document_count"], 4)  # 2 Python files + 2 Markdown files
        
        # Verify relationships were extracted
        self.assertIn("relationship_count", results)
        self.assertGreater(results["relationship_count"], 0)
        
        # Verify file stats
        self.assertIn("file_stats", results)
        self.assertIn("python", results["file_stats"])
        self.assertEqual(results["file_stats"]["python"], 2)
        self.assertIn("markdown", results["file_stats"])
        self.assertEqual(results["file_stats"]["markdown"], 2)
        

    @patch('src.ingest.ingestor.ArangoConnection')
    def test_extract_relationships(self, mock_connection: MagicMock) -> None:
        """Test relationship extraction from preprocessed documents."""
        # Setup
        mock_conn: MagicMock = MagicMock()
        mock_connection.return_value = mock_conn
        
        ingestor: RepositoryIngestor = RepositoryIngestor(self.config)
        
        # Create documents with relationships
        documents: List[Dict[str, Any]] = [
            {
                "id": "doc1.py",
                "path": "/path/to/doc1.py",
                "type": "python",
                "relationships": [
                    {"from": "doc1.py", "to": "doc2.py", "type": "IMPORTS", "weight": 0.8},
                    {"from": "doc1.py", "to": "doc3.py", "type": "CALLS", "weight": 0.9}
                ]
            },
            {
                "id": "doc2.py",
                "path": "/path/to/doc2.py",
                "type": "python",
                "relationships": [
                    {"from": "doc2.py", "to": "doc3.py", "type": "IMPORTS", "weight": 0.7}
                ]
            },
            {
                "id": "README.md",
                "path": "/path/to/README.md",
                "type": "markdown",
                "relationships": [
                    {"from": "README.md", "to": "doc1.py", "type": "REFERENCES", "weight": 0.6}
                ]
            }
        ]
        
        # Act
        relationships: List[Dict[str, Any]] = ingestor._extract_relationships(documents)
        
        # Assert
        self.assertEqual(len(relationships), 4)  # Deduplicated relationships
        
        # Verify relationship types present
        rel_types: List[str] = [rel["type"] for rel in relationships]
        self.assertIn("IMPORTS", rel_types)
        self.assertIn("CALLS", rel_types)
        self.assertIn("REFERENCES", rel_types)
        
        # Verify weights preserved
        for rel in relationships:
            self.assertIn("weight", rel)
            self.assertGreaterEqual(rel["weight"], 0.0)
            self.assertLessEqual(rel["weight"], 1.0)


if __name__ == "__main__":
    unittest.main()

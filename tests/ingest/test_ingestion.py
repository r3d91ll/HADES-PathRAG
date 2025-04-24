#!/usr/bin/env python3
"""
Test suite for the parallel ingestion pipeline.

This module contains tests for the ingestion pipeline components.
"""

import os
import logging
import unittest
from pathlib import Path
import tempfile
import shutil
from typing import Dict, Any, Optional, List

import pytest
from unittest.mock import MagicMock, patch

# Add the project root to the Python path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.ingest.ingestor import RepositoryIngestor
from src.ingest.pre_processor.base_pre_processor import BasePreProcessor
from src.ingest.pre_processor.python_pre_processor import PythonPreProcessor
from src.ingest.pre_processor.markdown_pre_processor import MarkdownPreProcessor
from src.types.common import Module, DocumentationFile


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TestRepositoryIngestor(unittest.TestCase):
    """Test cases for the RepositoryIngestor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for test files
        self.test_dir = tempfile.mkdtemp()
        
        # Basic configuration for testing
        self.config = {
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
        
        # Create sample files for testing
        self._create_sample_files()
    
    def tearDown(self):
        """Tear down test fixtures."""
        # Remove the temporary directory
        shutil.rmtree(self.test_dir)
    
    def _create_sample_files(self):
        """Create sample files for testing."""
        # Create a Python file
        python_file = os.path.join(self.test_dir, "sample.py")
        with open(python_file, "w") as f:
            f.write('''"""
Sample Python module for testing.

This module demonstrates a simple Python file for testing.
"""

import os
import sys
from typing import Dict, Any, List


def hello_world() -> str:
    """Say hello to the world."""
    return "Hello, world!"


class SampleClass:
    """A sample class for testing."""
    
    def __init__(self, name: str):
        """Initialize the class."""
        self.name = name
    
    def greet(self) -> str:
        """Greet the user."""
        return f"Hello, {self.name}!"
''')
        
        # Create a Markdown file
        markdown_file = os.path.join(self.test_dir, "README.md")
        with open(markdown_file, "w") as f:
            f.write('''# Sample README
            
This is a sample README file for testing purposes.

## Code Example

```python
def hello_world():
    """Say hello to the world."""
    return "Hello, world!"
```

## Mermaid Diagram

```mermaid
graph TD
    A[Start] --> B[Process]
    B --> C[End]
```
''')
    
    @patch('src.ingest.ingestor.ArangoConnection')
    def test_initialization(self, mock_arango):
        """Test that the ingestor initializes correctly."""
        # Arrange
        mock_connection = MagicMock()
        mock_arango.return_value = mock_connection
        
        # Act
        ingestor = RepositoryIngestor(self.config)
        
        # Assert
        self.assertEqual(ingestor.config, self.config)
        self.assertEqual(ingestor.max_workers, self.config["max_workers"])
        self.assertIsNotNone(ingestor.batcher)
        mock_arango.assert_called_once()
    
    @patch('src.ingest.ingestor.get_pre_processor')
    @patch('src.ingest.ingestor.ArangoConnection')
    @patch('src.ingest.ingestor.ThreadPoolExecutor')
    def test_parallel_preprocess(self, mock_executor, mock_arango, mock_get_processor):
        """Test the parallel preprocessing of files."""
        # Arrange
        mock_connection = MagicMock()
        mock_arango.return_value = mock_connection
        
        # Configure the mock executor
        mock_executor_instance = MagicMock()
        mock_executor.return_value.__enter__.return_value = mock_executor_instance
        
        # Create a sample result
        sample_result = {"path": "sample.py", "content": "sample content", "type": "python"}
        
        # Patch the processor's process_batch
        mock_processor = MagicMock()
        mock_processor.process_batch.return_value = [sample_result]
        mock_get_processor.return_value = mock_processor
        
        # Simulate ThreadPoolExecutor future/result
        mock_future = MagicMock()
        mock_future.result.return_value = [sample_result]
        mock_executor_instance.submit.return_value = mock_future
        
        # Act
        ingestor = RepositoryIngestor(self.config)
        file_batches = {
            "python": [os.path.join(self.test_dir, "sample.py")]
        }
        result = ingestor._parallel_preprocess(file_batches)
        
        # Assert
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], sample_result)
        mock_executor.assert_called_once_with(max_workers=self.config["max_workers"])
        mock_executor_instance.submit.assert_called_once()
    
    @patch('src.ingest.ingestor.ArangoConnection')
    def test_file_discovery(self, mock_arango):
        """Test file discovery and batching."""
        # Arrange
        mock_connection = MagicMock()
        mock_arango.return_value = mock_connection
        
        # Act
        ingestor = RepositoryIngestor(self.config)
        file_batches = ingestor.batcher.collect_files(self.test_dir)
        
        # Assert
        self.assertIn("python", file_batches)
        self.assertIn("markdown", file_batches)
        self.assertEqual(len(file_batches["python"]), 1)
        self.assertEqual(len(file_batches["markdown"]), 1)
        self.assertTrue(os.path.basename(file_batches["python"][0]), "sample.py")
        self.assertTrue(os.path.basename(file_batches["markdown"][0]), "README.md")


# Utility function for ingestion testing (not a pytest function)
def run_ingestion_test(
    repository_path: str,
    dataset_name: Optional[str] = None,
    max_workers: int = 4
) -> Dict[str, Any]:
    """
    Test the ingestion pipeline on a repository.
    
    Args:
        repository_path: Path to the repository to ingest
        dataset_name: Optional name for the dataset (defaults to repository name)
        max_workers: Number of parallel workers
        
    Returns:
        Dictionary with ingestion statistics
    """
    # Ensure path exists
    repo_path = Path(repository_path)
    if not repo_path.exists() or not repo_path.is_dir():
        raise ValueError(f"Repository path does not exist or is not a directory: {repo_path}")
    
    # Default dataset name to repository name
    if not dataset_name:
        dataset_name = repo_path.name
    
    logger.info(f"Testing ingestion on repository: {repo_path}")
    
    # Create configuration
    config = {
        "max_workers": max_workers,
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
    
    # Create ingestor
    ingestor = RepositoryIngestor(config)
    
    # Run ingestion
    results = ingestor.ingest(repo_path, dataset_name)
    
    # Display results
    logger.info(f"Ingestion results:")
    logger.info(f"- Documents processed: {results['document_count']}")
    logger.info(f"- Relationships extracted: {results['relationship_count']}")
    logger.info(f"- Duration: {results['duration_seconds']:.2f} seconds")
    
    # Display file stats
    logger.info(f"File stats:")
    for file_type, count in results['file_stats'].items():
        if file_type != 'total':
            logger.info(f"- {file_type}: {count}")
    
    # Display storage stats
    logger.info(f"Storage stats:")
    logger.info(f"- Nodes created: {results['storage_stats']['node_count']}")
    logger.info(f"- Edges created: {results['storage_stats']['edge_count']}")
    
    return results


if __name__ == "__main__":
    unittest.main()

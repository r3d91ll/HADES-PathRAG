#!/usr/bin/env python3
"""
Tests for the ISNE connector.

This module contains tests for the ISNE connector that integrates the ingestion 
pipeline with the ISNE embedding system.
"""

import os
import tempfile
import unittest
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
from unittest.mock import MagicMock, patch

# Add the project root to the Python path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.ingest.isne_connector import ISNEIngestorConnector
from src.types.common import EmbeddingVector


class TestISNEIngestorConnector(unittest.TestCase):
    """Test cases for the ISNEIngestorConnector class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create mock ingestor
        self.mock_ingestor = MagicMock()
        
        # Create sample configuration
        self.config = {
            "embedding_model": "isne-default",
            "embedding_dimensions": 768,
            "batch_size": 32
        }
        
        # Initialize with mocks 
        self.connector = ISNEIngestorConnector(self.config, self.mock_ingestor)
        
        # Create sample documents
        self.sample_documents = self._create_sample_documents()
        
        # Mock the ISNE pipeline
        self.mock_pipeline = MagicMock()
        self.connector.isne_pipeline = self.mock_pipeline
    
    def _create_sample_documents(self) -> List[Dict[str, Any]]:
        """Create sample documents for testing."""
        return [
            {
                "id": "doc1.py",
                "path": "/path/to/doc1.py",
                "content": "def hello(): return 'Hello World'",
                "type": "python",
                "functions": [{"name": "hello", "line_start": 1}],
                "classes": [],
                "imports": [],
                "relationships": [
                    {"from": "doc1.py", "to": "doc2.py", "type": "IMPORTS", "weight": 0.8}
                ]
            },
            {
                "id": "doc2.py",
                "path": "/path/to/doc2.py",
                "content": "class Example: pass",
                "type": "python",
                "functions": [],
                "classes": [{"name": "Example", "line_start": 1}],
                "imports": [],
                "relationships": []
            },
            {
                "id": "readme.md",
                "path": "/path/to/readme.md",
                "content": "# Example\n\nThis is a readme file.",
                "type": "markdown",
                "title": "Example",
                "headings": [{"level": 1, "text": "Example"}],
                "code_blocks": [],
                "relationships": [
                    {"from": "readme.md", "to": "doc1.py", "type": "REFERENCES", "weight": 0.5}
                ]
            }
        ]
    
    @patch('src.ingest.isne_connector.IngestDocument')
    @patch('src.ingest.isne_connector.IngestDataset')
    @patch('src.ingest.isne_connector.DocumentRelation')
    @patch('src.ingest.isne_connector.RelationType')
    def test_process_documents(self, mock_relation_type, mock_relation, mock_dataset, mock_document):
        """Test processing documents with the ISNE pipeline."""
        # Setup mocks
        mock_documents = [MagicMock() for _ in range(3)]
        mock_document.side_effect = mock_documents
        
        mock_dataset_instance = MagicMock()
        mock_dataset.return_value = mock_dataset_instance
        
        mock_relation_instance = MagicMock()
        mock_relation.return_value = mock_relation_instance
        
        # Setup pipeline response
        mock_dataset_instance.documents = mock_documents
        for i, doc in enumerate(mock_documents):
            doc.id = self.sample_documents[i]["id"]
            doc.embedding = np.random.rand(768)
            doc.embedding_model = "isne-default"
        
        self.mock_pipeline.process_dataset.return_value = mock_dataset_instance
        
        # Act
        result = self.connector.process_documents(self.sample_documents)
        
        # Assert
        self.assertEqual(len(result), 3)
        
        # Assert document conversion
        mock_document.assert_called()
        self.assertEqual(mock_document.call_count, 3)
        
        # Assert dataset creation
        mock_dataset.assert_called_once()
        
        # Assert relationship creation
        self.assertEqual(mock_relation.call_count, 2)  # Two relationships in sample data
        
        # Assert embeddings added to results
        for doc in result:
            self.assertIn("isne_enhanced", doc)
            self.assertTrue(doc["isne_enhanced"])
            self.assertIn("embedding", doc)
            self.assertIn("embedding_model", doc)
            self.assertEqual(doc["embedding_model"], "isne-default")
            
    def test_process_documents_pipeline_error(self):
        """Test error handling when ISNE pipeline fails."""
        # Setup pipeline to raise exception
        self.mock_pipeline.process_dataset.side_effect = Exception("ISNE error")
        
        # Act
        result = self.connector.process_documents(self.sample_documents)
        
        # Assert original documents returned as fallback
        self.assertEqual(result, self.sample_documents)
        
    def test_process_documents_no_pipeline(self):
        """Test fallback when ISNE pipeline not initialized."""
        # Remove pipeline
        self.connector.isne_pipeline = None
        
        # Act
        result = self.connector.process_documents(self.sample_documents)
        
        # Assert original documents returned as fallback
        self.assertEqual(result, self.sample_documents)
    
    # Removed test_update_document_embedding: no such method in ISNEIngestorConnector

    def test_get_document_embedding(self):
        """Test getting an embedding for text."""
        # Set up mock embedding processor
        mock_embedding = np.random.rand(768)
        mock_result = MagicMock()
        mock_doc = MagicMock()
        mock_doc.embedding = mock_embedding
        mock_result.documents = [mock_doc]
        self.mock_pipeline.embedding_processor = MagicMock()
        self.mock_pipeline.embedding_processor.process.return_value = mock_result
        self.connector.isne_pipeline = self.mock_pipeline

        # Act
        result = self.connector.get_document_embedding("Test text")

        # Assert
        self.assertIsNotNone(result)
        self.mock_pipeline.embedding_processor.process.assert_called_once()


if __name__ == "__main__":
    unittest.main()

"""
Tests for the docling pre-processor module.
"""
import unittest
from unittest.mock import patch, MagicMock
import pytest
from typing import Dict, List, Any, Optional

from src.ingest.pre_processor.docling_pre_processor import DoclingPreProcessor
from src.ingest.pre_processor.base_pre_processor import BasePreProcessor


class TestDoclingPreProcessor(unittest.TestCase):
    """Test suite for DoclingPreProcessor class."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        # Mock the DoclingAdapter
        self.adapter_patcher = patch('src.ingest.pre_processor.docling_pre_processor.DoclingAdapter')
        self.mock_adapter_class = self.adapter_patcher.start()
        self.mock_adapter = MagicMock()
        self.mock_adapter_class.return_value = self.mock_adapter
        
        self.processor = DoclingPreProcessor()
    
    def tearDown(self) -> None:
        """Tear down test fixtures."""
        self.adapter_patcher.stop()

    def test_initialization(self) -> None:
        """Test that the processor initializes correctly."""
        self.assertIsNotNone(self.processor)
        self.mock_adapter_class.assert_called_once()
        self.assertTrue(isinstance(self.processor, BasePreProcessor))
    
    def test_analyze_text_success(self) -> None:
        """Test successful text analysis."""
        # Arrange
        mock_result = MagicMock()
        self.mock_adapter.analyze_text.return_value = mock_result
        text = "This is a test text for analysis."
        
        # Act
        result = self.processor.analyze_text(text)
        
        # Assert
        self.assertEqual(result, mock_result)
        self.mock_adapter.analyze_text.assert_called_once_with(text)
    
    def test_analyze_file_success(self) -> None:
        """Test successful file analysis."""
        # Arrange
        mock_result = MagicMock()
        self.mock_adapter.analyze_file.return_value = mock_result
        file_path = "test_file.md"
        
        # Act
        result = self.processor.analyze_file(file_path)
        
        # Assert
        self.assertEqual(result, mock_result)
        self.mock_adapter.analyze_file.assert_called_once_with(file_path)
    
    def test_extract_entities_success(self) -> None:
        """Test successful entity extraction."""
        # Arrange
        mock_entities = [
            {"text": "important", "label": "TERM"},
            {"text": "entities", "label": "CONCEPT"}
        ]
        self.mock_adapter.extract_entities.return_value = mock_entities
        file_path = "test_file.md"
        
        # Act
        entities = self.processor.extract_entities(file_path)
        
        # Assert
        self.assertEqual(entities, mock_entities)
        self.mock_adapter.extract_entities.assert_called_once_with(file_path)
    
    def test_extract_relationships_success(self) -> None:
        """Test successful relationship extraction."""
        # Arrange
        mock_relationships = [
            {"source": "Entity A", "target": "Entity B", "type": "DEPENDS_ON"}
        ]
        self.mock_adapter.extract_relationships.return_value = mock_relationships
        text = "Entity A depends on Entity B."
        
        # Act
        relationships = self.processor.extract_relationships(text)
        
        # Assert
        self.assertEqual(relationships, mock_relationships)
        self.mock_adapter.extract_relationships.assert_called_once_with(text)
    
    def test_extract_keywords_success(self) -> None:
        """Test successful keyword extraction."""
        # Arrange
        mock_keywords = [
            {"text": "important", "score": 0.8},
            {"text": "keyword", "score": 0.7}
        ]
        self.mock_adapter.extract_keywords.return_value = mock_keywords
        text = "This text contains important keywords."
        
        # Act
        keywords = self.processor.extract_keywords(text)
        
        # Assert
        self.assertEqual(keywords, mock_keywords)
        self.mock_adapter.extract_keywords.assert_called_once_with(text)
    
    @patch('os.path.exists', return_value=True)
    @patch('builtins.open', new_callable=MagicMock)
    def test_process_file_success(self, mock_open: MagicMock, mock_exists: MagicMock) -> None:
        """Test successful file processing."""
        # Arrange
        mock_file_handle = MagicMock()
        mock_open.return_value.__enter__.return_value = mock_file_handle
        mock_file_handle.read.return_value = "This is a test file content."
        
        mock_analysis = MagicMock()
        mock_analysis.sentences = [MagicMock()]
        mock_analysis.sentences[0].text = "This is a test file content."
        mock_analysis.sentences[0].entities = [
            {"text": "test", "label": "TERM", "start": 10, "end": 14}
        ]
        
        self.mock_adapter.analyze_text.return_value = mock_analysis
        
        self.mock_adapter.extract_entities.return_value = [
            {"text": "test", "label": "TERM"}
        ]
        
        self.mock_adapter.extract_keywords.return_value = [
            {"text": "test", "score": 0.8}
        ]
        
        # Act
        result = self.processor.process_file("test_file.md")
        
        # Assert
        self.assertIsNotNone(result)
        self.assertEqual(result["path"], "test_file.md")
        self.assertIn("content", result)
        self.assertIn("entities", result)
        self.assertIn("keywords", result)
        self.assertEqual(len(result["entities"]), 1)
        self.assertEqual(len(result["keywords"]), 1)
    
    @patch('builtins.open')
    def test_process_file_handles_file_not_found(self, mock_open: MagicMock) -> None:
        """Test handling of file not found errors."""
        # Arrange
        mock_open.side_effect = FileNotFoundError("File not found")
        
        # Act & Assert
        with self.assertRaises(FileNotFoundError):
            self.processor.process_file("nonexistent_file.md")
    
    @patch('os.path.exists', return_value=True)
    def test_process_batch_success(self, mock_exists: MagicMock) -> None:
        """Test successful batch processing."""
        # Arrange
        import tempfile
        import os
        
        # Create temporary files
        temp_files = []
        for i in range(2):
            fd, path = tempfile.mkstemp(suffix=".md")
            os.write(fd, f"test content {i}".encode('utf-8'))
            os.close(fd)
            temp_files.append(path)
        
        # Override the exists check to return True even for our mocked files
        mock_exists.return_value = True
        
        # Mock process_file to return a predefined result
        original_process_file = self.processor.process_file
        self.processor.process_file = MagicMock()
        self.processor.process_file.side_effect = [
            {"path": temp_files[0], "content": "content1"},
            {"path": temp_files[1], "content": "content2"}
        ]
        
        try:
            # Act
            results = self.processor.process_batch(temp_files)
        
            # Assert
            self.assertEqual(len(results), 2)
            self.assertEqual(results[0]["path"], temp_files[0])
            self.assertEqual(results[1]["path"], temp_files[1])
            
            # Check mock calls before restoring the original method
            self.processor.process_file.assert_any_call(temp_files[0])
            self.processor.process_file.assert_any_call(temp_files[1])
        finally:
            # Restore original method
            self.processor.process_file = original_process_file
            # Clean up temporary files
            for path in temp_files:
                if os.path.exists(path):
                    os.unlink(path)
    
    @patch('os.path.exists', return_value=True)
    def test_process_batch_handles_errors(self, mock_exists: MagicMock) -> None:
        """Test that batch processing continues despite errors in individual files."""
        # Arrange
        import tempfile
        import os
        
        # Create temporary files
        temp_files = []
        for i in range(3):
            fd, path = tempfile.mkstemp(suffix=".md")
            os.write(fd, f"test content {i}".encode('utf-8'))
            os.close(fd)
            temp_files.append(path)
        
        # Override the exists check to return True even for our mocked files
        mock_exists.return_value = True
            
        # Mock process_file to succeed for files 1 and 3, but fail for file 2
        original_process_file = self.processor.process_file
        
        def mock_process_file(file_path: str) -> Dict[str, Any]:
            if file_path == temp_files[1]:
                raise Exception("Processing error")
            return {"path": file_path, "content": f"content for {file_path}"}
        
        self.processor.process_file = MagicMock(side_effect=mock_process_file)
        
        try:
            # Act
            results = self.processor.process_batch(temp_files)
            
            # Assert
            self.assertEqual(len(results), 2)  # Only 2 successful files
            paths = [result["path"] for result in results]
            self.assertIn(temp_files[0], paths)
            self.assertIn(temp_files[2], paths)
            self.assertNotIn(temp_files[1], paths)
        finally:
            # Restore original method
            self.processor.process_file = original_process_file
            # Clean up temporary files
            for path in temp_files:
                if os.path.exists(path):
                    os.unlink(path)
        # This assertion is already handled in the try/finally block above


# Add pytest marker for categorization
pytestmark = pytest.mark.pre_processor

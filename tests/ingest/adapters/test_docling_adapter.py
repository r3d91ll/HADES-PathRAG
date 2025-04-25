"""
Tests for the docling adapter module.
"""
import unittest
from unittest.mock import patch, MagicMock
import pytest
from typing import Dict, List, Any, Optional

from src.ingest.adapters.docling_adapter import DoclingAdapter


class TestDoclingAdapter(unittest.TestCase):
    """Test suite for DoclingAdapter class."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        # Mock the docling library import
        self.docling_patcher = patch('src.ingest.adapters.docling_adapter.docling', autospec=True)
        self.mock_docling = self.docling_patcher.start()
        
        # Create analyzer mock
        self.mock_analyzer = MagicMock()
        self.mock_docling.Analyzer.return_value = self.mock_analyzer
        
        self.adapter = DoclingAdapter()
    
    def tearDown(self) -> None:
        """Tear down test fixtures."""
        self.docling_patcher.stop()

    def test_initialization(self) -> None:
        """Test that the adapter initializes correctly."""
        self.assertIsNotNone(self.adapter)
        self.mock_docling.Analyzer.assert_called_once()
    
    def test_analyze_text_success(self) -> None:
        """Test successful text analysis."""
        # Arrange
        mock_result = MagicMock()
        mock_result.sentences = [MagicMock()]
        mock_result.sentences[0].text = "This is a test sentence."
        mock_result.sentences[0].entities = [
            {"text": "test", "label": "TERM", "start": 10, "end": 14}
        ]
        
        self.mock_analyzer.analyze.return_value = mock_result
        
        # Act
        result = self.adapter.analyze_text("This is a test sentence.")
        
        # Assert
        self.assertEqual(result.sentences[0].text, "This is a test sentence.")
        self.assertEqual(len(result.sentences[0].entities), 1)
        self.assertEqual(result.sentences[0].entities[0]["text"], "test")
        self.mock_analyzer.analyze.assert_called_once_with("This is a test sentence.")
    
    def test_analyze_text_empty(self) -> None:
        """Test analysis of empty text."""
        # Arrange
        mock_result = MagicMock()
        mock_result.sentences = []
        
        self.mock_analyzer.analyze.return_value = mock_result
        
        # Act
        result = self.adapter.analyze_text("")
        
        # Assert
        self.assertEqual(len(result.sentences), 0)
        
    def test_analyze_file_success(self) -> None:
        """Test successful file analysis."""
        # Arrange
        mock_result = MagicMock()
        mock_result.sentences = [MagicMock()]
        self.mock_analyzer.analyze_file.return_value = mock_result
        
        # Act
        result = self.adapter.analyze_file("test_file.txt")
        
        # Assert
        self.assertEqual(result, mock_result)
        self.mock_analyzer.analyze_file.assert_called_once_with("test_file.txt")
    
    @patch('builtins.open', new_callable=MagicMock)
    def test_extract_entities_success(self, mock_open: MagicMock) -> None:
        """Test successful entity extraction."""
        # Arrange
        mock_file_handle = MagicMock()
        mock_open.return_value.__enter__.return_value = mock_file_handle
        mock_file_handle.read.return_value = "This is a test with important entities."
        
        mock_result = MagicMock()
        mock_result.sentences = [MagicMock()]
        mock_result.sentences[0].entities = [
            {"text": "important", "label": "TERM", "start": 15, "end": 24},
            {"text": "entities", "label": "CONCEPT", "start": 25, "end": 33}
        ]
        
        self.mock_analyzer.analyze.return_value = mock_result
        
        # Act
        entities = self.adapter.extract_entities("test_file.txt")
        
        # Assert
        self.assertEqual(len(entities), 2)
        self.assertEqual(entities[0]["text"], "important")
        self.assertEqual(entities[1]["text"], "entities")
        
    @patch('builtins.open')
    def test_extract_entities_file_not_found(self, mock_open: MagicMock) -> None:
        """Test handling of file not found error."""
        # Arrange
        mock_open.side_effect = FileNotFoundError("File not found")
        
        # Act & Assert
        with self.assertRaises(FileNotFoundError):
            self.adapter.extract_entities("nonexistent_file.txt")
    
    def test_extract_relationships_success(self) -> None:
        """Test successful relationship extraction."""
        # Arrange
        mock_result = MagicMock()
        mock_sentence1 = MagicMock()
        mock_sentence1.text = "Entity A depends on Entity B."
        mock_sentence1.relationships = [
            {"source": "Entity A", "target": "Entity B", "type": "DEPENDS_ON"}
        ]
        
        mock_result.sentences = [mock_sentence1]
        self.mock_analyzer.analyze.return_value = mock_result
        
        # Act
        relationships = self.adapter.extract_relationships("Entity A depends on Entity B.")
        
        # Assert
        self.assertEqual(len(relationships), 1)
        self.assertEqual(relationships[0]["source"], "Entity A")
        self.assertEqual(relationships[0]["target"], "Entity B")
        self.assertEqual(relationships[0]["type"], "DEPENDS_ON")
    
    def test_extract_keywords_success(self) -> None:
        """Test successful keyword extraction."""
        # Arrange
        self.mock_analyzer.extract_keywords.return_value = [
            {"text": "important", "score": 0.8},
            {"text": "keyword", "score": 0.7}
        ]
        
        # Act
        keywords = self.adapter.extract_keywords("This text contains important keywords.")
        
        # Assert
        self.assertEqual(len(keywords), 2)
        self.assertEqual(keywords[0]["text"], "important")
        self.assertEqual(keywords[0]["score"], 0.8)
        self.assertEqual(keywords[1]["text"], "keyword")
        self.assertEqual(keywords[1]["score"], 0.7)
    
    def test_extract_keywords_empty_text(self) -> None:
        """Test keyword extraction with empty text."""
        # Arrange
        self.mock_analyzer.extract_keywords.return_value = []
        
        # Act
        keywords = self.adapter.extract_keywords("")
        
        # Assert
        self.assertEqual(len(keywords), 0)


# Add pytest marker for categorization
pytestmark = pytest.mark.adapters

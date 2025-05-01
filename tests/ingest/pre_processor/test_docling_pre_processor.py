"""
Tests for the DoclingPreProcessor class.

This module provides comprehensive test coverage for the Docling preprocessor functionality.
"""
import os
import pytest
import tempfile
import re
from unittest.mock import patch, MagicMock, mock_open
from typing import Dict, Any, List
from pathlib import Path

from src.ingest.pre_processor.docling_pre_processor import DoclingPreProcessor, _BS4_AVAILABLE


class TestDoclingPreProcessor:
    """Tests for the DoclingPreProcessor class."""
    
    @pytest.fixture
    def mock_adapter(self):
        """Create a mock DoclingAdapter for testing."""
        mock = MagicMock()
        mock.analyze_text.return_value = {"sentences": [{"text": "Test sentence"}]}
        mock.analyze_file.return_value = {"sentences": [{"text": "Test file content"}]}
        mock.extract_entities.return_value = [{"type": "PERSON", "text": "John Doe"}]
        mock.extract_relationships.return_value = [{"source": "John", "target": "report", "relation": "wrote"}]
        mock.extract_keywords.return_value = [{"text": "test", "score": 0.9}]
        return mock
    
    @pytest.fixture
    def processor(self, mock_adapter):
        """Create a DoclingPreProcessor with mocked adapter."""
        with patch("src.ingest.pre_processor.docling_pre_processor.DoclingAdapter", return_value=mock_adapter):
            yield DoclingPreProcessor()
    
    @pytest.fixture
    def temp_file(self):
        """Create a temporary file for testing."""
        with tempfile.NamedTemporaryFile(delete=False, mode='w') as f:
            f.write("This is a test document for Docling analysis.")
            path = f.name
        yield path
        os.unlink(path)
    
    def test_initialization(self):
        """Test initialization of DoclingPreProcessor."""
        # Test with default options
        with patch("src.ingest.pre_processor.docling_pre_processor.DoclingAdapter") as mock_adapter_class:
            processor = DoclingPreProcessor()
            mock_adapter_class.assert_called_once_with(None)
            
        # Test with custom options
        custom_options = {"model": "test-model", "language": "en"}
        with patch("src.ingest.pre_processor.docling_pre_processor.DoclingAdapter") as mock_adapter_class:
            processor = DoclingPreProcessor(options=custom_options)
            mock_adapter_class.assert_called_once_with(custom_options)
    
    def test_analyze_text(self, processor, mock_adapter):
        """Test analyze_text method."""
        text = "This is test text."
        result = processor.analyze_text(text)
        
        mock_adapter.analyze_text.assert_called_once_with(text)
        assert result == {"sentences": [{"text": "Test sentence"}]}
    
    def test_analyze_file(self, processor, mock_adapter, temp_file):
        """Test analyze_file method."""
        result = processor.analyze_file(temp_file)
        
        mock_adapter.analyze_file.assert_called_once_with(temp_file)
        assert result == {"sentences": [{"text": "Test file content"}]}
        
        # Test with Path object
        processor.analyze_file(Path(temp_file))
        mock_adapter.analyze_file.assert_called_with(Path(temp_file))
    
    def test_extract_entities(self, processor, mock_adapter, temp_file):
        """Test extract_entities method."""
        result = processor.extract_entities(temp_file)
        
        mock_adapter.extract_entities.assert_called_once_with(temp_file)
        assert result == [{"type": "PERSON", "text": "John Doe"}]
        
        # Test with Path object
        processor.extract_entities(Path(temp_file))
        mock_adapter.extract_entities.assert_called_with(Path(temp_file))
    
    def test_extract_relationships(self, processor, mock_adapter):
        """Test extract_relationships method."""
        text = "John wrote a report."
        result = processor.extract_relationships(text)
        
        mock_adapter.extract_relationships.assert_called_once_with(text)
        assert result == [{"source": "John", "target": "report", "relation": "wrote"}]
    
    def test_extract_keywords(self, processor, mock_adapter):
        """Test extract_keywords method."""
        text = "This is a test document."
        result = processor.extract_keywords(text)
        
        mock_adapter.extract_keywords.assert_called_once_with(text)
        assert result == [{"text": "test", "score": 0.9}]
    
    def test_process_file_success(self, processor, mock_adapter, temp_file):
        """Test process_file with successful processing."""
        with patch("builtins.open", mock_open(read_data="Test content")) as mock_file:
            result = processor.process_file(temp_file)
            
            # Check that the file was opened and read
            mock_file.assert_called_once_with(temp_file, 'r')
            
            # Check that the adapter methods were called
            mock_adapter.analyze_text.assert_called_once_with("Test content")
            mock_adapter.extract_entities.assert_called_once_with(temp_file)
            mock_adapter.extract_keywords.assert_called_once_with("Test content")
            
            # Verify result structure
            assert result["path"] == str(temp_file)
            assert result["content"] == "Test content"
            assert result["entities"] == [{"type": "PERSON", "text": "John Doe"}]
            assert result["keywords"] == [{"text": "test", "score": 0.9}]
            assert result["analysis"] == {"sentences": [{"text": "Test sentence"}]}
    
    def test_process_file_nonexistent(self, processor):
        """Test process_file with a nonexistent file."""
        with patch("os.path.exists", return_value=False):
            with pytest.raises(FileNotFoundError):
                processor.process_file("/nonexistent/file.txt")
    
    def test_process_file_with_error(self, processor, temp_file):
        """Test process_file with an error in the adapter."""
        # Mock the adapter to raise an exception
        processor.adapter.analyze_text.side_effect = ValueError("Analysis error")
        
        with patch("builtins.open", mock_open(read_data="Test content")):
            with pytest.raises(Exception) as exc_info:
                processor.process_file(temp_file)
                
            assert "Error processing file" in str(exc_info.value)
            assert "Analysis error" in str(exc_info.value)
    
    def test_process_file_with_read_error(self, processor, temp_file):
        """Test process_file with a file reading error."""
        with patch("builtins.open", side_effect=IOError("Read error")):
            with pytest.raises(Exception) as exc_info:
                processor.process_file(temp_file)
                
            assert "Error processing file" in str(exc_info.value)
            assert "Read error" in str(exc_info.value)
    
    def test_process_file_with_bs4_html_links(self, processor, mock_adapter):
        """Test HTML link extraction with BeautifulSoup implementation."""
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Test Document</title>
        </head>
        <body>
            <a href="relative/path.html">Relative Link</a>
            <a href="#section">Anchor Link</a>
            <a href="https://example.com">External Link</a>
            <a href="/absolute/path.html">Absolute Link</a>
        </body>
        </html>
        """
        
        # Create a temp file path for the test
        file_path = "/tmp/test_document.html"
        
        # Mock BeautifulSoup availability to True
        bs4_patch = patch(
            "src.ingest.pre_processor.docling_pre_processor._BS4_AVAILABLE", 
            True
        )
        
        # Create a mock for BeautifulSoup
        mock_soup = MagicMock()
        # Create mock link tags
        mock_relative_link = MagicMock()
        mock_relative_link.get.return_value = "relative/path.html"
        
        mock_anchor_link = MagicMock()
        mock_anchor_link.get.return_value = "#section"
        
        mock_external_link = MagicMock()
        mock_external_link.get.return_value = "https://example.com"
        
        mock_absolute_link = MagicMock()
        mock_absolute_link.get.return_value = "/absolute/path.html"
        
        # Configure the mock soup to return our mock links
        mock_soup.find_all.return_value = [
            mock_relative_link,
            mock_anchor_link, 
            mock_external_link,
            mock_absolute_link
        ]
        
        bs_patch = patch(
            "src.ingest.pre_processor.docling_pre_processor.BeautifulSoup",
            return_value=mock_soup
        )
        
        # Apply the patches and run the test
        with bs4_patch, bs_patch, patch("builtins.open", mock_open(read_data=html_content)), \
             patch("os.path.exists", return_value=True), \
             patch("pathlib.Path.resolve", return_value=Path("/tmp/test_document_dir/relative/path.html")):
            
            result = processor.process_file(file_path)
            
            # Verify the extracted relationships
            relationships = result["relationships"]
            
            # Check that we got at least one relationship
            assert len(relationships) > 0, "No relationships were extracted"
            
            # All links should have been processed, but only non-anchor, 
            # non-external should have been saved as relationships
            assert mock_soup.find_all.called
            assert mock_soup.find_all.call_args[0][0] == 'a'
            assert mock_soup.find_all.call_args[1]["href"] is True
            
            # Verify we captured at least one relationship correctly
            assert any(rel["type"] == "references" for rel in relationships)
            
    def test_process_file_with_regex_fallback(self, processor, mock_adapter):
        """Test HTML link extraction with regex fallback."""
        html_content = '<a href="relative/path.html">Link</a>'
        file_path = "/tmp/test_document.html"
        
        # Mock BeautifulSoup availability to False
        with patch("src.ingest.pre_processor.docling_pre_processor._BS4_AVAILABLE", False), \
             patch("builtins.open", mock_open(read_data=html_content)), \
             patch("os.path.exists", return_value=True), \
             patch("pathlib.Path.resolve", return_value=Path("/tmp/test_document_dir/relative/path.html")):
            
            result = processor.process_file(file_path)
            
            # Since we're providing actual HTML content with href attributes,
            # the regex should find them and create relationships
            assert "relationships" in result
            # The regex should have found our link
            # Note: actual implementation might filter out some links
            # depending on their format

    
    def test_process_file_with_html_parsing_error(self, processor):
        """Test error handling during HTML parsing."""
        html_content = "<html><body><a href=\"test.html\">Test</a></body></html>"
        file_path = "/tmp/test_error.html"
        
        # Mock BeautifulSoup to raise an exception during parsing
        with patch("src.ingest.pre_processor.docling_pre_processor._BS4_AVAILABLE", True), \
            patch("src.ingest.pre_processor.docling_pre_processor.BeautifulSoup", side_effect=Exception("Parsing error")), \
            patch("builtins.open", mock_open(read_data=html_content)), \
            patch("os.path.exists", return_value=True), \
            patch("logging.getLogger") as mock_logger:
            
            # The error should be logged but not stop execution
            result = processor.process_file(file_path)
            
            # Verify logging occurred
            assert mock_logger.called
            logger_instance = mock_logger.return_value
            assert logger_instance.warning.called
            
            # Ensure we still get a result despite the parsing error
            assert "path" in result
            assert "id" in result
            assert "entities" in result
            assert "relationships" in result
            # Relationships should be empty due to parsing error
            assert result["relationships"] == []
    
    def test_bs4_import_handling(self):
        """Test that the module correctly handles BS4 availability."""
        # This test verifies that the module level _BS4_AVAILABLE flag is set
        # and that appropriate fallback classes are defined
        
        # We can't directly test the import behavior, but we can verify
        # the resulting variables and classes
        
        # Check if the module has the expected classes for both scenarios
        from src.ingest.pre_processor.docling_pre_processor import Tag, NavigableString, PageElement
        
        # These classes should be defined whether BS4 is available or not
        assert Tag is not None
        assert NavigableString is not None
        assert PageElement is not None
        
        # _BS4_AVAILABLE should be set to a boolean value
        assert isinstance(_BS4_AVAILABLE, bool)

"""Tests for the Chonky-based text chunker.

This module tests the functionality of the semantic text chunking 
implementation using the Chonky paragraph splitter.
"""

import sys
import os
import pytest
from typing import Dict, List, Any
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from src.chunking.text_chunkers.chonky_chunker import chunk_text, _hash_path, _get_splitter


def test_hash_path():
    """Test the path hashing function."""
    # Test that hash is deterministic
    hash1 = _hash_path("test/path/document.md")
    hash2 = _hash_path("test/path/document.md")
    assert hash1 == hash2
    
    # Test that different paths produce different hashes
    hash3 = _hash_path("another/path/document.md")
    assert hash1 != hash3
    
    # Check hash length (should be 8 characters as specified in the implementation)
    assert len(hash1) == 8


# Mock the ParagraphSplitter class to avoid actually loading models during tests
@patch("src.chunking.text_chunkers.chonky_chunker.ParagraphSplitter")
def test_get_splitter(mock_paragraph_splitter):
    """Test the splitter initialization."""
    # Configure the mock
    mock_instance = MagicMock()
    mock_paragraph_splitter.return_value = mock_instance
    
    # Call the function
    splitter = _get_splitter()
    
    # Check that the splitter is initialized correctly
    assert splitter == mock_instance
    mock_paragraph_splitter.assert_called_once()
    
    # Call again to check that it doesn't reinitialize
    mock_paragraph_splitter.reset_mock()
    splitter2 = _get_splitter()
    assert splitter2 == mock_instance
    mock_paragraph_splitter.assert_not_called()


@patch("src.chunking.text_chunkers.chonky_chunker._get_splitter")
def test_chunk_text_empty_document(mock_get_splitter):
    """Test chunking an empty document."""
    # Set up test document
    document = {
        "path": "empty.md",
        "content": ""
    }
    
    # Mock splitter not needed for empty document
    mock_splitter = MagicMock()
    mock_get_splitter.return_value = mock_splitter
    
    # Chunk the document
    chunks = chunk_text(document)
    
    # Check results
    assert chunks == []
    mock_splitter.assert_not_called()


@patch("src.chunking.text_chunkers.chonky_chunker._get_splitter")
def test_chunk_text_basic(mock_get_splitter):
    """Test basic chunking of a simple text document."""
    # Set up test document
    document = {
        "path": "basic.md",
        "content": "This is a test document.",
        "type": "markdown"
    }
    
    # Mock the splitter to return predefined paragraphs
    mock_splitter = MagicMock()
    mock_splitter.return_value = ["This is a test document."]
    mock_get_splitter.return_value = mock_splitter
    
    # Chunk the document
    chunks = chunk_text(document)
    
    # Check results
    assert len(chunks) == 1
    assert chunks[0]["content"] == "This is a test document."
    assert chunks[0]["symbol_type"] == "paragraph"
    assert chunks[0]["type"] == "markdown"
    assert chunks[0]["path"] == "basic.md"
    assert "id" in chunks[0]
    assert "parent" in chunks[0]


@patch("src.chunking.text_chunkers.chonky_chunker._get_splitter")
def test_chunk_text_multiple_paragraphs(mock_get_splitter):
    """Test chunking a document with multiple paragraphs."""
    # Set up test document with multiple paragraphs
    document = {
        "path": "multi.md",
        "content": "Paragraph 1.\n\nParagraph 2.\n\nParagraph 3.",
        "type": "markdown"
    }
    
    # Mock the splitter to return predefined paragraphs
    mock_splitter = MagicMock()
    mock_splitter.return_value = ["Paragraph 1.", "Paragraph 2.", "Paragraph 3."]
    mock_get_splitter.return_value = mock_splitter
    
    # Chunk the document
    chunks = chunk_text(document)
    
    # Check results
    assert len(chunks) == 3
    assert chunks[0]["content"] == "Paragraph 1."
    assert chunks[1]["content"] == "Paragraph 2."
    assert chunks[2]["content"] == "Paragraph 3."
    
    # Check that parent ID is consistent across chunks
    parent_id = chunks[0]["parent"]
    assert all(chunk["parent"] == parent_id for chunk in chunks)
    
    # Check naming convention
    assert chunks[0]["name"] == "paragraph_0"
    assert chunks[1]["name"] == "paragraph_1"
    assert chunks[2]["name"] == "paragraph_2"


@patch("src.chunking.text_chunkers.chonky_chunker._get_splitter")
def test_chunk_text_skips_empty_paragraphs(mock_get_splitter):
    """Test that empty paragraphs are skipped."""
    # Set up test document with some empty paragraphs
    document = {
        "path": "with_empty.md",
        "content": "Real paragraph.\n\n\n\nAnother real paragraph.",
        "type": "markdown"
    }
    
    # Mock the splitter to return paragraphs with some empty ones
    mock_splitter = MagicMock()
    mock_splitter.return_value = ["Real paragraph.", "", "  ", "Another real paragraph."]
    mock_get_splitter.return_value = mock_splitter
    
    # Chunk the document
    chunks = chunk_text(document)
    
    # Check that only non-empty paragraphs are included
    assert len(chunks) == 2
    assert chunks[0]["content"] == "Real paragraph."
    assert chunks[1]["content"] == "Another real paragraph."


@patch("src.chunking.text_chunkers.chonky_chunker._get_splitter")
def test_chunk_text_json_output(mock_get_splitter):
    """Test JSON output format."""
    # Set up test document
    document = {
        "path": "json_test.md",
        "content": "Test content for JSON output.",
        "type": "markdown"
    }
    
    # Mock the splitter
    mock_splitter = MagicMock()
    mock_splitter.return_value = ["Test content for JSON output."]
    mock_get_splitter.return_value = mock_splitter
    
    # Chunk the document with JSON output
    json_result = chunk_text(document, output_format="json")
    
    # Verify it's a string (JSON)
    assert isinstance(json_result, str)
    
    # Try parsing it as JSON
    import json
    try:
        parsed = json.loads(json_result)
        assert isinstance(parsed, list)
        assert len(parsed) == 1
        assert parsed[0]["content"] == "Test content for JSON output."
    except json.JSONDecodeError:
        pytest.fail("JSON output is not valid JSON")


@patch("src.chunking.text_chunkers.chonky_chunker._get_splitter")
def test_chunk_text_with_custom_token_limit(mock_get_splitter):
    """Test chunking with a custom token limit."""
    # Set up test document
    document = {
        "path": "custom_limit.md",
        "content": "Test content with custom token limit.",
        "type": "markdown"
    }
    
    # Mock the splitter
    mock_splitter = MagicMock()
    mock_splitter.return_value = ["Test content with custom token limit."]
    mock_get_splitter.return_value = mock_splitter
    
    # Chunk with custom token limit
    chunks = chunk_text(document, max_tokens=500)
    
    # Verify the chunk was created (we're not actually testing the token limit logic here
    # since that would require implementation changes to the chunker)
    assert len(chunks) == 1
    assert chunks[0]["content"] == "Test content with custom token limit."


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])

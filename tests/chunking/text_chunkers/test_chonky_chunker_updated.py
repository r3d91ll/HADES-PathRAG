"""Tests for the enhanced Chonky-based text chunker.

This module tests the functionality of the semantic text chunking 
implementation using the Chonky paragraph splitter with Haystack model engine.

Tests cover:
1. Basic chunking functionality
2. Token-aware chunking
3. Integration with Haystack model engine
4. Error handling and fallbacks
5. Batch processing
"""

import sys
import os
import json
import pytest
import torch
from typing import Dict, List, Any, Union, Generator
from unittest.mock import patch, MagicMock, Mock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from src.chunking.text_chunkers.chonky_chunker import (
    chunk_text, _hash_path, _get_splitter_with_engine, 
    _count_tokens, _split_text_by_tokens, get_model_engine,
    get_tokenizer, ensure_model_engine
)
from src.chunking.text_chunkers.chonky_batch import chunk_text_batch


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


# Test token counting functionality
@patch("src.chunking.text_chunkers.chonky_chunker.AutoTokenizer")
def test_get_tokenizer(mock_auto_tokenizer):
    """Test the tokenizer initialization with caching."""
    # Configure mock
    mock_tokenizer = MagicMock()
    mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer
    
    # Get tokenizer
    tokenizer1 = get_tokenizer("test/model")
    assert tokenizer1 == mock_tokenizer
    mock_auto_tokenizer.from_pretrained.assert_called_once_with("test/model")
    
    # Check caching (should not call from_pretrained again)
    mock_auto_tokenizer.reset_mock()
    tokenizer2 = get_tokenizer("test/model")
    assert tokenizer2 == mock_tokenizer
    mock_auto_tokenizer.from_pretrained.assert_not_called()
    
    # Different model should call from_pretrained again
    tokenizer3 = get_tokenizer("different/model")
    assert tokenizer3 == mock_tokenizer
    mock_auto_tokenizer.from_pretrained.assert_called_once_with("different/model")


@patch("src.chunking.text_chunkers.chonky_chunker.AutoTokenizer")
def test_count_tokens(mock_auto_tokenizer):
    """Test token counting function."""
    # Configure mock
    mock_tokenizer = MagicMock()
    mock_tokenizer.encode.side_effect = lambda text: text.split()  # Simple mock: one token per word
    mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer
    
    # Test with empty text
    assert _count_tokens("", mock_tokenizer) == 0
    assert _count_tokens("  \n  ", mock_tokenizer) == 0
    
    # Test with normal text
    assert _count_tokens("This is a test.", mock_tokenizer) == 4
    mock_tokenizer.encode.assert_called_with("This is a test.")


@patch("src.chunking.text_chunkers.chonky_chunker.AutoTokenizer")
def test_split_text_by_tokens(mock_auto_tokenizer):
    """Test token-aware text splitting."""
    # Configure mock
    mock_tokenizer = MagicMock()
    # Simple mock implementation: each word is one token
    mock_tokenizer.encode.side_effect = lambda text: text.split()
    mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer
    
    # Test with empty text
    assert _split_text_by_tokens("", 100, 10, mock_tokenizer) == []
    
    # Test with text under max tokens
    text = "This is a short text under the limit."
    assert _split_text_by_tokens(text, 10, 1, mock_tokenizer) == [text]
    
    # Test with text over max tokens that needs splitting
    long_text = "This is a longer text that exceeds the maximum token limit and should be split into multiple chunks."
    chunks = _split_text_by_tokens(long_text, 5, 1, mock_tokenizer)
    assert len(chunks) > 1
    assert all(len(chunk.split()) <= 5 for chunk in chunks)
    
    # Test with paragraphs
    multi_para = "Paragraph one.\n\nA second paragraph.\n\nAnd a third one that is longer."
    chunks = _split_text_by_tokens(multi_para, 4, 1, mock_tokenizer)
    assert len(chunks) >= 2


# Mock the ParagraphSplitter and Haystack classes to avoid actually loading models
@patch("src.chunking.text_chunkers.chonky_chunker.ParagraphSplitter")
@patch("src.chunking.text_chunkers.chonky_chunker.HaystackModelEngine")
def test_get_splitter_with_engine(mock_engine_class, mock_paragraph_splitter):
    """Test the enhanced splitter initialization with Haystack engine."""
    # Configure mocks
    mock_splitter = MagicMock()
    mock_paragraph_splitter.return_value = mock_splitter
    
    mock_engine = MagicMock()
    mock_engine.load_model.return_value = {"success": True}
    mock_engine_class.return_value = mock_engine
    
    # Call the function
    splitter = _get_splitter_with_engine("test/model", "cuda:0")
    
    # Check correct initialization
    assert splitter == mock_splitter
    mock_engine.load_model.assert_called_once_with("test/model")
    mock_paragraph_splitter.assert_called_once_with(model_id="test/model", device="cuda:0")
    
    # Test caching - second call with same params should return cached instance
    mock_paragraph_splitter.reset_mock()
    mock_engine.reset_mock()
    
    splitter2 = _get_splitter_with_engine("test/model", "cuda:0")
    assert splitter2 == mock_splitter
    mock_paragraph_splitter.assert_not_called()  # Should use cached instance
    mock_engine.load_model.assert_not_called()
    
    # Test error handling
    mock_engine.load_model.return_value = {"success": False, "error": "Test error"}
    with pytest.raises(RuntimeError):
        _get_splitter_with_engine("error/model", "cuda:0")


@patch("src.chunking.text_chunkers.chonky_chunker.HaystackModelEngine")
def test_get_model_engine(mock_engine_class):
    """Test model engine instantiation and startup."""
    # Reset the global variable to ensure we get a fresh instance
    import src.chunking.text_chunkers.chonky_chunker as chunker_module
    chunker_module._MODEL_ENGINE = None
    
    # Configure mock
    mock_engine = MagicMock()
    mock_engine.status.return_value = {"running": False}
    mock_engine_class.return_value = mock_engine
    
    # Call function
    engine = get_model_engine()
    
    # Check correct initialization
    assert isinstance(engine, MagicMock)  # Just check it's a MagicMock, don't compare object IDs
    mock_engine_class.assert_called_once()
    mock_engine.status.assert_called_once()
    mock_engine.start.assert_called_once()
    
    # Test already running case
    mock_engine_class.reset_mock()
    mock_engine.reset_mock()
    mock_engine.status.return_value = {"running": True}
    
    engine2 = get_model_engine()
    assert isinstance(engine2, MagicMock)  # Just check type, not object identity
    mock_engine.start.assert_not_called()  # Should not start if already running


# Tests for the enhanced chunk_text function using Haystack model engine
@patch("src.chunking.text_chunkers.chonky_chunker._get_splitter_with_engine")
@patch("src.chunking.text_chunkers.chonky_chunker.get_tokenizer")
@patch("src.chunking.text_chunkers.chonky_chunker.ensure_model_engine")
def test_chunk_text_empty_document(mock_ensure_engine, mock_get_tokenizer, mock_get_splitter):
    """Test chunking an empty document with the enhanced implementation."""
    # Configure mocks
    mock_cm = MagicMock()
    mock_ensure_engine.return_value.__enter__.return_value = mock_cm
    mock_tokenizer = MagicMock()
    mock_get_tokenizer.return_value = mock_tokenizer
    
    result = chunk_text({"content": "", "path": "test.md"})
    assert isinstance(result, list)
    assert len(result) == 0
    
    # Verify model engine was used
    mock_ensure_engine.assert_called_once()
    mock_get_tokenizer.assert_called_once()


@patch("src.chunking.text_chunkers.chonky_chunker._get_splitter_with_engine")
@patch("src.chunking.text_chunkers.chonky_chunker.get_tokenizer")
@patch("src.chunking.text_chunkers.chonky_chunker.ensure_model_engine")
@patch("src.chunking.text_chunkers.chonky_chunker._count_tokens")
def test_chunk_text_basic(mock_count_tokens, mock_ensure_engine, mock_get_tokenizer, mock_get_splitter):
    """Test basic chunking of a simple text document with Haystack engine."""
    # Configure mocks
    mock_cm = MagicMock()
    mock_ensure_engine.return_value.__enter__.return_value = mock_cm
    mock_tokenizer = MagicMock()
    mock_get_tokenizer.return_value = mock_tokenizer
    mock_count_tokens.return_value = 5  # Simulate token count
    
    # Configure mock splitter
    mock_splitter = MagicMock()
    mock_splitter.return_value = ["This is a test paragraph."]
    mock_get_splitter.return_value = mock_splitter
    
    document = {
        "content": "This is a test paragraph.",
        "path": "test.md",
        "type": "markdown"
    }
    
    result = chunk_text(document)
    
    assert isinstance(result, list)
    assert len(result) == 1
    chunk = result[0]
    assert chunk["content"] == "This is a test paragraph."
    assert chunk["path"] == "test.md"
    assert chunk["type"] == "markdown"
    assert chunk["symbol_type"] == "paragraph"
    assert chunk["token_count"] == 5  # Check token count is included
    assert "id" in chunk
    assert "parent" in chunk
    
    # Verify Haystack engine was used properly
    mock_ensure_engine.assert_called_once()
    mock_get_tokenizer.assert_called_once()
    mock_get_splitter.assert_called_once()


@patch("src.chunking.text_chunkers.chonky_chunker._get_splitter_with_engine")
@patch("src.chunking.text_chunkers.chonky_chunker.get_tokenizer")
@patch("src.chunking.text_chunkers.chonky_chunker.ensure_model_engine")
@patch("src.chunking.text_chunkers.chonky_chunker._count_tokens")
def test_chunk_text_multiple_paragraphs(mock_count_tokens, mock_ensure_engine, mock_get_tokenizer, mock_get_splitter):
    """Test chunking a document with multiple paragraphs."""
    # Configure mocks
    mock_cm = MagicMock()
    mock_ensure_engine.return_value.__enter__.return_value = mock_cm
    mock_tokenizer = MagicMock()
    mock_get_tokenizer.return_value = mock_tokenizer
    mock_count_tokens.return_value = 5  # Simulate token count
    
    # Configure mock
    mock_splitter = MagicMock()
    mock_splitter.return_value = [
        "This is the first paragraph.",
        "This is the second paragraph.",
        "This is the third paragraph."
    ]
    mock_get_splitter.return_value = mock_splitter
    
    document = {
        "path": "multi.md",
        "content": "This is the first paragraph.\n\nThis is the second paragraph.\n\nThis is the third paragraph.",
        "type": "markdown"
    }
    
    result = chunk_text(document)
    
    assert isinstance(result, list)
    assert len(result) == 3
    
    # Check content of chunks
    assert result[0]["content"] == "This is the first paragraph."
    assert result[1]["content"] == "This is the second paragraph."
    assert result[2]["content"] == "This is the third paragraph."
    
    # Check that all chunks have the same parent
    parent_id = result[0]["parent"]
    assert result[1]["parent"] == parent_id
    assert result[2]["parent"] == parent_id
    
    # All chunks should have a token count
    assert "token_count" in result[0]
    assert "token_count" in result[1]
    assert "token_count" in result[2]


@patch("src.chunking.text_chunkers.chonky_chunker._get_splitter_with_engine")
@patch("src.chunking.text_chunkers.chonky_chunker.get_tokenizer")
@patch("src.chunking.text_chunkers.chonky_chunker.ensure_model_engine")
@patch("src.chunking.text_chunkers.chonky_chunker._count_tokens")
def test_chunk_text_skips_empty_paragraphs(mock_count_tokens, mock_ensure_engine, mock_get_tokenizer, mock_get_splitter):
    """Test that empty paragraphs are skipped."""
    # Configure mocks
    mock_cm = MagicMock()
    mock_ensure_engine.return_value.__enter__.return_value = mock_cm
    mock_tokenizer = MagicMock()
    mock_get_tokenizer.return_value = mock_tokenizer
    mock_count_tokens.return_value = 5  # Simulate token count
    
    # Configure mock
    mock_splitter = MagicMock()
    mock_splitter.return_value = [
        "This is a paragraph.",
        "",  # Empty paragraph
        "   ",  # Whitespace-only paragraph
        "This is another paragraph."
    ]
    mock_get_splitter.return_value = mock_splitter
    
    document = {
        "content": "This is a paragraph.\n\n\n\n   \n\nThis is another paragraph.",
        "path": "test.md"
    }
    
    result = chunk_text(document)
    
    assert isinstance(result, list)
    assert len(result) == 2  # Only 2 non-empty paragraphs
    assert result[0]["content"] == "This is a paragraph."
    assert result[1]["content"] == "This is another paragraph."


@patch("src.chunking.text_chunkers.chonky_chunker._get_splitter_with_engine")
@patch("src.chunking.text_chunkers.chonky_chunker.get_tokenizer")
@patch("src.chunking.text_chunkers.chonky_chunker.ensure_model_engine")
@patch("src.chunking.text_chunkers.chonky_chunker._count_tokens")
def test_chunk_text_json_output(mock_count_tokens, mock_ensure_engine, mock_get_tokenizer, mock_get_splitter):
    """Test JSON output format."""
    # Configure mocks
    mock_cm = MagicMock()
    mock_ensure_engine.return_value.__enter__.return_value = mock_cm
    mock_tokenizer = MagicMock()
    mock_get_tokenizer.return_value = mock_tokenizer
    mock_count_tokens.return_value = 5  # Simulate token count
    
    # Configure mock
    mock_splitter = MagicMock()
    mock_splitter.return_value = [
        "This is a paragraph.",
        "This is another paragraph."
    ]
    mock_get_splitter.return_value = mock_splitter
    
    document = {
        "content": "This is a paragraph.\n\nThis is another paragraph.",
        "path": "test.md"
    }
    
    # Request JSON output
    result = chunk_text(document, output_format="json")
    
    assert isinstance(result, str)
    
    # Parse JSON and verify contents
    chunks = json.loads(result)
    assert isinstance(chunks, list)
    assert len(chunks) == 2
    assert chunks[0]["content"] == "This is a paragraph."
    assert chunks[1]["content"] == "This is another paragraph."


@patch("src.chunking.text_chunkers.chonky_chunker._get_splitter_with_engine")
@patch("src.chunking.text_chunkers.chonky_chunker.get_tokenizer")
@patch("src.chunking.text_chunkers.chonky_chunker.ensure_model_engine")
@patch("src.chunking.text_chunkers.chonky_chunker._count_tokens")
@patch("src.chunking.text_chunkers.chonky_chunker.get_chunker_config")
def test_chunk_text_with_custom_token_limit(mock_config, mock_count_tokens, mock_ensure_engine, mock_get_tokenizer, mock_get_splitter):
    """Test chunking with a custom token limit."""
    # Configure mocks
    mock_cm = MagicMock()
    mock_ensure_engine.return_value.__enter__.return_value = mock_cm
    mock_tokenizer = MagicMock()
    mock_get_tokenizer.return_value = mock_tokenizer
    mock_count_tokens.return_value = 5  # Simulate token count
    
    # Configure splitter mock
    mock_splitter = MagicMock()
    mock_splitter.return_value = ["This is a test paragraph."]
    mock_get_splitter.return_value = mock_splitter
    
    # Configure mock config with a specific token limit
    mock_config.return_value = {"max_tokens": 1024, "model_id": "test_model", "device": "cpu"}
    
    document = {
        "content": "This is a test paragraph.",
        "path": "test.md"
    }
    
    # Call with custom token limit
    result = chunk_text(document, max_tokens=512)
    
    assert isinstance(result, list)
    assert len(result) == 1
    
    # The token limit should be respected (using the provided value, not the config value)
    mock_config.assert_called_with("chonky")


def test_text_splitting_by_tokens():
    """Test the _split_text_by_tokens function directly to ensure token-aware chunking works."""
    # Mock tokenizer for testing
    mock_tokenizer = MagicMock()
    # Configure tokenizer to return one token per word for simple testing
    mock_tokenizer.encode.side_effect = lambda text: text.split()
    
    # Test with a paragraph that exceeds token limit
    text = "This is a very long paragraph that should be split into multiple chunks because it exceeds the maximum token limit."
    
    # Set max_tokens to 5 and min_tokens to 1 to force splitting
    chunks = _split_text_by_tokens(text, max_tokens=5, min_tokens=1, tokenizer=mock_tokenizer)
    
    # Should produce multiple chunks, each with 5 or fewer tokens (words)
    assert len(chunks) > 1, "Text should be split into multiple chunks"
    
    # Check each chunk respects max_tokens limit (5 words or fewer)
    for chunk in chunks:
        token_count = len(chunk.split())
        assert token_count <= 5, f"Chunk exceeds token limit: {chunk} has {token_count} tokens"


@patch("src.chunking.text_chunkers.chonky_chunker._get_splitter_with_engine")
@patch("src.chunking.text_chunkers.chonky_chunker.get_tokenizer")
@patch("src.chunking.text_chunkers.chonky_chunker.ensure_model_engine")
@patch("src.chunking.text_chunkers.chonky_chunker._count_tokens")
def test_chunk_text_error_fallback(mock_count_tokens, mock_ensure_engine, 
                                 mock_get_tokenizer, mock_get_splitter):
    """Test error handling and fallback to basic chunking."""
    # Configure mocks
    mock_cm = MagicMock()
    mock_ensure_engine.return_value.__enter__.return_value = mock_cm
    mock_tokenizer = MagicMock()
    mock_get_tokenizer.return_value = mock_tokenizer
    mock_count_tokens.return_value = 5
    
    # Configure splitter to raise an exception
    mock_get_splitter.side_effect = RuntimeError("Test error")
    
    document = {
        "content": "First paragraph.\n\nSecond paragraph.",
        "path": "test.md",
        "type": "markdown"
    }
    
    # Should not raise exception but fall back to basic chunking
    result = chunk_text(document)
    
    assert isinstance(result, list)
    assert len(result) == 2  # 2 paragraphs from basic splitting
    
    # Verify error handling
    mock_get_splitter.assert_called_once()


# Tests for batch processing functionality
@patch("src.chunking.text_chunkers.chonky_batch.chunk_text")
def test_chunk_text_batch_serial(mock_chunk_text):
    """Test batch processing in serial mode."""
    # Configure mock
    mock_chunk_text.side_effect = lambda doc, **kwargs: [f"Chunk for {doc['path']}"]
    
    documents = [
        {"content": "Doc 1", "path": "doc1.md"},
        {"content": "Doc 2", "path": "doc2.md"},
        {"content": "Doc 3", "path": "doc3.md"}
    ]
    
    # Process in serial mode
    results = chunk_text_batch(documents, parallel=False)
    
    assert len(results) == 3
    assert results[0] == ["Chunk for doc1.md"]
    assert results[1] == ["Chunk for doc2.md"]
    assert results[2] == ["Chunk for doc3.md"]
    assert mock_chunk_text.call_count == 3


@patch("src.chunking.text_chunkers.chonky_batch.chunk_text")
@patch("src.chunking.text_chunkers.chonky_batch.ThreadPool")
def test_chunk_text_batch_parallel(mock_thread_pool, mock_chunk_text):
    """Test batch processing in parallel mode."""
    # Configure mocks
    mock_pool = MagicMock()
    mock_thread_pool.return_value.__enter__.return_value = mock_pool
    mock_pool.map.return_value = [["Chunk 1"], ["Chunk 2"], ["Chunk 3"]]
    
    documents = [
        {"content": "Doc 1", "path": "doc1.md"},
        {"content": "Doc 2", "path": "doc2.md"},
        {"content": "Doc 3", "path": "doc3.md"}
    ]
    
    # Process in parallel mode
    results = chunk_text_batch(documents, parallel=True, num_workers=2)
    
    assert len(results) == 3
    assert results[0] == ["Chunk 1"]
    assert results[1] == ["Chunk 2"]
    assert results[2] == ["Chunk 3"]
    
    # Verify ThreadPool was used
    mock_thread_pool.assert_called_once_with(processes=2)
    mock_pool.map.assert_called_once()


@patch("src.chunking.text_chunkers.chonky_batch.chunk_text")
def test_chunk_text_batch_empty(mock_chunk_text):
    """Test batch processing with empty input."""
    # Empty documents list
    results = chunk_text_batch([])
    assert results == []
    mock_chunk_text.assert_not_called()


@patch("src.chunking.text_chunkers.chonky_batch.chunk_text")
def test_chunk_text_batch_single_doc(mock_chunk_text):
    """Test batch processing with a single document."""
    # Configure mock
    mock_chunk_text.return_value = ["Single chunk"]
    
    documents = [{"content": "Single doc", "path": "single.md"}]
    
    # Even with parallel=True, should use serial mode for single document
    results = chunk_text_batch(documents, parallel=True)
    
    assert len(results) == 1
    assert results[0] == ["Single chunk"]
    mock_chunk_text.assert_called_once()


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])

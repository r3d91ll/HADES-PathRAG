#!/usr/bin/env python
"""Debug script to trace memory usage in chunky_batch.py.

This script is designed to identify memory leaks or excessive memory usage
in the chunk_document_batch function, especially when saving to disk.
"""

import sys
import os
import tempfile
import gc
import tracemalloc
from unittest.mock import patch, MagicMock

# Mock torch and transformers to avoid import errors
sys.modules['torch'] = MagicMock()
sys.modules['transformers'] = MagicMock()

# Now import the function to debug
from src.chunking.text_chunkers.chonky_batch import chunk_document_batch


def debug_memory_usage(func_name, func, *args, **kwargs):
    """Run a function and track peak memory usage."""
    gc.collect()  # Force garbage collection before starting
    
    # Start tracing memory allocations
    tracemalloc.start()
    
    # Run the function with tracking
    try:
        result = func(*args, **kwargs)
        current, peak = tracemalloc.get_traced_memory()
        print(f"{func_name} - Current memory usage: {current / 10**6:.2f} MB")
        print(f"{func_name} - Peak memory usage: {peak / 10**6:.2f} MB")
        
        # Display top 10 memory allocations
        print("\nTop 10 memory allocations:")
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics('lineno')
        for i, stat in enumerate(top_stats[:10]):
            print(f"#{i}: {stat}")
            
        return result
    finally:
        tracemalloc.stop()


def test_chunk_document_batch_basic():
    """Test basic functionality without save_to_disk."""
    print("\n--- Testing chunk_document_batch (basic) ---")
    
    # Create a test document
    doc = {"id": "test-doc", "content": "test content"}
    
    # Mock chunk_document
    with patch('src.chunking.text_chunkers.chonky_batch.chunk_document') as mock_chunk, \
         patch('src.chunking.text_chunkers.chonky_batch.DocumentSchema.model_validate') as mock_validate:
        
        mock_chunk.return_value = {"id": "test-doc", "content": "test content", "chunks": []}
        mock_validate.return_value = MagicMock()
        
        # Test with basic configuration
        debug_memory_usage(
            "chunk_document_batch (basic)",
            chunk_document_batch,
            [doc],
            return_pydantic=True,
            save_to_disk=False
        )


def test_chunk_document_batch_save_to_disk():
    """Test with save_to_disk=True to identify memory issues."""
    print("\n--- Testing chunk_document_batch (save_to_disk) ---")
    
    # Create a test document
    doc = {"id": "test-doc", "content": "test content"}
    
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as tempdir:
        # Mock dependencies
        with patch('src.chunking.text_chunkers.chonky_batch.chunk_document') as mock_chunk, \
             patch('src.chunking.text_chunkers.chonky_batch.DocumentSchema.model_validate') as mock_validate, \
             patch('builtins.open') as mock_open:
            
            mock_chunk.return_value = {"id": "test-doc", "content": "test content", "chunks": []}
            mock_validate.return_value = MagicMock()
            mock_file = MagicMock()
            mock_open.return_value.__enter__.return_value = mock_file
            
            # Test with save_to_disk=True
            debug_memory_usage(
                "chunk_document_batch (save_to_disk)",
                chunk_document_batch,
                [doc],
                save_to_disk=True,
                output_dir=tempdir
            )


if __name__ == "__main__":
    test_chunk_document_batch_basic()
    test_chunk_document_batch_save_to_disk()

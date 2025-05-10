"""Batch processing utilities for Chonky chunker.

This module provides batch processing functionality for the Chonky chunker,
enabling parallel processing of multiple documents and improving throughput.
"""

from typing import Any, Dict, List, Optional, Union
import logging
from multiprocessing.pool import ThreadPool

from .chonky_chunker import chunk_text

logger = logging.getLogger(__name__)


def chunk_text_batch(
    documents: List[Dict[str, Any]], *, 
    max_tokens: int = 2048, 
    output_format: str = "python",
    parallel: bool = True,
    num_workers: int = 4
) -> Union[List[List[Dict[str, Any]]], List[str]]:
    """Process multiple documents in parallel using Chonky chunking.
    
    Parameters
    ----------
    documents:
        List of pre-processed document dicts, each containing content and path keys
    max_tokens:
        Token budget for each chunk
    output_format:
        Output format ("python" or "json")
    parallel:
        Whether to process in parallel using a thread pool
    num_workers:
        Number of worker threads when parallel=True
        
    Returns
    -------
    Union[List[List[Dict[str, Any]]], List[str]]
        List of chunk results for each document
    """
    # Configure chunking function with proper typing based on output format
    if output_format == "json":
        def process_doc(doc: Dict[str, Any]) -> str:
            result = chunk_text(doc, max_tokens=max_tokens, output_format=output_format)
            assert isinstance(result, str), "Expected string output for JSON format"
            return result
    else:
        def process_doc(doc: Dict[str, Any]) -> List[Dict[str, Any]]:
            result = chunk_text(doc, max_tokens=max_tokens, output_format=output_format)
            assert isinstance(result, list), "Expected list output for Python format"
            return result
    
    if not documents:
        # Return empty list with appropriate type based on output_format
        return [] if output_format == "python" else []  # The type checker will handle this
    
    # Process documents (either serially or in parallel)
    if len(documents) == 1 or not parallel:
        results = [process_doc(doc) for doc in documents]
    else:
        # For multiple documents, use thread pool
        with ThreadPool(processes=min(num_workers, len(documents))) as pool:
            results = pool.map(process_doc, documents)
            results = list(results)
    
    # Return the results directly - they should already be correctly typed due to our process_doc function
    if output_format == "json":
        # Type checker needs explicit casting to know these are all strings
        return results  # type: ignore
    else:
        # For python format, results are List[List[Dict[str, Any]]]
        return results  # type: ignore

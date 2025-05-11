"""Batch processing utilities for Chonky chunker.

This module provides batch processing functionality for the Chonky chunker,
enabling parallel processing of multiple documents and improving throughput.
"""

from typing import Any, Dict, List, Optional, Union, cast
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
    
    This function processes multiple documents in batch, either serially or in parallel,
    using the Chonky neural chunking model. It preserves the original text casing and
    formatting while identifying natural paragraph and section boundaries.
    
    The batch processing can significantly improve throughput when processing many documents,
    especially when using parallel processing on multi-core systems. The function uses a
    thread pool to distribute the workload across multiple workers when parallel=True.
    
    Configuration is loaded from src/config/chunker_config.yaml and the batch_size parameter
    controls how many documents are processed at once in parallel mode.
    
    Args:
        documents: List of document dictionaries, each with the following keys:
            - content: The text content of the document
            - path: Path to the document (used for ID generation)
            - type: Document type (e.g., "markdown", "text")
        max_tokens: Maximum tokens per chunk (default: 2048)
        output_format: Output format, either "python" for Python objects or "json" for
                      JSON string (default: "python")
        parallel: Whether to process documents in parallel using a thread pool
                 (default: True)
        num_workers: Number of worker threads when parallel=True (default: 4)
        
    Returns:
        If output_format is "python": List of lists of chunk dictionaries, where each inner
        list contains the chunks for the corresponding document in the input list.
        If output_format is "json": List of JSON strings, one for each document.
        
    Raises:
        ValueError: If any document is missing required fields or has invalid content
        RuntimeError: If there are issues with the model engine (will fall back to basic chunking)
    """
    
    # Define a generic process function that returns the appropriate type
    def process_doc(doc: Dict[str, Any]) -> Union[str, List[Dict[str, Any]]]:
        result = chunk_text(doc, max_tokens=max_tokens, output_format=output_format)
        if output_format == "json":
            assert isinstance(result, str), "Expected string output for JSON format"
            return result
        else:
            assert isinstance(result, list), "Expected list output for Python format"
            return cast(List[Dict[str, Any]], result)
    
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
    
    # Return the results with proper type annotation
    if output_format == "json":
        # For JSON format, results are List[str]
        return cast(List[str], results)
    else:
        # For python format, results are List[List[Dict[str, Any]]]
        return cast(List[List[Dict[str, Any]]], results)

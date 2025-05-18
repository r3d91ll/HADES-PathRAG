#!/bin/bash

cat > /home/todd/ML-Lab/Olympus/HADES-PathRAG/src/chunking/text_chunkers/chonky_batch.py << 'EOF'
"""Batch processing utilities for Chonky chunker.

This module provides batch processing functionality for the Chonky chunker,
enabling parallel processing of multiple documents and improving throughput.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import os
import json
import uuid
from datetime import datetime
from multiprocessing.pool import ThreadPool
from typing import Any, Dict, List, Optional, Union, Type, TypeVar, Sequence, Tuple, cast, Callable
from typing_extensions import Literal

# Import tqdm for progress reporting, handling the case where it's not installed
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    # Define a simple fallback if tqdm is not available
    def tqdm(
        iterable: Any = None, 
        *args: Any, 
        **kwargs: Any
    ) -> Any:
        """No-op tqdm if not available"""
        return iterable if iterable is not None else (lambda x: x)
    TQDM_AVAILABLE = False
    
from .chonky_chunker import chunk_text, chunk_document, BaseDocument, DocumentSchema
from ...schema.document_schema import DocumentType, SchemaVersion

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
    if not documents:
        return []
    
    def process_doc_python(doc: Dict[str, Any]) -> List[Dict[str, Any]]:
        try:
            # Extract required fields from the document
            content = doc.get("content", "")
            doc_id = doc.get("id", None)
            path = doc.get("path", doc.get("source", "unknown"))
            doc_type = doc.get("type", doc.get("document_type", "text"))
            
            # Pass extracted fields to chunk_text with Python output format
            result: Any = chunk_text(
                content=content, 
                doc_id=doc_id,
                path=path,
                doc_type=doc_type,
                max_tokens=max_tokens, 
                output_format="python"
            )
            
            # Create a properly typed result with default empty list
            dict_results: List[Dict[str, Any]] = []
            
            # Handle each possible result type
            if result is None:
                return dict_results
                
            # Case: Result is a list
            if isinstance(result, list):
                dict_results = [
                    chunk for chunk in result 
                    if isinstance(chunk, dict)
                ]
                
            # Case: Result is a dict with a 'chunks' field
            elif isinstance(result, dict) and 'chunks' in result:
                chunks = result.get('chunks', [])
                if isinstance(chunks, list):
                    dict_results = [
                        chunk for chunk in chunks 
                        if isinstance(chunk, dict)
                    ]
            
            return dict_results
            
        except Exception as e:
            logger.error(f"Error processing document: {e}")
            return []
    
    def process_doc_json(doc: Dict[str, Any]) -> str:
        try:
            # Extract required fields from the document
            content = doc.get("content", "")
            doc_id = doc.get("id", None)
            path = doc.get("path", doc.get("source", "unknown"))
            doc_type = doc.get("type", doc.get("document_type", "text"))
            
            # Pass extracted fields to chunk_text with JSON output format
            result: Any = chunk_text(
                content=content, 
                doc_id=doc_id,
                path=path,
                doc_type=doc_type,
                max_tokens=max_tokens, 
                output_format="json"
            )
            
            # Handle possible result types
            if result is None:
                return "[]"
                
            if isinstance(result, str):
                return result
                
            # Fallback if the result is not a string (should not happen)
            return json.dumps([])
            
        except Exception as e:
            logger.error(f"Error processing document: {e}")
            return "[]"
    
    # Process for output format "python" or "json"
    if output_format == "json":
        # Process the documents with JSON output format
        if not parallel or len(documents) == 1:
            # Process serially using the JSON processor
            json_results = [process_doc_json(doc) for doc in documents]
        else:
            # For multiple documents, use thread pool with the JSON processor
            with ThreadPool(processes=min(num_workers, len(documents))) as pool:
                json_results = list(pool.map(process_doc_json, documents))
                
        return json_results
    else:
        # Process with Python output format (default)
        if not parallel or len(documents) == 1:
            # Process serially using the Python processor
            python_results = [process_doc_python(doc) for doc in documents]
        else:
            with ThreadPool(processes=min(num_workers, len(documents))) as pool:
                python_results = list(pool.map(process_doc_python, documents))
        
        return python_results


def process_document_to_dict(
    document: Union[Dict[str, Any], BaseDocument, DocumentSchema],
    max_tokens: int = 2048
) -> Dict[str, Any]:
    """Process a document to a dictionary with chunks.
    
    This function processes a single document, which can be a dictionary, a BaseDocument,
    or a DocumentSchema object, and returns a dictionary with the original document fields
    plus added chunks.
    
    Args:
        document: The document to process, can be a dictionary or object
        max_tokens: Maximum tokens per chunk
        
    Returns:
        Dictionary representation of the document with chunks
        
    Raises:
        ValueError: If the document type is unsupported
    """
    try:
        # Handle different document types
        if hasattr(document, 'dict') and callable(getattr(document, 'dict')):
            # Handle Pydantic models
            doc_dict = document.dict()
        elif isinstance(document, dict):
            # Handle plain dictionaries
            doc_dict = document
        else:
            # Try to convert to dict if possible, or raise error
            try:
                doc_dict = dict(document)
            except (TypeError, ValueError):
                raise ValueError(f"Unsupported document type: {type(document)}")
        
        # Extract content and check if it's empty
        content = str(doc_dict.get('content', ''))
        if not content.strip():
            logger.warning("Document has no content, returning original")
            return doc_dict
        
        # Process with chunk_text
        doc_id = doc_dict.get('id')
        path = str(doc_dict.get('path', doc_dict.get('source', 'unknown')))
        doc_type = str(doc_dict.get('type', doc_dict.get('document_type', 'text')))
        
        result = chunk_text(
            content=content,
            doc_id=doc_id,
            path=path,
            doc_type=doc_type,
            max_tokens=max_tokens,
            output_format="python"
        )
        
        # Update the document with chunks
        if isinstance(result, dict) and 'chunks' in result:
            # When chunk_text returns a document with chunks
            chunks = result.get('chunks', [])
            doc_dict['chunks'] = chunks
            
            # Copy any other fields that might have been added
            for key, value in result.items():
                if key != 'chunks' and key not in doc_dict:
                    doc_dict[key] = value
        elif isinstance(result, list):
            # When chunk_text returns just a list of chunks
            doc_dict['chunks'] = result
        
        return doc_dict
        
    except Exception as e:
        logger.error(f"Error processing document: {e}")
        return document if isinstance(document, dict) else {}


def chunk_document_batch(
    documents: List[Union[Dict[str, Any], BaseDocument, DocumentSchema]], *, 
    max_tokens: int = 2048,
    return_pydantic: bool = True,
    save_to_disk: bool = False,
    output_dir: Optional[str] = None,
    parallel: bool = True,
    num_workers: int = 4
) -> List[Union[Dict[str, Any], DocumentSchema]]:
    """Process multiple documents in parallel using Chonky chunking and update with chunks.
    
    This function processes multiple documents in batch, either serially or in parallel,
    using the Chonky neural chunking model. Each document is updated with its generated chunks.
    
    Args:
        documents: List of documents (can be dictionaries, BaseDocument, or DocumentSchema)
        max_tokens: Maximum tokens per chunk (default: 2048)
        return_pydantic: Whether to return DocumentSchema objects (default: True)
        save_to_disk: Whether to save processed documents to disk (default: False)
        output_dir: Directory to save chunked documents (if save_to_disk is True)
        parallel: Whether to process documents in parallel using a thread pool
                 (default: True)
        num_workers: Number of worker threads when parallel=True (default: 4)
        
    Returns:
        List of updated documents with chunks attached
        
    Raises:
        ValueError: If any document is missing required fields or has invalid content
        RuntimeError: If there are issues with the model engine (will fall back to basic chunking)
    """
    if not documents:
        return []
    
    # Prepare output directory if saving to disk
    if save_to_disk and output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    def process_single(doc: Union[Dict[str, Any], BaseDocument, DocumentSchema]) -> Optional[Union[Dict[str, Any], DocumentSchema]]:
        try:
            # Process the document
            processed_doc = process_document_to_dict(doc, max_tokens=max_tokens)
            
            # Save to disk if requested
            if save_to_disk and output_dir:
                try:
                    # Generate filename based on document ID or UUID
                    doc_id = processed_doc.get('id', str(uuid.uuid4()))
                    filename = f"{doc_id}.json"
                    filepath = os.path.join(output_dir, filename)
                    
                    # Write to file
                    with open(filepath, 'w', encoding='utf-8') as f:
                        json.dump(processed_doc, f, ensure_ascii=False, indent=2)
                        
                    logger.info(f"Saved document to {filepath}")
                except Exception as e:
                    logger.error(f"Error saving document to disk: {e}")
            
            # Convert to DocumentSchema if requested
            if return_pydantic:
                try:
                    return DocumentSchema(**processed_doc)
                except Exception as e:
                    logger.error(f"Error converting to DocumentSchema: {e}")
                    return processed_doc
            else:
                return processed_doc
                
        except Exception as e:
            logger.error(f"Error processing document: {e}")
            return None
    
    # Process documents (either serially or in parallel)
    results: List[Union[Dict[str, Any], DocumentSchema]] = []
    
    if len(documents) == 1 or not parallel:
        # Process serially
        for doc in documents:
            result = process_single(doc)
            if result is not None:
                results.append(result)
    else:
        # For multiple documents, use thread pool
        with ThreadPoolExecutor(max_workers=min(num_workers, len(documents))) as executor:
            # Submit all documents for processing
            future_to_doc = {
                executor.submit(process_single, doc): doc 
                for doc in documents
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_doc):
                try:
                    result = future.result()
                    if result is not None:
                        results.append(result)
                except Exception as e:
                    logger.error(f"Error processing document: {e}")
    
    return results


def process_documents_batch(
    documents: List[Union[Dict[str, Any], BaseDocument, DocumentSchema]],
    max_tokens: int = 2048,
    return_pydantic: bool = False,
    parallel: bool = True,
    num_workers: int = 4
) -> List[Dict[str, Any]]:
    """Process a batch of documents and add chunks.
    
    Args:
        documents: List of documents (can be various types)
        max_tokens: Maximum tokens per chunk
        return_pydantic: Whether to return DocumentSchema objects
        parallel: Whether to process in parallel
        num_workers: Number of worker threads for parallel processing
        
    Returns:
        List of processed document dictionaries with chunks
    """
    # Convert all documents to dictionaries for consistent processing
    doc_dicts: List[Dict[str, Any]] = []
    
    for doc in documents:
        if hasattr(doc, 'dict') and callable(getattr(doc, 'dict')):
            # Handle Pydantic models
            doc_dicts.append(doc.dict())
        elif isinstance(doc, dict):
            # Handle dictionaries
            doc_dicts.append(doc)
        else:
            # Try to convert to dictionary or skip
            try:
                doc_dicts.append(dict(doc))
            except (TypeError, ValueError):
                logger.warning(f"Skipping document of unsupported type: {type(doc)}")
    
    # Define processor for single document
    def process_single_doc(doc_dict: Dict[str, Any]) -> Dict[str, Any]:
        try:
            processed = process_document_to_dict(doc_dict, max_tokens=max_tokens)
            if isinstance(processed, dict):
                return processed
            # Handle the case where a DocumentSchema is returned
            if hasattr(processed, 'dict') and callable(getattr(processed, 'dict')):
                return processed.dict()
            # Fallback
            return doc_dict
        except Exception as e:
            logger.error(f"Error in batch processing: {e}")
            return doc_dict
    
    # Process serially or in parallel
    processed_docs: List[Dict[str, Any]] = []
    
    if not parallel or len(doc_dicts) == 1:
        # Process serially
        processed_docs = [process_single_doc(doc) for doc in doc_dicts]
    else:
        # Process in parallel
        with ThreadPool(processes=min(num_workers, len(doc_dicts))) as pool:
            processed_docs = list(pool.map(process_single_doc, doc_dicts))
    
    return processed_docs


def chunk_documents(
    documents: List[Any],
    batch_size: int = 10,
    max_tokens: int = 2048,
    progress: bool = False,
    num_workers: int = 4
) -> List[Dict[str, Any]]:
    """Process multiple documents in batches with optional progress reporting.
    
    This function processes documents in batches for better memory management,
    and optionally displays a progress bar using tqdm.
    
    Args:
        documents: List of documents (can be various types)
        batch_size: Size of each processing batch
        max_tokens: Maximum tokens per chunk
        progress: Whether to show progress bar
        num_workers: Number of worker threads per batch
        
    Returns:
        List of processed document dictionaries with chunks
    """
    if not documents:
        return []
    
    # Process in batches for better memory management
    results: List[Dict[str, Any]] = []
    
    # Create batches
    batches = [documents[i:i+batch_size] for i in range(0, len(documents), batch_size)]
    
    # Set up iteration with or without progress bar
    if progress and TQDM_AVAILABLE:
        batch_iter = tqdm(batches, desc="Processing documents", unit="batch")
    else:
        batch_iter = batches
    
    # Process each batch
    for batch in batch_iter:
        batch_results = process_documents_batch(
            batch,
            max_tokens=max_tokens,
            parallel=(num_workers > 1),
            num_workers=num_workers
        )
        results.extend(batch_results)
    
    return results
EOF

chmod +x /home/todd/ML-Lab/Olympus/HADES-PathRAG/fix_chonky_batch.sh

"""
Core functionality for document processing.

This module provides the primary interface for processing documents of various formats
and transforming them into standardized JSON objects. These objects can be directly
passed to the next stage in the processing pipeline (e.g., chunking, embedding) or
optionally saved to disk for inspection or archiving.

The focus is on in-memory processing for optimal performance in pipeline scenarios.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Callable, cast, TypeVar, Type

from pydantic import ValidationError, BaseModel

from .utils.format_detector import detect_format_from_path
from .utils.metadata_extractor import extract_metadata
from .adapters.registry import get_adapter_for_format
from .schemas.utils import validate_document, add_validation_to_adapter
from .schemas.base import BaseDocument

# Setup logging
logger = logging.getLogger(__name__)

# Type variable for return type flexibility
T = TypeVar('T', bound=BaseModel)


def process_document(
    file_path: Union[str, Path], 
    options: Optional[Dict[str, Any]] = None
) -> Union[BaseDocument, Dict[str, Any]]:
    """
    Process a document file, converting it to a standardized format.
    
    Args:
        file_path: Path to the document file
        options: Optional processing options
    
    Returns:
        Pydantic model instance (BaseDocument or subclass) or dictionary with processed content
        
    Raises:
        FileNotFoundError: If the file does not exist
        ValidationError: If the processed document fails validation
    """
    options = options or {}
    path_obj = Path(file_path) if isinstance(file_path, str) else file_path
    
    # Check if file exists first
    if not path_obj.exists():
        raise FileNotFoundError(f"File not found: {path_obj}")
    
    # Detect the document format
    format_type = detect_format_from_path(path_obj)
    
    # Get the appropriate adapter
    adapter = get_adapter_for_format(format_type)
    
    # Process the document
    processed_doc = adapter.process(path_obj, options)
    
    # Enrich with metadata using our heuristic extraction
    content = processed_doc.get("content", "")
    metadata = extract_metadata(content, str(path_obj), format_type)
    
    # Merge extracted metadata with any existing metadata
    existing_metadata = processed_doc.get("metadata", {})
    for key, value in metadata.items():
        if key not in existing_metadata or existing_metadata[key] == "UNK":
            existing_metadata[key] = value
    
    # Update the processed document with the enriched metadata
    processed_doc["metadata"] = existing_metadata
    
    # Validate the processed document using Pydantic
    try:
        validated_doc = validate_document(processed_doc)
        # Return the Pydantic model directly to maintain type safety
        # Only convert to dict if explicitly requested via options
        if options.get("return_dict", False):
            return validated_doc.model_dump()
        return validated_doc
    except ValidationError as e:
        logger.warning(f"Document validation failed: {e}")
        # Add validation error to the document
        processed_doc["_validation_error"] = str(e)
        return processed_doc


def process_text(
    text: str, 
    format_type: str = "text", 
    format_or_options: Optional[Union[str, Dict[str, Any]]] = None, 
    options: Optional[Dict[str, Any]] = None
) -> Union[BaseDocument, Dict[str, Any]]:
    """
    Process text content directly, assuming a specific format.
    
    Args:
        text: Text content to process
        format_type: Format of the text (e.g., "markdown", "html", "json")
        format_or_options: Either a format string or options dictionary (for flexibility)
        options: Optional processing options
    
    Returns:
        Pydantic model instance or dictionary with processed content
        
    Raises:
        ValidationError: If the processed document fails validation
    """
    # Handle flexible parameter usage
    if isinstance(format_or_options, dict):
        options = format_or_options
    elif isinstance(format_or_options, str):
        format_type = format_or_options
    
    options = options or {}
    
    # Get the appropriate adapter
    adapter = get_adapter_for_format(format_type)
    
    # Create a temporary file for processing if needed
    if hasattr(adapter, "process_text"):
        # Use direct text processing if available
        processed_doc = adapter.process_text(text, options)
    else:
        # Fall back to file-based processing
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=f".{format_type}", mode="w", encoding="utf-8", delete=False) as tmp:
            tmp.write(text)
            tmp_path = Path(tmp.name)
        
        try:
            processed_doc = adapter.process(tmp_path, options)
        finally:
            # Clean up temporary file
            if tmp_path.exists():
                tmp_path.unlink()
    
    # Enrich with metadata
    metadata = extract_metadata(text, f"text_{format_type}", format_type)
    
    # Merge extracted metadata with any existing metadata
    existing_metadata = processed_doc.get("metadata", {})
    for key, value in metadata.items():
        if key not in existing_metadata or existing_metadata[key] == "UNK":
            existing_metadata[key] = value
    
    # Update the processed document with the enriched metadata
    processed_doc["metadata"] = existing_metadata
    
    # Validate the processed document
    try:
        validated_doc = validate_document(processed_doc)
        # Return the Pydantic model directly to maintain type safety
        # Only convert to dict if explicitly requested via options
        if options.get("return_dict", False):
            return validated_doc.model_dump()
        return validated_doc
    except ValidationError as e:
        logger.warning(f"Document validation failed: {e}")
        # Add validation error to the document
        processed_doc["_validation_error"] = str(e)
        return processed_doc


def batch_process_documents(
    file_paths: List[Union[str, Path]], 
    options: Optional[Dict[str, Any]] = None,
    parallel: bool = True,
    max_workers: Optional[int] = None
) -> List[Union[BaseDocument, Dict[str, Any]]]:
    """
    Process multiple documents in parallel.
    
    Args:
        file_paths: List of paths to document files
        options: Optional processing options
        parallel: Whether to process documents in parallel
        max_workers: Maximum number of worker threads (None = auto)
    
    Returns:
        List of processed documents as Pydantic models or dictionaries
    """
    options = options or {}
    
    if not parallel:
        # Process sequentially
        return [process_document(path, options) for path in file_paths]
    
    # Process in parallel
    from concurrent.futures import ThreadPoolExecutor
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(
            lambda path: process_document(path, options), 
            file_paths
        ))
    
    return results


def save_processed_document(
    document: Union[BaseDocument, Dict[str, Any]], 
    output_path: Union[str, Path],
    pretty: bool = True
) -> Path:
    """
    Save a processed document to disk.
    
    This is primarily for debugging and inspection purposes, not part of the main
    processing pipeline.
    
    Args:
        document: Processed document (Pydantic model or dictionary)
        output_path: Path to save the document
        pretty: Whether to format the JSON with indentation
    
    Returns:
        Path to the saved document
    """
    path_obj = Path(output_path) if isinstance(output_path, str) else output_path
    
    # Ensure parent directory exists
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to dictionary if it's a Pydantic model
    if isinstance(document, BaseModel):
        document_dict = document.model_dump()
    else:
        document_dict = document
    
    # Save to disk
    with open(path_obj, 'w', encoding='utf-8') as f:
        if pretty:
            json.dump(document_dict, f, indent=2, ensure_ascii=False)
        else:
            json.dump(document_dict, f, ensure_ascii=False)
    
    return path_obj

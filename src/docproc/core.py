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
from typing import Dict, Any, Optional, Union, List, Callable, cast

from pydantic import ValidationError

from .utils.format_detector import detect_format_from_path
from .utils.metadata_extractor import extract_metadata
from .adapters.registry import get_adapter_for_format
from .schemas.utils import validate_document, add_validation_to_adapter
from .schemas.base import BaseDocument

# Setup logging
logger = logging.getLogger(__name__)


def process_document(file_path: Union[str, Path], options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Process a document file, converting it to a standardized format.
    
    Args:
        file_path: Path to the document file
        options: Optional processing options
    
    Returns:
        Dictionary with processed content and metadata
        
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
        return validated_doc.model_dump()
    except ValidationError as e:
        logger.warning(f"Document validation failed: {e}")
        # Add validation error to the document
        processed_doc["_validation_error"] = str(e)
        return processed_doc


def process_text(text: str, format_type: str = "text", format_or_options: Optional[Union[str, Dict[str, Any]]] = None, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Process text content directly, assuming a specific format.
    
    Args:
        text: Text content to process
        format_type: Format of the text (e.g., "markdown", "html", "json")
        format_or_options: Either a format string or options dictionary (for flexibility)
        options: Optional processing options
    
    Returns:
        Dictionary with processed content and metadata
        
    Raises:
        ValidationError: If the processed document fails validation
    """
    # Handle flexible parameter usage
    if isinstance(format_or_options, dict):
        # If the third parameter is a dict, it's options
        actual_options = format_or_options
        actual_format = format_type
    elif isinstance(format_or_options, str):
        # If the third parameter is a string, it's the format
        actual_format = format_or_options
        actual_options = options or {}
    else:
        # Default case
        actual_format = format_type
        actual_options = options or {}
    
    # Get the appropriate adapter
    adapter = get_adapter_for_format(actual_format)
    
    # Process the text
    processed_doc = adapter.process_text(text, actual_options)
    
    # Enrich with metadata using our heuristic extraction
    content = processed_doc.get("content", "")
    source = actual_options.get("source", "direct_text")
    metadata = extract_metadata(content, source, actual_format)
    
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
        return validated_doc.model_dump()
    except ValidationError as e:
        logger.warning(f"Document validation failed: {e}")
        # Add validation error to the document
        processed_doc["_validation_error"] = str(e)
        return processed_doc


def get_format_for_document(file_path: Union[str, Path]) -> str:
    """
    Get the format for a document file.
    
    Args:
        file_path: Path to the document file
    
    Returns:
        Detected format as a string
    
    Raises:
        FileNotFoundError: If the file does not exist
    """
    path_obj = Path(file_path) if isinstance(file_path, str) else file_path
    
    # Check if file exists
    if not path_obj.exists():
        raise FileNotFoundError(f"File not found: {path_obj}")
    
    # Delegate to the utility function
    return detect_format_from_path(path_obj)


def detect_format(file_path: Union[str, Path]) -> str:
    """
    Detect the format of a document file.
    
    Args:
        file_path: Path to the document file
    
    Returns:
        Detected format as a string
    """
    path_obj = Path(file_path) if isinstance(file_path, str) else file_path
    
    # Delegate to the utility function
    return detect_format_from_path(path_obj)


def save_processed_document(document: Dict[str, Any], output_path: Union[str, Path]) -> Path:
    """
    Save a processed document JSON to disk.
    
    This is an optional utility for saving documents to disk when needed.
    The core pipeline will typically pass documents between stages in memory.
    
    Args:
        document: Processed document JSON to save
        output_path: Path where the JSON file should be saved
        
    Returns:
        Path to the saved JSON file
    """
    path_obj = Path(output_path) if isinstance(output_path, str) else output_path
    
    # Ensure parent directory exists
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    # Validate the document before saving if it hasn't been validated already
    if not document.get("_validated", False):
        try:
            # Attempt to validate the document
            validated_doc = validate_document(document)
            document = validated_doc.model_dump()
            document["_validated"] = True
        except ValidationError as e:
            logger.warning(f"Document validation failed before saving: {e}")
            # Add validation error to the document
            document["_validation_error"] = str(e)
    
    # Write JSON to file
    with open(path_obj, 'w', encoding='utf-8') as f:
        json.dump(document, f, ensure_ascii=False, indent=2)
    
    return path_obj


def process_documents_batch(
    file_paths: List[Union[str, Path]], 
    options: Optional[Dict[str, Any]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    on_success: Optional[Callable[[Dict[str, Any], Path], None]] = None,
    on_error: Optional[Callable[[str, Exception], None]] = None,
    validate: bool = True
) -> Dict[str, int]:
    """
    Process a batch of documents in parallel and return statistics.
    
    This is a convenience method that processes multiple documents and
    optionally saves them to disk or passes them to callback functions.
    
    Args:
        file_paths: List of paths to process
        options: Processing options to pass to the adapter
        output_dir: Optional directory to save outputs (None = no saving)
        on_success: Optional callback for successful processing
        on_error: Optional callback for processing errors
        validate: Whether to validate documents with Pydantic (default: True)
        
    Returns:
        Dictionary with processing statistics
    """
    stats = {
        "total": len(file_paths),
        "success": 0,
        "error": 0,
        "saved": 0,
        "validation_failures": 0
    }
    
    # Create output directory if needed
    if output_dir:
        out_dir = Path(output_dir) if isinstance(output_dir, str) else output_dir
        out_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each document
    for file_path in file_paths:
        path_obj = Path(file_path) if isinstance(file_path, str) else file_path
        
        try:
            # Process the document
            processed_doc = process_document(path_obj, options)
            
            # Check for validation errors
            if "_validation_error" in processed_doc:
                stats["validation_failures"] += 1
                logger.warning(f"Validation failed for {path_obj}: {processed_doc['_validation_error']}")
            
            stats["success"] += 1
            
            # Save if output directory is provided
            if output_dir:
                out_path = Path(output_dir) / f"{path_obj.stem}.json"
                save_processed_document(processed_doc, out_path)
                stats["saved"] += 1
            
            # Call success callback if provided
            if on_success:
                out_path = Path(output_dir) / f"{path_obj.stem}.json" if output_dir else None
                on_success(processed_doc, out_path)
                
        except Exception as e:
            stats["error"] += 1
            
            # Call error callback if provided
            if on_error:
                on_error(str(path_obj), e)
            else:
                # Log the error
                logger.error(f"Error processing {path_obj}: {e}")
    
    return stats

"""
Core functionality for document processing.

This module provides the primary interface for processing documents of various formats
and transforming them into standardized JSON objects. These objects can be directly
passed to the next stage in the processing pipeline (e.g., chunking, embedding) or
optionally saved to disk for inspection or archiving.

The focus is on in-memory processing for optimal performance in pipeline scenarios.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Callable

from .utils.format_detector import detect_format_from_path
from .adapters.registry import get_adapter_for_format


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
    return adapter.process(path_obj, options)


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
    return adapter.process_text(text, actual_options)


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
    
    # Write JSON to file
    with open(path_obj, 'w', encoding='utf-8') as f:
        json.dump(document, f, ensure_ascii=False, indent=2)
    
    return path_obj


def process_documents_batch(
    file_paths: List[Union[str, Path]], 
    options: Optional[Dict[str, Any]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    on_success: Optional[Callable[[Dict[str, Any], Path], None]] = None,
    on_error: Optional[Callable[[str, Exception], None]] = None
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
        
    Returns:
        Dictionary with processing statistics
    """
    stats = {"processed": 0, "errors": 0, "skipped": 0}
    
    # Process each document
    for path in file_paths:
        try:
            # Convert to Path if needed
            path_obj = Path(path) if isinstance(path, str) else path
            
            # Skip if file doesn't exist
            if not path_obj.exists():
                stats["skipped"] += 1
                if on_error:
                    on_error(str(path_obj), FileNotFoundError(f"File not found: {path_obj}"))
                continue
                
            # Process the document
            result = process_document(path_obj, options)
            stats["processed"] += 1
            
            # Save if output directory is specified
            if output_dir is not None:
                out_dir = Path(output_dir) if isinstance(output_dir, str) else output_dir
                rel_path = path_obj.with_suffix(path_obj.suffix + ".json")
                out_path = out_dir / rel_path.name
                save_processed_document(result, out_path)
                
                # Call success callback if provided
                if on_success:
                    on_success(result, out_path)
            elif on_success:
                # Call success callback without saving
                on_success(result, path_obj)
                
        except Exception as e:
            stats["errors"] += 1
            if on_error:
                on_error(str(path), e)
    
    return stats

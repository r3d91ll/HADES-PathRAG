"""
Core functionality for document processing.

This module provides the primary interface for processing documents of various formats.
"""

from pathlib import Path
from typing import Dict, Any, Optional, Union, List

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
    """
    options = options or {}
    path_obj = Path(file_path) if isinstance(file_path, str) else file_path
    
    # Detect the document format
    format_type = detect_format_from_path(path_obj)
    
    # Get the appropriate adapter
    adapter = get_adapter_for_format(format_type)
    
    # Process the document
    return adapter.process(path_obj, options)


def process_text(text: str, format_type: str = "text", format: Optional[str] = None, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Process text content directly, assuming a specific format.
    
    Args:
        text: Text content to process
        format_type: Format of the text (e.g., "markdown", "html", "json")
        format: Alternative way to specify format (for backward compatibility)
        options: Optional processing options
    
    Returns:
        Dictionary with processed content and metadata
    """
    options = options or {}
    
    # Use format parameter if provided (for backward compatibility)
    if format is not None:
        format_type = format
    
    # Get the appropriate adapter
    adapter = get_adapter_for_format(format_type)
    
    # Process the text
    return adapter.process_text(text, options)


def detect_format(file_path: Union[str, Path]) -> str:
    """
    Detect the format of a document file.
    
    Args:
        file_path: Path to the document file
    
    Returns:
        Detected format as a string
    """
    path_obj = Path(file_path) if isinstance(file_path, str) else file_path
    return detect_format_from_path(path_obj)

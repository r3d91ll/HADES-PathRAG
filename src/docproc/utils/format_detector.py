"""
Format detection utilities for document processing.

This module provides functions to detect document formats based on file extension,
content analysis, and other heuristics.
"""

import os
import mimetypes
from pathlib import Path
from typing import Optional, Union

# Initialize mimetypes
mimetypes.init()


def detect_format_from_path(file_path: Path) -> str:
    """
    Detect document format based on file extension.
    
    Args:
        file_path: Path to the document file
        
    Returns:
        Detected format as a string
        
    Raises:
        ValueError: If format cannot be determined
    """
    # Get the file extension
    ext = file_path.suffix.lower()
    
    # Map extensions to format types
    if ext == '.pdf':
        return 'pdf'
    elif ext in {'.html', '.htm'}:
        return 'html'
    elif ext == '.md':
        return 'markdown'
    elif ext in {'.docx', '.doc'}:
        return 'docx'
    elif ext == '.py':
        return 'python'
    elif ext in {'.js', '.java', '.cpp', '.c', '.h', '.cs', '.go', '.rs', '.ts'}:
        return 'code'
    elif ext == '.json':
        return 'json'
    elif ext in {'.yaml', '.yml'}:
        return 'yaml'
    elif ext == '.xml':
        return 'xml'
    elif ext == '.csv':
        return 'csv'
    elif ext == '.txt':
        return 'text'
    
    # If extension doesn't match, try using mimetypes
    mime_type, _ = mimetypes.guess_type(str(file_path))
    if mime_type:
        if mime_type.startswith('text/html'):
            return 'html'
        elif mime_type == 'application/pdf':
            return 'pdf'
        elif mime_type.startswith('text/'):
            return 'text'
        elif mime_type.startswith('application/json'):
            return 'json'
        elif mime_type.startswith('application/xml'):
            return 'xml'
    
    # If all else fails, default to 'text'
    # In a production system, we might want to inspect file contents here
    return 'text'


def detect_format_from_content(content: str) -> str:
    """
    Detect document format based on content analysis.
    
    Args:
        content: Text content to analyze
        
    Returns:
        Detected format as a string
    """
    # Trim whitespace and get first line
    content = content.strip()
    first_line = content.split('\n', 1)[0] if '\n' in content else content
    
    # Check for common format signatures
    if content.startswith('%PDF-'):
        return 'pdf'
    elif first_line.startswith('<!DOCTYPE html>') or '<html' in content.lower():
        return 'html'
    elif first_line.startswith('{') and content.strip().endswith('}'):
        # Likely JSON
        return 'json'
    elif first_line.startswith('<') and '>' in first_line:
        # Likely XML
        return 'xml'
    elif first_line.startswith('#') or '---' in content[:100]:
        # Likely markdown or YAML
        if '```' in content or '##' in content:
            return 'markdown'
        else:
            return 'yaml'
    elif 'def ' in content and 'import ' in content:
        # Likely Python code
        return 'code'
    elif 'function ' in content or 'class ' in content:
        # Likely other code
        return 'code'
    
    # Default to text
    return 'text'

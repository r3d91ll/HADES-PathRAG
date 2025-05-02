"""
Document format adapters for document processing.

This module provides adapters for different document formats like PDF, HTML, and code files.
"""

from .registry import register_adapter, get_adapter_for_format, get_adapter_class, get_supported_formats
from .base import BaseAdapter

# Import all adapters to ensure they are registered
from .pdf_adapter import PDFAdapter
from .html_adapter import HTMLAdapter
from .code_adapter import CodeAdapter
from .json_adapter import JSONAdapter
from .yaml_adapter import YAMLAdapter
from .xml_adapter import XMLAdapter
from .csv_adapter import CSVAdapter

# Add more adapters as they are implemented

__all__ = [
    'BaseAdapter',
    'register_adapter',
    'get_adapter_for_format',
    'get_adapter_class',
    'get_supported_formats',
    'PDFAdapter',
    'HTMLAdapter',
    'CodeAdapter',
    'JSONAdapter',
    'YAMLAdapter',
    'XMLAdapter',
    'CSVAdapter'
]

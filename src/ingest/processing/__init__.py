"""
Processing package for HADES-PathRAG.

This package contains components for file discovery, batching, 
and preprocessing during the ingestion pipeline.
"""

from .file_processor import FileProcessor
from .preprocessor_manager import PreprocessorManager

__all__ = [
    'FileProcessor',
    'PreprocessorManager'
]

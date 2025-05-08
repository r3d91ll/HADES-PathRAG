"""
Pre-processor module for HADES-PathRAG.

This module provides parallel processing functionality for multiple file types,
extracting structured information, relationships, and preparing documents for
the HADES-PathRAG knowledge graph.

This module now serves as an adapter over the new 'src.docproc' module, providing
backward compatibility while leveraging the improved document processing capabilities.

The modular architecture supports multiple file types including:
- Python source code
- Markdown with Mermaid diagrams
- HTML and websites
- PDF documents
- JSON, YAML, XML, and CSV data files
- Various code formats
"""

import logging
from typing import Type, Dict, Union, Callable, Any

# Set up logging
logger = logging.getLogger(__name__)

# Import configuration
from src.ingest.pre_processor.config_models import PreProcessorConfig
from src.ingest.pre_processor.config import (
    load_config,
    save_config,
    get_default_config
)

# Import pre-processor components
from .base_pre_processor import BasePreProcessor, DocProcAdapter
from .python_pre_processor import PythonPreProcessor
from .file_processor import FileProcessor
from .docling_pre_processor import DoclingPreProcessor
from .website_pdf_pre_processors import WebsitePreProcessor, PDFPreProcessor

# Define a type alias for registry entries that can be either a class type or a factory function
PreProcessorEntry = Union[Type[BasePreProcessor], Callable[[], BasePreProcessor]]

# Registry of pre-processors by file type - declare before manager import
PRE_PROCESSOR_REGISTRY: Dict[str, PreProcessorEntry] = {
    # Original processors for backward compatibility
    'python': PythonPreProcessor,  # Keep the specialized Python processor for now
    
    # Use the new docproc adapters for all other formats
    'markdown': lambda: DocProcAdapter(format_override='markdown'),
    'pdf': PDFPreProcessor,
    'html': WebsitePreProcessor,
    'website': WebsitePreProcessor,
    'website_html': WebsitePreProcessor,
    'pdf_file': PDFPreProcessor,
    'docx': lambda: DocProcAdapter(format_override='docx'),
    
    # Add support for all the new formats from docproc
    'json': lambda: DocProcAdapter(format_override='json'),
    'yaml': lambda: DocProcAdapter(format_override='yaml'),
    'xml': lambda: DocProcAdapter(format_override='xml'),
    'csv': lambda: DocProcAdapter(format_override='csv'),
    'code': lambda: DocProcAdapter(format_override='code'),
    'text': lambda: DocProcAdapter(format_override='text'),
}

def get_pre_processor(file_type: str) -> BasePreProcessor:
    """
    Factory function to get the appropriate pre-processor for a file type.
    
    Args:
        file_type: Type of file to process (e.g., 'python', 'markdown', 'pdf', 'html', 'docx')
    Returns:
        Pre-processor instance
    Raises:
        ValueError: If no pre-processor is available for the file type
    """
    pre_processor_class_or_factory = PRE_PROCESSOR_REGISTRY.get(file_type)
    if pre_processor_class_or_factory is None:
        # Check if we have a generic DocProcAdapter that can handle this format
        logger.info(f"No specific pre-processor for {file_type}, using DocProcAdapter")
        return DocProcAdapter(format_override=file_type)
        
    # Check if it's a factory function (lambda) or a class
    if callable(pre_processor_class_or_factory) and not isinstance(pre_processor_class_or_factory, type):
        # It's a factory function, call it to get an instance
        factory_result: BasePreProcessor = pre_processor_class_or_factory()
        return factory_result
    else:
        # It's a class, instantiate it
        class_result: BasePreProcessor = pre_processor_class_or_factory()
        return class_result

# Import manager after defining get_pre_processor to avoid circular imports
from .manager import PreprocessorManager

# Re-export chunking helpers so callers can still import via pre_processor
from src.ingest.chunking import chunk_code, chunk_text

__all__ = [
    # Configuration
    "PreProcessorConfig",
    # Chunking functions
    "chunk_code",
    "chunk_text",
    "load_config",
    "save_config",
    "get_default_config",
    
    # Pre-processor interface and implementations
    "BasePreProcessor",
    "DocProcAdapter",
    "PythonPreProcessor",
    "FileProcessor",
    "PreprocessorManager",
    # Chunking
    "chunk_code",
    "get_pre_processor",
    "PRE_PROCESSOR_REGISTRY"
]

"""
Pre-processor module for HADES-PathRAG.

This module provides parallel processing functionality for multiple file types,
extracting structured information, relationships, and preparing documents for
the HADES-PathRAG knowledge graph.

The modular architecture supports multiple file types including:
- Python source code
- Markdown with Mermaid diagrams
- (More file types will be added)
"""

# Import configuration
from src.ingest.pre_processor.config_models import PreProcessorConfig
from src.ingest.pre_processor.config import (
    load_config,
    save_config,
    get_default_config
)

# Import pre-processor components
from src.ingest.pre_processor.base_pre_processor import BasePreProcessor
from src.ingest.pre_processor.python_pre_processor import PythonPreProcessor
from src.ingest.pre_processor.docling_pre_processor import DoclingPreProcessor

from typing import Type, Dict
# Registry of pre-processors by file type
PRE_PROCESSOR_REGISTRY: Dict[str, Type[BasePreProcessor]] = {
    'python': PythonPreProcessor,
    'markdown': DoclingPreProcessor,
    'pdf': DoclingPreProcessor,
    'html': DoclingPreProcessor,
    'docx': DoclingPreProcessor,
    # Add other Docling-supported types as needed
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
    pre_processor_class = PRE_PROCESSOR_REGISTRY.get(file_type)
    if pre_processor_class is None:
        raise ValueError(f"No pre-processor available for file type: {file_type}")
    return pre_processor_class()

__all__ = [
    # Configuration
    "PreProcessorConfig",
    "load_config",
    "save_config",
    "get_default_config",
    
    # Pre-processor interface and implementations
    "BasePreProcessor",
    "PythonPreProcessor",
    "MarkdownPreProcessor",
    "get_pre_processor",
    "PRE_PROCESSOR_REGISTRY"
]

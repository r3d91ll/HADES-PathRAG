"""
Base classes for document chunking in HADES-PathRAG.

This module provides the base classes and interfaces for document chunking
components used in the HADES-PathRAG system.
"""

from typing import Dict, List, Any, Optional, Union
from abc import ABC, abstractmethod


class BaseChunker(ABC):
    """Base class for all document chunkers.
    
    A chunker is responsible for breaking down a document into smaller, semantically
    meaningful chunks that can be processed and embedded independently. Chunkers may
    be specialized for specific document types (e.g., text, code, PDF) and should
    preserve the semantic structure of the document.
    """
    
    def __init__(self, name: str = "base", config: Optional[Dict[str, Any]] = None):
        """Initialize the base chunker.
        
        Args:
            name: Name of the chunker
            config: Configuration options for the chunker
        """
        self.name = name
        self.config = config or {}
    
    @abstractmethod
    def chunk(self, content: Union[str, Dict[str, Any]], **kwargs) -> List[Dict[str, Any]]:
        """Chunk a document into smaller parts.
        
        Args:
            content: Document content to chunk
            **kwargs: Additional arguments specific to the chunker implementation
            
        Returns:
            List of chunk dictionaries, where each chunk contains at least:
            - text: The chunk content
            - metadata: Additional information about the chunk
        """
        pass


# Function to register chunkers in the registry
_CHUNKER_REGISTRY: Dict[str, type] = {}

def register_chunker(name: str, chunker_cls: type):
    """Register a chunker class with a name.
    
    Args:
        name: Name to register the chunker with
        chunker_cls: Chunker class to register
    """
    _CHUNKER_REGISTRY[name] = chunker_cls


def get_chunker(name: str, **kwargs) -> BaseChunker:
    """Get a chunker instance by name.
    
    Args:
        name: Name of the chunker to get
        **kwargs: Arguments to pass to the chunker constructor
        
    Returns:
        Chunker instance
        
    Raises:
        ValueError: If no chunker is registered with the given name
    """
    if name not in _CHUNKER_REGISTRY:
        raise ValueError(f"No chunker registered with name '{name}'")
    
    chunker_cls = _CHUNKER_REGISTRY[name]
    return chunker_cls(**kwargs)

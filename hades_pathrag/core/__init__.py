"""
Core PathRAG implementation.

This module contains the main PathRAG implementation that integrates
the embedders, graph, and storage components.
"""
from typing import List, Type, TypeVar, Dict, Any

# Type variable for PathRAG classes
T = TypeVar('T', bound='PathRAG')

from .pathrag import PathRAG
from .config import PathRAGConfig
from .factory import (
    create_embedder,
    create_graph,
    create_document_storage,
    create_vector_storage,
    create_graph_storage,
    create_pathrag_components
)

# __all__ defines the public API
__all__: List[str] = [
    'PathRAG',
    'PathRAGConfig',
    'create_embedder',
    'create_graph',
    'create_document_storage',
    'create_vector_storage',
    'create_graph_storage',
    'create_pathrag_components',
]

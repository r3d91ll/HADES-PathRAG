"""
Embedding models for PathRAG.

This module contains implementations of various embedding models,
with a focus on ISNE (Inductive Shallow Node Embedding).
"""

from typing import List, Type, TypeVar, Dict, Any, Optional, Union

# Type variable for embedding models
T = TypeVar('T', bound='BaseEmbedder')

# Import base classes and interfaces
from .base import BaseEmbedder
from .interfaces import (
    TrainableEmbedder, ComparableEmbedder, 
    EmbeddingStats, EmbeddingMetrics, EmbeddingCache
)

# Import concrete implementations
from .isne import ISNEEmbedder

# Import factory methods
from .factory import (
    EmbedderRegistry, create_embedder, create_embedder_from_file,
    register_external_embedder
)

# __all__ defines the public API
__all__: List[str] = [
    # Base classes and interfaces
    'BaseEmbedder',
    'TrainableEmbedder',
    'ComparableEmbedder',
    'EmbeddingStats',
    'EmbeddingMetrics',
    'EmbeddingCache',
    
    # Concrete implementations
    'ISNEEmbedder',
    
    # Factory methods
    'EmbedderRegistry',
    'create_embedder',
    'create_embedder_from_file',
    'register_external_embedder'
]

"""
Fallback embedder for error handling and debugging.

This module provides a simple fallback embedder that returns zeros for
all embedding operations, used as a safety mechanism when primary
embedders fail to load.
"""
from typing import Dict, List, Optional, Any, Tuple
import logging
import numpy as np

from hades_pathrag.embeddings.base import BaseEmbedder
from hades_pathrag.embeddings.interfaces import EmbeddingStats

logger = logging.getLogger(__name__)


class FallbackEmbedder(BaseEmbedder):
    """
    Simple fallback embedder that returns zeros.
    
    This embedder is used when the primary embedders fail to load, ensuring
    that the system can still function in a degraded state.
    """
    
    def __init__(self, dim: int = 768, name: str = "fallback") -> None:
        """
        Initialize the fallback embedder.
        
        Args:
            dim: Dimension of embeddings to return
            name: Name of the embedder
        """
        self.embedding_dim = dim
        self._name = name
        self._warned = False
        logger.warning(f"Using fallback embedder with dimension {dim}")
    
    @property
    def name(self) -> str:
        """Get the name of the embedder."""
        return self._name
    
    def encode(self, node_id: str, neighbors: List[str]) -> np.ndarray:
        """
        Return zero embeddings (fallback behavior).
        
        Args:
            node_id: ID of the node to encode
            neighbors: List of neighbor node IDs
            
        Returns:
            Zero vector of proper dimension
        """
        if not self._warned:
            logger.warning(f"Using fallback embedding for {node_id}")
            self._warned = True
        return np.zeros(self.embedding_dim)
    
    def get_embedding(self, node_id: str) -> Optional[np.ndarray]:
        """
        Get embedding for a node (always returns zeros).
        
        Args:
            node_id: ID of the node
            
        Returns:
            Zero vector of proper dimension
        """
        return np.zeros(self.embedding_dim)
    
    def get_stats(self) -> EmbeddingStats:
        """
        Get statistics about the embeddings.
        
        Returns:
            Basic statistics
        """
        return {
            "num_embeddings": 0,
            "embedding_dim": self.embedding_dim,
            "model_type": "fallback",
            "warning": "Using fallback embedder"
        }
    
    def save(self, path: str) -> None:
        """
        Pretend to save the model (no-op).
        
        Args:
            path: Path that would be used to save
        """
        logger.warning(f"Attempted to save fallback embedder to {path}, this is a no-op")
    
    @classmethod
    def load(cls, path: str) -> "FallbackEmbedder":
        """
        Pretend to load the model.
        
        Args:
            path: Path that would be used to load
            
        Returns:
            New FallbackEmbedder instance
        """
        logger.warning(f"Attempted to load fallback embedder from {path}, creating new instance")
        return cls()

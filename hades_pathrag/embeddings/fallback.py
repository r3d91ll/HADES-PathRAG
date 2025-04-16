"""
Fallback embedder for error handling and debugging.

This module provides a simple fallback embedder that returns zeros for
all embedding operations, used as a safety mechanism when primary
embedders fail to load.
"""
from typing import Dict, List, Optional, Tuple, Union
import logging
import numpy as np

from hades_pathrag.embeddings.base import BaseEmbedder
from hades_pathrag.embeddings.interfaces import EmbeddingStats
from hades_pathrag.typings import (
    EmbeddingArray, NodeIDType, Graph, EmbeddingDict, PathType
)

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
    
    def encode(self, node_id: NodeIDType, neighbors: List[NodeIDType]) -> EmbeddingArray:
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
        return np.zeros(self.embedding_dim, dtype=np.float32)
    
    def get_embedding(self, node_id: NodeIDType) -> Optional[EmbeddingArray]:
        """
        Get embedding for a node (always returns zeros).
        
        Args:
            node_id: ID of the node
            
        Returns:
            Zero vector of proper dimension
        """
        return np.zeros(self.embedding_dim, dtype=np.float32)
    
    def get_stats(self) -> EmbeddingStats:
        """
        Get statistics about the embeddings.
        
        Returns:
            Basic statistics
        """
        return EmbeddingStats(
            model_name=self._name,
            embedding_dim=self.embedding_dim,
            node_count=0,
            training_parameters={"warning": "Using fallback embedder"}
        )
    
    def fit(self, graph: Graph) -> None:
        """
        Pretend to fit the model (no-op).
        
        Args:
            graph: NetworkX graph representing the code structure
        """
        if not self._warned:
            logger.warning("Using fallback embedder's fit method (no-op)")
            self._warned = True
    
    def encode_text(self, text: str) -> EmbeddingArray:
        """
        Generate a zero embedding for the given text (fallback behavior).
        
        Args:
            text: Text to embed
            
        Returns:
            Zero vector of proper dimension
        """
        if not self._warned:
            logger.warning("Using fallback embedder for text encoding")
            self._warned = True
        return np.zeros(self.embedding_dim, dtype=np.float32)
    
    def batch_encode(self, nodes: Union[List[Tuple[NodeIDType, List[NodeIDType], Optional[str]]], Tuple[List[NodeIDType], List[List[NodeIDType]]]]) -> EmbeddingArray:
        """
        Generate zero embeddings for multiple nodes (fallback behavior).
        
        Args:
            nodes: Either a list of node tuples or a tuple of (node_ids, neighbor_lists)
            
        Returns:
            Matrix of zero embeddings
        """
        if not self._warned:
            logger.warning("Using fallback embedder for batch encoding")
            self._warned = True
            
        # Handle both input formats
        if isinstance(nodes, tuple):
            # Format is (node_ids, neighbor_lists)
            node_ids, _ = nodes
            count = len(node_ids)
        else:
            # Format is list of (node_id, neighbors, text) tuples
            count = len(nodes)
            
        # Return a matrix of zeros with the appropriate shape
        return np.zeros((count, self.embedding_dim), dtype=np.float32)
    
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

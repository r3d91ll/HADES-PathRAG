"""
Extended interfaces and data structures for embeddings in the PathRAG framework.

This module provides additional interfaces and concrete implementations for
embeddings in the PathRAG framework.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, TypeVar, Generic, Union, Any, Type, cast, Protocol, Tuple

import numpy as np
import networkx as nx
from pydantic import BaseModel, Field

from .base import BaseEmbedder, T

# Type definitions for embeddings
NodeID = str
Embedding = np.ndarray


class EmbeddingMetrics(BaseModel):
    """Metrics for evaluating embedding quality."""
    
    inductive_accuracy: float = Field(
        default=0.0,
        description="Accuracy on unseen nodes relative to seen nodes"
    )
    training_loss: float = Field(
        default=0.0, 
        description="Final training loss"
    )
    link_prediction_auc: Optional[float] = Field(
        default=None, 
        description="AUC for link prediction task"
    )
    node_classification_f1: Optional[float] = Field(
        default=None, 
        description="F1 score for node classification task"
    )
    training_time_seconds: float = Field(
        default=0.0, 
        description="Time taken to train the model"
    )
    embedding_time_ms: float = Field(
        default=0.0, 
        description="Average time to generate one embedding in milliseconds"
    )
    memory_usage_mb: Optional[float] = Field(
        default=None, 
        description="Approximate memory usage in MB"
    )


class EmbeddingStats(BaseModel):
    """Statistics about the embedding model and its performance."""
    
    model_name: str = Field(
        ..., 
        description="Name of the embedding model"
    )
    embedding_dim: int = Field(
        ..., 
        description="Dimensionality of embeddings"
    )
    node_count: int = Field(
        default=0, 
        description="Number of nodes in the model"
    )
    training_parameters: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Training parameters used"
    )
    metrics: EmbeddingMetrics = Field(
        default_factory=EmbeddingMetrics, 
        description="Performance metrics"
    )
    last_training_time: Optional[str] = Field(
        default=None, 
        description="Timestamp of last training"
    )
    last_update_time: Optional[str] = Field(
        default=None, 
        description="Timestamp of last update"
    )


class TrainableEmbedder(Protocol):
    """Protocol for embedders that support incremental training."""
    
    @abstractmethod
    def partial_fit(self, graph: nx.Graph, new_nodes: List[NodeID]) -> None:
        """
        Update the embedder model with new nodes while preserving existing embeddings.
        
        Args:
            graph: NetworkX graph with both existing and new nodes
            new_nodes: List of new node IDs to train on
        """
        pass
    
    @abstractmethod
    def get_training_stats(self) -> EmbeddingStats:
        """
        Get statistics about the embedding model and its training performance.
        
        Returns:
            Statistics about the embedding model
        """
        pass
    
    @abstractmethod
    def similarity(self, embedding1: Embedding, embedding2: Embedding) -> float:
        """
        Calculate similarity between two embeddings.
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            Similarity score between 0 and 1
        """
        pass


class ComparableEmbedder(Protocol):
    """Protocol for embedders that support comparison operations."""
    
    @abstractmethod
    def evaluate(self, test_graph: nx.Graph, task: str = "link_prediction") -> Dict[str, float]:
        """
        Evaluate the embedder on a test graph.
        
        Args:
            test_graph: NetworkX graph for testing
            task: Evaluation task, one of "link_prediction" or "node_classification"
            
        Returns:
            Dictionary of evaluation metrics
        """
        pass
    
    @abstractmethod
    def compare_with(self, other: BaseEmbedder, graph: nx.Graph) -> Dict[str, Any]:
        """
        Compare this embedder with another embedder on the same graph.
        
        Args:
            other: Another embedder to compare with
            graph: NetworkX graph for comparison
            
        Returns:
            Dictionary of comparison metrics
        """
        pass


class EmbeddingCacheProtocol(Protocol):
    """Protocol for embedding cache implementations."""
    
    def get(self, node_id: NodeID) -> Optional[Embedding]:
        """
        Get embedding for a node from the cache.
        
        Args:
            node_id: Node ID to retrieve
            
        Returns:
            Cached embedding if available, None otherwise
        """
        ...
    
    def put(self, node_id: NodeID, embedding: Embedding) -> None:
        """
        Store embedding for a node in the cache.
        
        Args:
            node_id: Node ID to store
            embedding: Embedding to cache
        """
        ...
    
    def invalidate(self, node_id: NodeID) -> None:
        """
        Invalidate a cached embedding.
        
        Args:
            node_id: Node ID to invalidate
        """
        ...
    
    def clear(self) -> None:
        """Clear all cached embeddings."""
        ...
    
    def size(self) -> int:
        """Get the current size of the cache.
        
        Returns:
            Number of items in the cache
        """
        ...


@dataclass
class EmbeddingCache(EmbeddingCacheProtocol):
    """Implementation of an embedding cache."""
    
    max_size: int = 10000
    cache: Dict[NodeID, Embedding] = field(default_factory=dict)
    
    def get(self, node_id: NodeID) -> Optional[Embedding]:
        """
        Get embedding for a node from the cache.
        
        Args:
            node_id: Node ID to retrieve
            
        Returns:
            Cached embedding if available, None otherwise
        """
        return self.cache.get(node_id)
    
    def put(self, node_id: NodeID, embedding: Embedding) -> None:
        """
        Store embedding for a node in the cache.
        
        Args:
            node_id: Node ID to store
            embedding: Embedding to cache
        """
        # Implement LRU cache if needed
        if len(self.cache) >= self.max_size:
            # Simple strategy: remove a random item
            if self.cache:
                self.cache.pop(next(iter(self.cache)))
        
        self.cache[node_id] = embedding
    
    def invalidate(self, node_id: NodeID) -> None:
        """
        Invalidate a cached embedding.
        
        Args:
            node_id: Node ID to invalidate
        """
        if node_id in self.cache:
            del self.cache[node_id]
    
    def clear(self) -> None:
        """Clear all cached embeddings."""
        self.cache.clear()
        
    def size(self) -> int:
        """Get the current size of the cache.
        
        Returns:
            Number of items in the cache
        """
        return len(self.cache)

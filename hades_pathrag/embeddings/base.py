"""
Base interfaces for embeddings in the PathRAG framework.

This module defines the abstract base classes for all embedding models.
"""
from abc import ABC, abstractmethod
from typing import List, Optional, TypeVar, Tuple, Union, Type

import numpy as np
import networkx as nx

# Import common types from our centralized typing module
from hades_pathrag.typings import (
    EmbeddingArray, NodeIDType, NodeData, Graph, GraphLike
)

# Type variable for embedding models
T = TypeVar('T', bound='BaseEmbedder')


class BaseEmbedder(ABC):
    """Base class for all embedding models."""
    
    @abstractmethod
    def fit(self, graph: Graph) -> None:
        """
        Fit the embedder model to the given graph.
        
        Args:
            graph: NetworkX graph representing the code structure
        """
        pass
    
    @abstractmethod
    def encode(self, node_id: NodeIDType, neighbors: List[NodeIDType]) -> EmbeddingArray:
        """
        Generate an embedding for the specified node.
        
        Args:
            node_id: ID of the node to embed
            neighbors: List of neighbor node IDs
            
        Returns:
            Node embedding vector
        """
        pass
    
    @abstractmethod
    def encode_text(self, text: str) -> EmbeddingArray:
        """
        Generate an embedding for the given text.
        
        Args:
            text: Text to embed
            
        Returns:
            Text embedding vector
        """
        pass
    
    @abstractmethod
    def batch_encode(self, nodes: Union[List[Tuple[NodeIDType, List[NodeIDType], Optional[str]]], Tuple[List[NodeIDType], List[List[NodeIDType]]]]) -> Union[EmbeddingArray, List[EmbeddingArray]]:
        """
        Generate embeddings for multiple nodes at once.
        
        This method can be called with either:
        - A list of (node_id, neighbors, text) tuples for enhanced embedders
        - A tuple of (node_ids, neighbor_lists) for basic embedders
        
        Args:
            nodes: Either a list of node tuples or a tuple of (node_ids, neighbor_lists)
            
        Returns:
            Either a matrix of node embeddings, or a list of embedding vectors
        """
        pass
    
    @abstractmethod
    def save(self, path: str) -> None:
        """
        Save the embedder model to disk.
        
        Args:
            path: Path to save the model to
        """
        pass
    
    @classmethod
    @abstractmethod
    def load(cls: Type[T], path: str) -> T:
        """
        Load the embedder model from disk.
        
        Args:
            path: Path to load the model from
            
        Returns:
            Loaded embedder model
        """
        pass

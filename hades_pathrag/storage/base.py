"""
Base interfaces for storage operations in the PathRAG framework.

This module defines the abstract base classes for vector, document, and
graph storage operations.
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, TypeVar, Generic, Any, Set, Tuple, Type, Union

import numpy as np

# Import common types from our centralized typing module
from hades_pathrag.typings import (
    EmbeddingArray, NodeIDType, NodeData, EdgeData
)

# Type variables for storage classes
T = TypeVar('T', bound='BaseStorage')
VT = TypeVar('VT', bound='BaseVectorStorage')
DT = TypeVar('DT', bound='BaseDocumentStorage')
GT = TypeVar('GT', bound='BaseGraphStorage')


class BaseStorage(ABC):
    """Base class for all storage implementations."""
    
    @abstractmethod
    def initialize(self) -> None:
        """
        Initialize storage system, creating necessary collections or tables.
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """
        Close all connections and release resources.
        """
        pass


class BaseVectorStorage(BaseStorage):
    """Base class for vector storage implementations."""
    
    @abstractmethod
    def store_embedding(self, node_id: NodeIDType, embedding: EmbeddingArray, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Store a node embedding with optional metadata.
        
        Args:
            node_id: Unique identifier for the node
            embedding: Vector embedding of the node
            metadata: Optional metadata to store with the embedding
        """
        pass
    
    @abstractmethod
    def get_embedding(self, node_id: NodeIDType) -> Optional[EmbeddingArray]:
        """
        Get embedding by node ID.
        
        Args:
            node_id: ID of the node to retrieve
            
        Returns:
            Node embedding if found, None otherwise
        """
        pass
    
    @abstractmethod
    def retrieve_similar(
        self, 
        query_embedding: EmbeddingArray, 
        k: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[NodeIDType, float, Dict[str, Any]]]:
        """
        Find nodes with embeddings similar to the query embedding.
        
        Args:
            query_embedding: Query vector
            k: Number of results to return
            filter_metadata: Optional filter to apply to metadata
            
        Returns:
            List of (node_id, similarity_score, metadata) tuples
        """
        pass
    
    @abstractmethod
    def delete_embedding(self, node_id: NodeIDType) -> bool:
        """
        Delete a node embedding.
        
        Args:
            node_id: ID of the node to delete
            
        Returns:
            True if deleted, False if not found
        """
        pass
    
    @abstractmethod
    def update_metadata(self, node_id: NodeIDType, metadata: Dict[str, Any]) -> bool:
        """
        Update metadata for a node embedding.
        
        Args:
            node_id: ID of the node to update
            metadata: New metadata to store
            
        Returns:
            True if updated, False if not found
        """
        pass


class BaseDocumentStorage(BaseStorage):
    """Base class for document storage implementations."""
    
    @abstractmethod
    def store_document(self, doc_id: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Store a document with optional metadata.
        
        Args:
            doc_id: Unique identifier for the document
            content: Text content of the document
            metadata: Optional metadata to store with the document
        """
        pass
    
    @abstractmethod
    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """
        Get document by ID.
        
        Args:
            doc_id: ID of the document to retrieve
            
        Returns:
            Document with content and metadata if found, None otherwise
        """
        pass
    
    @abstractmethod
    def list_documents(self, filter_metadata: Optional[Dict[str, Any]] = None) -> List[str]:
        """
        List document IDs, optionally filtered by metadata.
        
        Args:
            filter_metadata: Optional filter to apply to metadata
            
        Returns:
            List of document IDs matching the filter
        """
        pass
    
    @abstractmethod
    def delete_document(self, doc_id: str) -> bool:
        """
        Delete a document.
        
        Args:
            doc_id: ID of the document to delete
            
        Returns:
            True if deleted, False if not found
        """
        pass


class BaseGraphStorage(BaseStorage):
    """Base class for graph storage implementations integrating with BaseGraph."""
    
    @abstractmethod
    def store_node(self, node_id: NodeIDType, attributes: NodeData) -> None:
        """
        Store a node with attributes.
        
        Args:
            node_id: Unique identifier for the node
            attributes: Node attributes
        """
        pass
    
    @abstractmethod
    def store_edge(
        self,
        source_id: NodeIDType,
        target_id: NodeIDType,
        relation_type: str,
        weight: float = 1.0,
        attributes: Optional[EdgeData] = None
    ) -> None:
        """
        Store an edge between nodes.
        
        Args:
            source_id: Source node ID
            target_id: Target node ID
            relation_type: Type of relationship
            weight: Edge weight
            attributes: Optional edge attributes
        """
        pass
    
    @abstractmethod
    def get_node(self, node_id: NodeIDType) -> Optional[NodeData]:
        """
        Get node by ID.
        
        Args:
            node_id: ID of the node to retrieve
            
        Returns:
            Node attributes if found, None otherwise
        """
        pass
    
    @abstractmethod
    def get_neighbors(
        self, 
        node_id: NodeIDType, 
        direction: str = "outbound", 
        relation_types: Optional[List[str]] = None
    ) -> List[Tuple[NodeIDType, str, float]]:
        """
        Get neighbors of a node.
        
        Args:
            node_id: ID of the node
            direction: Direction of edges ("outbound", "inbound", or "any")
            relation_types: Optional filter by relation types
            
        Returns:
            List of (neighbor_id, relation_type, weight) tuples
        """
        pass
    
    @abstractmethod
    def query_subgraph(
        self,
        start_nodes: List[NodeIDType],
        max_depth: int = 2,
        relation_types: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Query a subgraph starting from given nodes.
        
        Args:
            start_nodes: List of starting node IDs
            max_depth: Maximum traversal depth
            relation_types: Optional filter by relation types
            
        Returns:
            Dictionary with "nodes" and "edges" representing the subgraph
        """
        pass

"""
Extended interfaces and data structures for storage operations in PathRAG.

This module provides additional interfaces and helper classes for 
storage operations in the PathRAG framework.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Set, Tuple, Any, TypeVar, Generic, Protocol, Union, runtime_checkable

import numpy as np
from pydantic import BaseModel, Field

# Import common types from our centralized typing module
from hades_pathrag.typings import (
    EmbeddingArray, NodeIDType, NodeData, EdgeData, PathType
)

from .base import (
    BaseStorage, BaseVectorStorage, BaseDocumentStorage, BaseGraphStorage
)


class StorageStats(BaseModel):
    """Statistics about a storage system."""
    
    storage_type: str = Field(
        ...,
        description="Type of storage (vector, document, graph)"
    )
    item_count: int = Field(
        default=0,
        description="Number of items in storage"
    )
    storage_size_bytes: Optional[int] = Field(
        default=None,
        description="Size of storage in bytes if available"
    )
    index_stats: Dict[str, Any] = Field(
        default_factory=dict,
        description="Statistics about storage indices"
    )
    query_stats: Dict[str, float] = Field(
        default_factory=dict,
        description="Query performance statistics"
    )
    last_updated: Optional[str] = Field(
        default=None,
        description="Timestamp of last update"
    )


class DocumentChunk(BaseModel):
    """Represents a chunk of a document."""
    
    chunk_id: str = Field(
        ...,
        description="Unique identifier for the chunk"
    )
    doc_id: str = Field(
        ...,
        description="ID of the parent document"
    )
    content: str = Field(
        ...,
        description="Text content of the chunk"
    )
    chunk_index: int = Field(
        default=0,
        description="Index of this chunk within the document"
    )
    token_count: Optional[int] = Field(
        default=None,
        description="Number of tokens in the chunk"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata for the chunk"
    )


class BulkOperationResult(BaseModel):
    """Result of a bulk operation."""
    
    success_count: int = Field(
        default=0,
        description="Number of successful operations"
    )
    error_count: int = Field(
        default=0,
        description="Number of failed operations"
    )
    errors: Dict[str, str] = Field(
        default_factory=dict,
        description="Map of IDs to error messages for failed operations"
    )


class QueryOperator(str, Enum):
    """Operators for metadata queries."""
    EQUALS = "eq"
    NOT_EQUALS = "neq"
    GREATER_THAN = "gt"
    GREATER_EQUAL = "gte"
    LESS_THAN = "lt"
    LESS_EQUAL = "lte"
    CONTAINS = "contains"
    NOT_CONTAINS = "not_contains"
    IN = "in"
    NOT_IN = "not_in"
    MATCHES = "matches"  # Regex match


class MetadataCondition(BaseModel):
    """Condition for querying metadata."""
    
    field: str = Field(
        ...,
        description="Field name to apply condition to"
    )
    operator: QueryOperator = Field(
        default=QueryOperator.EQUALS,
        description="Comparison operator"
    )
    value: Any = Field(
        ...,
        description="Value to compare against"
    )


class MetadataQuery(BaseModel):
    """Query for filtering by metadata."""
    
    conditions: List[MetadataCondition] = Field(
        default_factory=list,
        description="List of conditions to apply"
    )
    combine_operator: str = Field(
        default="AND",
        description="How to combine conditions ('AND' or 'OR')"
    )


@runtime_checkable
class EnhancedVectorStorage(Protocol):
    """Protocol for enhanced vector storage implementations."""
    
    @abstractmethod
    def bulk_store_embeddings(
        self, 
        items: List[Tuple[NodeIDType, EmbeddingArray, Optional[Dict[str, Any]]]]
    ) -> BulkOperationResult:
        """
        Store multiple embeddings in a single batch operation.
        
        Args:
            items: List of (node_id, embedding, metadata) tuples
            
        Returns:
            Result of the bulk operation
        """
        ...
    
    @abstractmethod
    def query_by_metadata(
        self, 
        query: MetadataQuery,
        limit: int = 100
    ) -> List[Tuple[NodeIDType, Dict[str, Any]]]:
        """
        Find nodes by metadata query.
        
        Args:
            query: Metadata query to filter by
            limit: Maximum number of results
            
        Returns:
            List of (node_id, metadata) tuples
        """
        ...
    
    @abstractmethod
    def hybrid_search(
        self,
        query_embedding: EmbeddingArray,
        metadata_query: MetadataQuery,
        k: int = 10,
        vector_weight: float = 0.5
    ) -> List[Tuple[NodeIDType, float, Dict[str, Any]]]:
        """
        Perform hybrid search combining vector similarity and metadata filtering.
        
        Args:
            query_embedding: Query vector
            metadata_query: Metadata query for filtering
            k: Number of results to return
            vector_weight: Weight of vector similarity vs metadata in final score
            
        Returns:
            List of (node_id, score, metadata) tuples
        """
        ...
    
    @abstractmethod
    def get_stats(self) -> StorageStats:
        """
        Get statistics about the vector storage.
        
        Returns:
            Storage statistics
        """
        ...


@runtime_checkable
class EnhancedDocumentStorage(Protocol):
    """Protocol for enhanced document storage implementations."""
    
    @abstractmethod
    def chunk_document(
        self, 
        doc_id: str, 
        content: str,
        chunk_size: int = 1024,
        chunk_overlap: int = 128,
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[DocumentChunk]:
        """
        Split a document into chunks and store them.
        
        Args:
            doc_id: Unique identifier for the document
            content: Text content of the document
            chunk_size: Size of each chunk in tokens
            chunk_overlap: Number of overlapping tokens between chunks
            metadata: Optional metadata to store with the chunks
            
        Returns:
            List of created document chunks
        """
        ...
    
    @abstractmethod
    def get_chunks(
        self, 
        doc_id: str
    ) -> List[DocumentChunk]:
        """
        Get all chunks for a document.
        
        Args:
            doc_id: ID of the document
            
        Returns:
            List of document chunks
        """
        ...
    
    @abstractmethod
    def search_documents(
        self, 
        query: str,
        limit: int = 10
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Search documents by text content.
        
        Args:
            query: Text query to search for
            limit: Maximum number of results
            
        Returns:
            List of (doc_id, score, metadata) tuples
        """
        ...
    
    @abstractmethod
    def bulk_store_documents(
        self,
        documents: List[Tuple[str, str, Optional[Dict[str, Any]]]]
    ) -> BulkOperationResult:
        """
        Store multiple documents in a single batch operation.
        
        Args:
            documents: List of (doc_id, content, metadata) tuples
            
        Returns:
            Result of the bulk operation
        """
        ...
    
    @abstractmethod
    def get_stats(self) -> StorageStats:
        """
        Get statistics about the document storage.
        
        Returns:
            Storage statistics
        """
        ...


@runtime_checkable
class EnhancedGraphStorage(Protocol):
    """Protocol for enhanced graph storage implementations."""
    
    @abstractmethod
    def bulk_store_nodes(
        self,
        nodes: List[Tuple[NodeIDType, NodeData]]
    ) -> BulkOperationResult:
        """
        Store multiple nodes in a single batch operation.
        
        Args:
            nodes: List of (node_id, attributes) tuples
            
        Returns:
            Result of the bulk operation
        """
        ...
    
    @abstractmethod
    def bulk_store_edges(
        self,
        edges: List[Tuple[NodeIDType, NodeIDType, str, float, Optional[EdgeData]]]
    ) -> BulkOperationResult:
        """
        Store multiple edges in a single batch operation.
        
        Args:
            edges: List of (source_id, target_id, relation_type, weight, attributes) tuples
            
        Returns:
            Result of the bulk operation
        """
        ...
    
    @abstractmethod
    def find_shortest_path(
        self,
        source_id: NodeIDType,
        target_id: NodeIDType,
        max_depth: int = 5
    ) -> Optional[PathType]:
        """
        Find the shortest path between two nodes.
        
        Args:
            source_id: Source node ID
            target_id: Target node ID
            max_depth: Maximum path depth to search
            
        Returns:
            List of node IDs in the path if found, None otherwise
        """
        ...
    
    @abstractmethod
    def get_node_degree(
        self,
        node_id: NodeIDType,
        direction: str = "outbound"
    ) -> int:
        """
        Get the degree of a node.
        
        Args:
            node_id: ID of the node
            direction: Direction of edges ("outbound", "inbound", or "any")
            
        Returns:
            Node degree
        """
        ...
    
    @abstractmethod
    def get_connected_components(self) -> List[Set[NodeIDType]]:
        """
        Get all connected components in the graph.
        
        Returns:
            List of sets of node IDs, each set representing a connected component
        """
        ...
    
    @abstractmethod
    def get_stats(self) -> StorageStats:
        """
        Get statistics about the graph storage.
        
        Returns:
            Storage statistics
        """
        ...


class StorageTransaction(Protocol):
    """Protocol for storage transactions."""
    
    def begin(self) -> bool:
        """Begin a transaction.
        
        Returns:
            True if transaction started successfully, False otherwise
        """
        ...
    
    def commit(self) -> bool:
        """Commit the transaction.
        
        Returns:
            True if transaction committed successfully, False otherwise
        """
        ...
    
    def rollback(self) -> None:
        """Roll back the transaction."""
        ...
    
    def __enter__(self) -> 'StorageTransaction':
        """Context manager entry."""
        ...
    
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        ...

"""
Factory methods for creating storage implementations.

This module provides factory methods for creating storage adapters
with different implementations and configurations.
"""
from typing import Dict, List, Optional, Any, Union, Callable, TypeVar, cast
import logging

from .base import BaseStorage, BaseVectorStorage, BaseDocumentStorage, BaseGraphStorage
from .interfaces import (
    EnhancedVectorStorage, EnhancedDocumentStorage, EnhancedGraphStorage,
    StorageStats, DocumentChunk, BulkOperationResult,
    QueryOperator, MetadataCondition, MetadataQuery, StorageTransaction
)
from .arango import (
    ArangoDBConnection, ArangoVectorStorage, ArangoDocumentStorage, ArangoGraphStorage
)
from .arango_enhanced import (
    EnhancedArangoVectorStorage, EnhancedArangoDocumentStorage, EnhancedArangoGraphStorage
)
from .transaction import ArangoTransaction, create_transaction
from ..core.config import PathRAGConfig

logger = logging.getLogger(__name__)


class StorageRegistry:
    """Registry of available storage implementations."""
    
    _vector_registry: Dict[str, type] = {
        "arango": ArangoVectorStorage,
        "arango_enhanced": EnhancedArangoVectorStorage,
    }
    
    _document_registry: Dict[str, type] = {
        "arango": ArangoDocumentStorage,
        "arango_enhanced": EnhancedArangoDocumentStorage,
    }
    
    _graph_registry: Dict[str, type] = {
        "arango": ArangoGraphStorage,
        "arango_enhanced": EnhancedArangoGraphStorage,
    }
    
    @classmethod
    def register_vector_storage(cls, name: str, storage_cls: type) -> None:
        """Register a vector storage implementation."""
        cls._vector_registry[name.lower()] = storage_cls
        logger.info(f"Registered vector storage: {name}")
    
    @classmethod
    def register_document_storage(cls, name: str, storage_cls: type) -> None:
        """Register a document storage implementation."""
        cls._document_registry[name.lower()] = storage_cls
        logger.info(f"Registered document storage: {name}")
    
    @classmethod
    def register_graph_storage(cls, name: str, storage_cls: type) -> None:
        """Register a graph storage implementation."""
        cls._graph_registry[name.lower()] = storage_cls
        logger.info(f"Registered graph storage: {name}")
    
    @classmethod
    def get_vector_storage(cls, name: str) -> Optional[type]:
        """Get vector storage class by name."""
        return cls._vector_registry.get(name.lower())
    
    @classmethod
    def get_document_storage(cls, name: str) -> Optional[type]:
        """Get document storage class by name."""
        return cls._document_registry.get(name.lower())
    
    @classmethod
    def get_graph_storage(cls, name: str) -> Optional[type]:
        """Get graph storage class by name."""
        return cls._graph_registry.get(name.lower())
    
    @classmethod
    def list_vector_storage(cls) -> List[str]:
        """List available vector storage implementations."""
        return list(cls._vector_registry.keys())
    
    @classmethod
    def list_document_storage(cls) -> List[str]:
        """List available document storage implementations."""
        return list(cls._document_registry.keys())
    
    @classmethod
    def list_graph_storage(cls) -> List[str]:
        """List available graph storage implementations."""
        return list(cls._graph_registry.keys())


def create_arango_connection(
    config: Union[PathRAGConfig, Dict[str, Any]]
) -> ArangoDBConnection:
    """
    Create an ArangoDB connection from configuration.
    
    Args:
        config: PathRAG configuration or dictionary
        
    Returns:
        ArangoDB connection
    """
    if isinstance(config, PathRAGConfig):
        # Extract connection details from config
        return ArangoDBConnection(
            host=config.db_host,
            port=config.db_port,
            username=config.db_username,
            password=config.db_password,
            database=config.db_name  # Use proper parameter name
            # ArangoDBConnection doesn't accept ssl parameter
        )
    elif isinstance(config, dict):
        # Extract connection details from dictionary
        return ArangoDBConnection(
            host=config.get("db_host", "localhost"),
            port=config.get("db_port", 8529),
            username=config.get("db_username", "root"),
            password=config.get("db_password", ""),
            database=config.get("db_name", "pathrag")  # Use proper parameter name
            # ArangoDBConnection doesn't accept ssl parameter
        )
    else:
        raise ValueError("Invalid configuration type")


def create_vector_storage(
    storage_type: str,
    config: Union[PathRAGConfig, Dict[str, Any]],
    **kwargs: Any
) -> BaseVectorStorage:
    """
    Create a vector storage adapter.
    
    Args:
        storage_type: Type of storage (e.g., "arango", "arango_enhanced")
        config: PathRAG configuration or dictionary
        **kwargs: Additional configuration for the storage adapter
        
    Returns:
        Vector storage adapter
        
    Raises:
        ValueError: If the storage type is not supported
    """
    storage_cls = StorageRegistry.get_vector_storage(storage_type)
    if not storage_cls:
        available = ", ".join(StorageRegistry.list_vector_storage())
        raise ValueError(
            f"Unsupported vector storage type: {storage_type}. "
            f"Available types: {available}"
        )
    
    # Create ArangoDB connection if needed
    if storage_type.startswith("arango"):
        connection = create_arango_connection(config)
        
        # Get collection name from config
        collection_name = (
            config.vector_collection_name
            if isinstance(config, PathRAGConfig)
            else config.get("vector_collection_name", "vectors")
        )
        
        # Create storage with connection
        return cast(BaseVectorStorage, storage_cls(connection, collection_name, **kwargs))
    
    # For other storage types, pass config directly
    return cast(BaseVectorStorage, storage_cls(config, **kwargs))


def create_document_storage(
    storage_type: str,
    config: Union[PathRAGConfig, Dict[str, Any]],
    **kwargs: Any
) -> BaseDocumentStorage:
    """
    Create a document storage adapter.
    
    Args:
        storage_type: Type of storage (e.g., "arango", "arango_enhanced")
        config: PathRAG configuration or dictionary
        **kwargs: Additional configuration for the storage adapter
        
    Returns:
        Document storage adapter
        
    Raises:
        ValueError: If the storage type is not supported
    """
    storage_cls = StorageRegistry.get_document_storage(storage_type)
    if not storage_cls:
        available = ", ".join(StorageRegistry.list_document_storage())
        raise ValueError(
            f"Unsupported document storage type: {storage_type}. "
            f"Available types: {available}"
        )
    
    # Create ArangoDB connection if needed
    if storage_type.startswith("arango"):
        connection = create_arango_connection(config)
        
        # Get collection name from config
        collection_name = (
            config.document_collection_name
            if isinstance(config, PathRAGConfig)
            else config.get("document_collection_name", "documents")
        )
        
        # Create storage with connection
        return cast(BaseVectorStorage, storage_cls(connection, collection_name, **kwargs))
    
    # For other storage types, pass config directly
    return cast(BaseVectorStorage, storage_cls(config, **kwargs))


def create_graph_storage(
    storage_type: str,
    config: Union[PathRAGConfig, Dict[str, Any]],
    **kwargs: Any
) -> BaseGraphStorage:
    """
    Create a graph storage adapter.
    
    Args:
        storage_type: Type of storage (e.g., "arango", "arango_enhanced")
        config: PathRAG configuration or dictionary
        **kwargs: Additional configuration for the storage adapter
        
    Returns:
        Graph storage adapter
        
    Raises:
        ValueError: If the storage type is not supported
    """
    storage_cls = StorageRegistry.get_graph_storage(storage_type)
    if not storage_cls:
        available = ", ".join(StorageRegistry.list_graph_storage())
        raise ValueError(
            f"Unsupported graph storage type: {storage_type}. "
            f"Available types: {available}"
        )
    
    # Create ArangoDB connection if needed
    if storage_type.startswith("arango"):
        connection = create_arango_connection(config)
        
        # Get names from config
        if isinstance(config, PathRAGConfig):
            graph_name = config.graph_name
            node_collection = config.node_collection_name
            edge_collection = config.edge_collection_name
        else:
            graph_name = config.get("graph_name", "knowledge_graph")
            node_collection = config.get("node_collection_name", "nodes")
            edge_collection = config.get("edge_collection_name", "edges")
        
        # Create storage with connection
        return cast(BaseGraphStorage, storage_cls(
            connection=connection,
            graph_name=graph_name,
            node_collection_name=node_collection,
            edge_collection_name=edge_collection,
            **kwargs
        ))
    
    # For other storage types, pass config directly
    return cast(BaseVectorStorage, storage_cls(config, **kwargs))


def create_storage_transaction(
    config: Union[PathRAGConfig, Dict[str, Any]],
    collections_read: Optional[List[str]] = None,
    collections_write: Optional[List[str]] = None
) -> StorageTransaction:
    """
    Create a storage transaction manager.
    
    Args:
        config: PathRAG configuration or dictionary
        collections_read: Collections to read from
        collections_write: Collections to write to
        
    Returns:
        Storage transaction manager
    """
    # Currently only ArangoDB transactions are supported
    connection = create_arango_connection(config)
    
    return create_transaction(
        connection=connection,
        collections_read=collections_read,
        collections_write=collections_write
    )

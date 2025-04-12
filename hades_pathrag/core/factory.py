"""
Factory methods for creating PathRAG components.

This module provides factory methods for creating embedder, graph, and storage
components with proper configuration.
"""
from typing import Dict, Optional, Type, TypeVar, Any, cast

import logging

from ..embeddings.base import BaseEmbedder
from ..graph.base import BaseGraph
from ..storage.base import BaseStorage, BaseVectorStorage, BaseDocumentStorage, BaseGraphStorage
from .config import PathRAGConfig

# Import implementations
from ..embeddings.isne import ISNEEmbedder
from ..graph.networkx_impl import NetworkXGraph
from ..storage.arango import ArangoDBConnection, ArangoVectorStorage, ArangoDocumentStorage, ArangoGraphStorage

logger = logging.getLogger(__name__)

# Type variables for factory return types
T = TypeVar('T', bound=BaseEmbedder)
G = TypeVar('G', bound=BaseGraph)
S = TypeVar('S', bound=BaseStorage)


def create_embedder(config: PathRAGConfig) -> BaseEmbedder:
    """
    Create an embedder instance based on configuration.
    
    Args:
        config: PathRAG configuration
        
    Returns:
        Configured embedder instance
        
    Raises:
        ValueError: If the specified embedding model is not supported
    """
    embedding_model = config.embedding_model.lower()
    
    if embedding_model == "isne":
        return ISNEEmbedder(
            embedding_dim=config.embedding_dim,
            learning_rate=config.learning_rate,
            epochs=config.training_epochs,
            batch_size=config.training_batch_size,
            negative_samples=config.negative_samples
        )
    # Future implementations will add more embedding models here
    
    # Fallback to ISNE
    logger.warning(f"Unsupported embedding model: {embedding_model}, falling back to ISNE")
    return ISNEEmbedder(embedding_dim=config.embedding_dim)


def create_graph(config: PathRAGConfig) -> BaseGraph:
    """
    Create a graph instance based on configuration.
    
    Args:
        config: PathRAG configuration
        
    Returns:
        Configured graph instance
    """
    # Currently we only support NetworkX graphs
    # In future versions, we'll support other graph implementations
    return NetworkXGraph(directed=True)


def create_storage_connection(config: PathRAGConfig) -> ArangoDBConnection:
    """
    Create a storage connection based on configuration.
    
    Args:
        config: PathRAG configuration
        
    Returns:
        Configured storage connection
    """
    return ArangoDBConnection(
        host=config.arango_host,
        port=config.arango_port,
        username=config.arango_username,
        password=config.arango_password,
        database=config.arango_database
    )


def create_vector_storage(config: PathRAGConfig) -> BaseVectorStorage:
    """
    Create a vector storage instance based on configuration.
    
    Args:
        config: PathRAG configuration
        
    Returns:
        Configured vector storage instance
    """
    connection = create_storage_connection(config)
    return ArangoVectorStorage(
        connection=connection,
        collection_name="embeddings",
        dimension=config.embedding_dim
    )


def create_document_storage(config: PathRAGConfig) -> BaseDocumentStorage:
    """
    Create a document storage instance based on configuration.
    
    Args:
        config: PathRAG configuration
        
    Returns:
        Configured document storage instance
    """
    connection = create_storage_connection(config)
    return ArangoDocumentStorage(
        connection=connection,
        collection_name="documents"
    )


def create_graph_storage(config: PathRAGConfig) -> BaseGraphStorage:
    """
    Create a graph storage instance based on configuration.
    
    Args:
        config: PathRAG configuration
        
    Returns:
        Configured graph storage instance
    """
    connection = create_storage_connection(config)
    return ArangoGraphStorage(
        connection=connection,
        graph_name="pathrag",
        node_collection_name="nodes",
        edge_collection_name="edges"
    )


def create_pathrag_components(config: PathRAGConfig) -> Dict[str, Any]:
    """
    Create all PathRAG components based on configuration.
    
    Args:
        config: PathRAG configuration
        
    Returns:
        Dictionary with all configured components
    """
    embedder = create_embedder(config)
    graph = create_graph(config)
    vector_storage = create_vector_storage(config)
    document_storage = create_document_storage(config)
    graph_storage = create_graph_storage(config)
    
    return {
        "embedder": embedder,
        "graph": graph,
        "vector_storage": vector_storage,
        "document_storage": document_storage,
        "graph_storage": graph_storage,
        "config": config
    }

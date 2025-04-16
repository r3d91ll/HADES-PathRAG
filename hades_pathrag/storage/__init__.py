"""
Storage interfaces and implementations for PathRAG.

This module contains abstract storage interfaces and concrete implementations
for different storage backends used in PathRAG.
"""

from typing import List, Dict, Any, Optional, Set, Tuple, Union, Type, TypeVar

# Type variables for storage classes
T = TypeVar('T', bound='BaseStorage')

# Import centralized type definitions
from hades_pathrag.typings import NodeIDType, EmbeddingArray

# Import base classes and interfaces
from .base import (
    BaseStorage, BaseVectorStorage, BaseDocumentStorage, BaseGraphStorage
)
from .interfaces import (
    EnhancedVectorStorage, EnhancedDocumentStorage, EnhancedGraphStorage,
    StorageStats, DocumentChunk, BulkOperationResult,
    QueryOperator, MetadataCondition, MetadataQuery, StorageTransaction
)

# Import concrete implementations
from .arango import (
    ArangoDBConnection, ArangoVectorStorage, ArangoDocumentStorage, ArangoGraphStorage
)
from .arango_enhanced import (
    EnhancedArangoVectorStorage, EnhancedArangoDocumentStorage, EnhancedArangoGraphStorage
)
from .transaction import ArangoTransaction, create_transaction
from .factory import (
    StorageRegistry, create_vector_storage, create_document_storage,
    create_graph_storage, create_storage_transaction, create_arango_connection
)

# __all__ defines the public API
__all__: List[str] = [
    # Type definitions
    'NodeIDType',
    'EmbeddingArray',
    
    # Base classes and interfaces
    'BaseStorage',
    'BaseVectorStorage',
    'BaseDocumentStorage',
    'BaseGraphStorage',
    'EnhancedVectorStorage',
    'EnhancedDocumentStorage',
    'EnhancedGraphStorage',
    'StorageStats',
    'DocumentChunk',
    'BulkOperationResult',
    'QueryOperator',
    'MetadataCondition',
    'MetadataQuery',
    'StorageTransaction',
    
    # Concrete implementations
    'ArangoDBConnection',
    'ArangoVectorStorage',
    'ArangoDocumentStorage',
    'ArangoGraphStorage',
    'EnhancedArangoVectorStorage',
    'EnhancedArangoDocumentStorage',
    'EnhancedArangoGraphStorage',
    'ArangoTransaction',
    
    # Factory methods
    'StorageRegistry',
    'create_vector_storage',
    'create_document_storage',
    'create_graph_storage',
    'create_storage_transaction',
    'create_arango_connection',
    'create_transaction'
]

"""
Core data models for the ISNE (Inductive Shallow Node Embedding) pipeline.

This module defines the foundational data structures for representing documents, 
embeddings, relationships, and configuration options in the ISNE pipeline.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Optional, Set, Union, Any, Tuple, TypeVar, Generic, Protocol, runtime_checkable
import uuid
import numpy as np
from datetime import datetime


# Type alias for embedding vectors
EmbeddingVector = Union[List[float], np.ndarray]


class RelationType(str, Enum):
    """Enum representing types of relationships between documents."""
    CONTAINS = "contains"
    REFERENCES = "references"
    CONNECTS_TO = "connects_to"
    SIMILAR_TO = "similar_to"
    DEPENDS_ON = "depends_on"
    PART_OF = "part_of"
    RELATED_TO = "related_to"
    DERIVED_FROM = "derived_from"
    IMPLEMENTS = "implements"
    CUSTOM = "custom"


@dataclass
class DocumentRelation:
    """
    Represents a relationship between two documents.
    
    This class captures the directional relationship between documents
    with a specific type and optional metadata.
    """
    source_id: str
    target_id: str
    relation_type: RelationType
    weight: float = 1.0
    bidirectional: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relation_type": self.relation_type.value,
            "weight": self.weight,
            "bidirectional": self.bidirectional,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }


@dataclass
class IngestDocument:
    """
    Represents a document to be processed by the ISNE pipeline.
    
    This class holds the document content, metadata, and computed embeddings.
    """
    # Core document properties
    id: str
    content: str
    source: str  # Origin of the document (filename, URL, etc.)
    document_type: str  # Type of document (code, markdown, text, etc.)
    
    # Document metadata
    title: Optional[str] = None
    author: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Embedding data
    embedding: Optional[EmbeddingVector] = None
    embedding_model: Optional[str] = None
    
    # Processing metadata
    chunks: List[Dict[str, Any]] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        # Convert embedding to list if it's a numpy array
        embedding_list = None
        if self.embedding is not None:
            if isinstance(self.embedding, np.ndarray):
                embedding_list = self.embedding.tolist()
            else:
                embedding_list = self.embedding
                
        return {
            "id": self.id,
            "content": self.content,
            "source": self.source,
            "document_type": self.document_type,
            "title": self.title,
            "author": self.author,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "metadata": self.metadata,
            "embedding": embedding_list,
            "embedding_model": self.embedding_model,
            "chunks": self.chunks,
            "tags": self.tags
        }


@dataclass
class IngestDataset:
    """
    Represents a collection of documents and their relationships.
    
    This class is the central data structure for the ISNE pipeline,
    containing documents and the graph of relationships between them.
    """
    # Dataset identification
    id: str
    name: str
    description: Optional[str] = None
    
    # Document and relationship collections
    documents: Dict[str, IngestDocument] = field(default_factory=dict)
    relations: List[DocumentRelation] = field(default_factory=list)
    
    # Dataset metadata
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_document(self, document: IngestDocument) -> None:
        """Add a document to the dataset."""
        self.documents[document.id] = document
        self.updated_at = datetime.now()
    
    def add_relation(self, relation: DocumentRelation) -> None:
        """Add a relationship between documents to the dataset."""
        self.relations.append(relation)
        self.updated_at = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "documents": {doc_id: doc.to_dict() for doc_id, doc in self.documents.items()},
            "relations": [rel.to_dict() for rel in self.relations],
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "metadata": self.metadata
        }


@dataclass
class EmbeddingConfig:
    """Configuration for embedding models in the ISNE pipeline."""
    model_name: str
    model_dimension: int
    batch_size: int = 16
    use_gpu: bool = True
    normalize_embeddings: bool = True
    cache_embeddings: bool = True
    pooling_strategy: str = "mean"
    max_length: int = 512
    
    # vLLM acceleration options
    use_vllm: bool = False
    vllm_server_url: Optional[str] = None
    vllm_tensor_parallel_size: int = 1
    vllm_gpu_memory_utilization: float = 0.8
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "model_name": self.model_name,
            "model_dimension": self.model_dimension,
            "batch_size": self.batch_size,
            "use_gpu": self.use_gpu,
            "normalize_embeddings": self.normalize_embeddings,
            "cache_embeddings": self.cache_embeddings,
            "pooling_strategy": self.pooling_strategy,
            "max_length": self.max_length
        }


@dataclass
class ISNEConfig:
    """
    Configuration for the ISNE (Inductive Shallow Node Embedding) model.
    
    This class contains parameters for the ISNE model architecture,
    training process, and graph construction.
    """
    # Model architecture
    input_dim: int 
    hidden_dim: int = 128
    output_dim: int = 128
    num_layers: int = 2
    
    # Training parameters
    learning_rate: float = 0.01
    weight_decay: float = 0.0005
    epochs: int = 100
    early_stopping: bool = True
    patience: int = 10
    
    # Graph construction
    min_edge_weight: float = 0.1
    max_distance: int = 3
    include_self_loops: bool = True
    bidirectional_edges: bool = False
    
    # Processing options
    use_gpu: bool = True
    batch_size: int = 32
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "output_dim": self.output_dim,
            "num_layers": self.num_layers,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "epochs": self.epochs,
            "early_stopping": self.early_stopping,
            "patience": self.patience,
            "min_edge_weight": self.min_edge_weight,
            "max_distance": self.max_distance,
            "include_self_loops": self.include_self_loops,
            "bidirectional_edges": self.bidirectional_edges,
            "use_gpu": self.use_gpu,
            "batch_size": self.batch_size
        }

"""
Document schema standardization for HADES-PathRAG.

This module defines Pydantic models for document schema validation,
ensuring consistent structure and validation throughout the pipeline.
"""
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union, Literal
import uuid

import numpy as np
from pydantic import BaseModel, Field, validator, root_validator
from pydantic.version import VERSION as PYDANTIC_VERSION

# We're using Pydantic v2
PYDANTIC_V2 = True

# Type alias for embedding vectors
EmbeddingVector = Union[List[float], np.ndarray]


class SchemaVersion(str, Enum):
    """Schema version enumeration for backward compatibility tracking."""
    V1 = "1.0.0"
    V2 = "2.0.0"  # Current version


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
    
    # New relation types can be added here as the system evolves


class DocumentType(str, Enum):
    """Enum representing types of documents."""
    CODE = "code"
    MARKDOWN = "markdown"
    TEXT = "text"
    PDF = "pdf"
    HTML = "html"
    JSON = "json"
    XML = "xml"
    YAML = "yaml"
    CSV = "csv"
    UNKNOWN = "unknown"


class ChunkMetadata(BaseModel):
    """Metadata for document chunks."""
    start_offset: int = Field(..., description="Start position in the original document")
    end_offset: int = Field(..., description="End position in the original document")
    chunk_type: str = Field(default="text", description="Type of the chunk (text, code, etc.)")
    chunk_index: int = Field(..., description="Sequential index of the chunk")
    parent_id: str = Field(..., description="ID of the parent document")
    
    # Additional metadata fields
    context_before: Optional[str] = Field(default=None, description="Text context before the chunk")
    context_after: Optional[str] = Field(default=None, description="Text context after the chunk")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional chunk-specific metadata")

    model_config = {
        "arbitrary_types_allowed": True,
        "validate_assignment": True,
        "extra": "allow",
        "json_encoders": {
            np.ndarray: lambda x: x.tolist() if x is not None else None
        }
    }


class DocumentRelationSchema(BaseModel):
    """Schema for document relations."""
    source_id: str = Field(..., description="ID of the source document")
    target_id: str = Field(..., description="ID of the target document")
    relation_type: RelationType = Field(..., description="Type of relationship")
    weight: float = Field(default=1.0, description="Weight or strength of the relationship")
    bidirectional: bool = Field(default=False, description="Whether the relationship is bidirectional")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional relation metadata")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique ID for this relation")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")

    class Config:
        """Pydantic configuration."""
        arbitrary_types_allowed = True
        json_encoders = {
            np.ndarray: lambda x: x.tolist() if x is not None else None,
            datetime: lambda x: x.isoformat() if x is not None else None
        }
        
        if not PYDANTIC_V2:
            # Pydantic v1 specific config
            validate_assignment = True
            extra = "allow"
        
    if PYDANTIC_V2:
        # Pydantic v2 compatible model configuration
        model_config = {
            "arbitrary_types_allowed": True,
            "validate_assignment": True,
            "extra": "allow",
            "json_encoders": {
                np.ndarray: lambda x: x.tolist() if x is not None else None,
                datetime: lambda x: x.isoformat() if x is not None else None
            }
        }

    @validator('relation_type', pre=True)
    def validate_relation_type(cls, v):
        """Validate relation type."""
        if isinstance(v, str) and v not in [item.value for item in RelationType]:
            return RelationType.CUSTOM
        return v


class DocumentSchema(BaseModel):
    """
    Pydantic schema for document validation and standardization.
    
    This model enforces structure and type safety for documents in the HADES-PathRAG system.
    """
    # Core document properties with validation
    id: str = Field(..., description="Unique identifier for the document")
    content: str = Field(..., description="Document content text")
    source: str = Field(..., description="Origin of the document (filename, URL, etc.)")
    document_type: DocumentType = Field(..., description="Type of document")
    
    # Schema versioning
    schema_version: SchemaVersion = Field(default=SchemaVersion.V2, description="Schema version for compatibility")
    
    # Document metadata
    title: Optional[str] = Field(default=None, description="Document title")
    author: Optional[str] = Field(default=None, description="Document author")
    created_at: Optional[datetime] = Field(default=None, description="Document creation timestamp")
    updated_at: Optional[datetime] = Field(default=None, description="Document last update timestamp")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional document metadata")
    
    # Embedding data
    embedding: Optional[EmbeddingVector] = Field(default=None, description="Document embedding vector")
    embedding_model: Optional[str] = Field(default=None, description="Model used to generate the embedding")
    
    # Processing metadata
    chunks: List[ChunkMetadata] = Field(default_factory=list, description="Document chunks metadata")
    tags: List[str] = Field(default_factory=list, description="Document tags for categorization")

    class Config:
        """Pydantic configuration."""
        arbitrary_types_allowed = True
        json_encoders = {
            np.ndarray: lambda x: x.tolist() if x is not None else None,
            datetime: lambda x: x.isoformat() if x is not None else None
        }
        
        if not PYDANTIC_V2:
            # Pydantic v1 specific config
            validate_assignment = True
            extra = "allow"
        
    if PYDANTIC_V2:
        # Pydantic v2 compatible model configuration
        model_config = {
            "arbitrary_types_allowed": True,
            "validate_assignment": True,
            "extra": "allow",
            "json_encoders": {
                np.ndarray: lambda x: x.tolist() if x is not None else None,
                datetime: lambda x: x.isoformat() if x is not None else None
            }
        }

    @validator('document_type', pre=True)
    def validate_document_type(cls, v):
        """Validate document type."""
        if isinstance(v, str):
            try:
                return DocumentType(v)
            except ValueError:
                return DocumentType.UNKNOWN
        return v

    @validator('id')
    def validate_id(cls, v):
        """Validate ID is not empty."""
        if not v:
            return str(uuid.uuid4())
        return v

    @root_validator
    def ensure_timestamps(cls, values):
        """Ensure timestamps are present and valid."""
        if not values.get('created_at'):
            values['created_at'] = datetime.now()
        if not values.get('updated_at'):
            values['updated_at'] = datetime.now()
        return values

    @root_validator
    def ensure_title(cls, values):
        """Derive title from source if not provided."""
        if not values.get('title') and values.get('source'):
            source = values.get('source', '')
            # Extract filename from path if source looks like a path
            if '/' in source:
                source = source.split('/')[-1]
            if '.' in source:
                source = source.split('.')[0]
            values['title'] = source
        return values
    
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
            "document_type": self.document_type.value,
            "schema_version": self.schema_version.value,
            "title": self.title,
            "author": self.author,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "metadata": self.metadata,
            "embedding": embedding_list,
            "embedding_model": self.embedding_model,
            "chunks": [chunk.dict() for chunk in self.chunks],
            "tags": self.tags
        }
    
    @classmethod
    def from_ingest_document(cls, doc):
        """
        Convert an existing IngestDocument to a DocumentSchema.
        
        Args:
            doc: An IngestDocument instance
            
        Returns:
            DocumentSchema: A validated document schema instance
        """
        # Handle document_type conversion
        doc_type = doc.document_type
        if isinstance(doc_type, str):
            try:
                doc_type = DocumentType(doc_type)
            except ValueError:
                doc_type = DocumentType.UNKNOWN
        
        # Convert chunks if they exist
        chunks = []
        for chunk_data in doc.chunks:
            # Ensure required fields exist
            if not all(k in chunk_data for k in ['start_offset', 'end_offset', 'chunk_index']):
                continue
                
            chunk = ChunkMetadata(
                start_offset=chunk_data.get('start_offset'),
                end_offset=chunk_data.get('end_offset'),
                chunk_type=chunk_data.get('chunk_type', 'text'),
                chunk_index=chunk_data.get('chunk_index'),
                parent_id=doc.id,
                context_before=chunk_data.get('context_before'),
                context_after=chunk_data.get('context_after'),
                metadata=chunk_data.get('metadata', {})
            )
            chunks.append(chunk)
        
        return cls(
            id=doc.id,
            content=doc.content,
            source=doc.source,
            document_type=doc_type,
            schema_version=SchemaVersion.V2,
            title=doc.title,
            author=doc.author,
            created_at=doc.created_at,
            updated_at=doc.updated_at,
            metadata=doc.metadata,
            embedding=doc.embedding,
            embedding_model=doc.embedding_model,
            chunks=chunks,
            tags=doc.tags
        )


class DatasetSchema(BaseModel):
    """Schema for dataset containing multiple documents and their relations."""
    # Dataset identification
    id: str = Field(..., description="Unique identifier for the dataset")
    name: str = Field(..., description="Name of the dataset")
    description: Optional[str] = Field(default=None, description="Dataset description")
    
    # Schema versioning
    schema_version: SchemaVersion = Field(default=SchemaVersion.V2, description="Schema version for compatibility")
    
    # Document and relationship collections
    documents: Dict[str, DocumentSchema] = Field(default_factory=dict, description="Documents in the dataset")
    relations: List[DocumentRelationSchema] = Field(
        default_factory=list, description="Relations between documents"
    )
    
    # Dataset metadata
    created_at: datetime = Field(default_factory=datetime.now, description="Dataset creation timestamp")
    updated_at: Optional[datetime] = Field(default=None, description="Dataset last update timestamp")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional dataset metadata")

    class Config:
        """Pydantic configuration."""
        arbitrary_types_allowed = True
        json_encoders = {
            np.ndarray: lambda x: x.tolist() if x is not None else None,
            datetime: lambda x: x.isoformat() if x is not None else None
        }
        
        if not PYDANTIC_V2:
            # Pydantic v1 specific config
            validate_assignment = True
            extra = "allow"
        
    if PYDANTIC_V2:
        # Pydantic v2 compatible model configuration
        model_config = {
            "arbitrary_types_allowed": True,
            "validate_assignment": True,
            "extra": "allow",
            "json_encoders": {
                np.ndarray: lambda x: x.tolist() if x is not None else None,
                datetime: lambda x: x.isoformat() if x is not None else None
            }
        }

    @validator('id')
    def validate_id(cls, v):
        """Validate ID is not empty."""
        if not v:
            return str(uuid.uuid4())
        return v

    @root_validator
    def ensure_timestamps(cls, values):
        """Ensure timestamps are present and valid."""
        if not values.get('created_at'):
            values['created_at'] = datetime.now()
        if not values.get('updated_at'):
            values['updated_at'] = datetime.now()
        return values
    
    def add_document(self, document: DocumentSchema) -> None:
        """Add a document to the dataset."""
        self.documents[document.id] = document
        self.updated_at = datetime.now()
    
    def add_relation(self, relation: DocumentRelationSchema) -> None:
        """Add a relationship between documents to the dataset."""
        self.relations.append(relation)
        self.updated_at = datetime.now()
    
    @classmethod
    def from_ingest_dataset(cls, dataset):
        """
        Convert an existing IngestDataset to a DatasetSchema.
        
        Args:
            dataset: An IngestDataset instance
            
        Returns:
            DatasetSchema: A validated dataset schema instance
        """
        # Convert documents
        documents = {}
        for doc_id, doc in dataset.documents.items():
            documents[doc_id] = DocumentSchema.from_ingest_document(doc)
        
        # Convert relations
        relations = []
        for rel in dataset.relations:
            relation = DocumentRelationSchema(
                source_id=rel.source_id,
                target_id=rel.target_id,
                relation_type=rel.relation_type,
                weight=rel.weight,
                bidirectional=rel.bidirectional,
                metadata=rel.metadata,
                id=rel.id,
                created_at=rel.created_at
            )
            relations.append(relation)
        
        return cls(
            id=dataset.id,
            name=dataset.name,
            description=dataset.description,
            schema_version=SchemaVersion.V2,
            documents=documents,
            relations=relations,
            created_at=dataset.created_at,
            updated_at=dataset.updated_at,
            metadata=dataset.metadata
        )

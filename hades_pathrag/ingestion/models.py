"""
Data models for the ingestion pipeline.

This module defines the basic data structures used in the ingestion pipeline,
such as documents, relationships, and datasets.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Any, Union
import uuid


class RelationType(str, Enum):
    """Types of relationships between documents."""
    CITES = "cites"
    REFERENCES = "references"
    LINKS_TO = "links_to"
    SIMILAR_TO = "similar_to"
    CONTAINS = "contains"
    PART_OF = "part_of"
    PREREQUISITE = "prerequisite"
    FOLLOWS = "follows"


@dataclass
class DocumentRelation:
    """A relation between two documents."""
    source_id: str
    target_id: str
    relation_type: RelationType
    weight: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IngestDocument:
    """A document to be ingested into the system."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: str = ""
    title: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None
    
    def __post_init__(self) -> None:
        if not self.title and self.content:
            # Extract a simple title from the first line if none provided
            self.title = self.content.split("\n")[0][:100]


@dataclass
class IngestDataset:
    """A collection of documents and their relationships to be ingested."""
    name: str
    documents: List[IngestDocument] = field(default_factory=list)
    relationships: List[DocumentRelation] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_document(self, document: IngestDocument) -> None:
        """Add a document to the dataset."""
        self.documents.append(document)
    
    def add_relationship(self, relationship: DocumentRelation) -> None:
        """Add a relationship to the dataset."""
        self.relationships.append(relationship)
    
    def add_documents(self, documents: List[IngestDocument]) -> None:
        """Add multiple documents to the dataset."""
        self.documents.extend(documents)
    
    def add_relationships(self, relationships: List[DocumentRelation]) -> None:
        """Add multiple relationships to the dataset."""
        self.relationships.extend(relationships)
    
    def get_document_by_id(self, doc_id: str) -> Optional[IngestDocument]:
        """Get a document by its ID."""
        for doc in self.documents:
            if doc.id == doc_id:
                return doc
        return None

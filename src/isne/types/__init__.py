"""
Types module for the ISNE (Inductive Shallow Node Embedding) pipeline.

This module provides type definitions for the core data structures 
used throughout the ISNE pipeline implementation.
"""

from .models import (
    IngestDocument,
    IngestDataset,
    DocumentRelation,
    RelationType,
    EmbeddingVector,
    EmbeddingConfig,
    ISNEConfig
)

__all__ = [
    "IngestDocument",
    "IngestDataset",
    "DocumentRelation",
    "RelationType",
    "EmbeddingVector", 
    "EmbeddingConfig",
    "ISNEConfig"
]

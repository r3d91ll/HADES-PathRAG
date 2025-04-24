"""
Central type definitions for HADES-PathRAG.

This module provides foundational type definitions used across
the HADES-PathRAG codebase to ensure type consistency.
"""

from .common import (
    NodeID, 
    EdgeID, 
    NodeData, 
    EdgeData, 
    EmbeddingVector,
    DocumentContent,
    StorageConfig,
    EmbeddingConfig,
    GraphConfig
)

__all__ = [
    "NodeID",
    "EdgeID", 
    "NodeData", 
    "EdgeData", 
    "EmbeddingVector",
    "DocumentContent",
    "StorageConfig",
    "EmbeddingConfig",
    "GraphConfig"
]

"""
Integration module for ISNE pipeline.

This module provides adapters and utilities for integrating the ISNE pipeline
with other components of the HADES-PathRAG system.
"""

from .arango_adapter import ArangoISNEAdapter
from .pathrag_adapter import PathRAGISNEAdapter

__all__ = [
    "ArangoISNEAdapter",
    "PathRAGISNEAdapter"
]

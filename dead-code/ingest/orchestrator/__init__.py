"""
Orchestrator package for HADES-PathRAG.

This package contains the main orchestration components for
the ingestion pipeline, coordinating file processing, preprocessing,
embedding, and storage.
"""

from .ingestor import RepositoryIngestor

__all__ = [
    'RepositoryIngestor'
]

"""HADES-PathRAG Data Ingestion Pipeline.

This package provides the ingestion pipeline for processing and storing code
repositories into a unified knowledge graph with structural relationships
and semantic embeddings.
"""

from __future__ import annotations

from .orchestrator.ingestor import RepositoryIngestor

__all__ = ["RepositoryIngestor"]

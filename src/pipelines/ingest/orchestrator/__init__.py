"""Ingestion pipeline orchestration components.

This package contains the main orchestration components for the ingestion pipeline,
coordinating file processing, preprocessing, embedding, and storage.
"""

from __future__ import annotations

from .ingestor import RepositoryIngestor

__all__ = ["RepositoryIngestor"]

"""
Data ingestion pipeline for HADES-PathRAG.

This package provides functionality for loading data, building graphs,
computing embeddings using ISNE, and storing results in ArangoDB.
"""

from hades_pathrag.ingestion.pipeline import (
    IngestDataset,
    IngestDocument,
    IngestionPipeline,
    ISNEEmbeddingProcessor,
)

__all__ = [
    "IngestDataset",
    "IngestDocument",
    "IngestionPipeline",
    "ISNEEmbeddingProcessor",
]

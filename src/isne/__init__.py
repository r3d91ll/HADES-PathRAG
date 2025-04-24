"""
ISNE (Inductive Shallow Node Embedding) pipeline for HADES-PathRAG.

This module provides a comprehensive pipeline for processing and embedding
documents using Inductive Shallow Node Embedding technology, which creates
graph-based embeddings that capture relationships between documents.
"""

from src.isne.types.models import (
    IngestDocument,
    IngestDataset,
    DocumentRelation,
    RelationType,
    EmbeddingVector,
    EmbeddingConfig,
    ISNEConfig
)

from src.isne.models.isne_model import ISNEModel
from src.isne.layers.isne_layer import ISNELayer, ISNEFeaturePropagation

from src.isne.loaders.base_loader import BaseLoader, LoaderResult, LoaderConfig
from src.isne.loaders.text_directory_loader import TextDirectoryLoader
from src.isne.loaders.json_loader import JSONLoader
from src.isne.loaders.csv_loader import CSVLoader

from src.isne.processors.base_processor import BaseProcessor, ProcessorResult, ProcessorConfig
from src.isne.processors.embedding_processor import EmbeddingProcessor
from src.isne.processors.graph_processor import GraphProcessor
from src.isne.processors.chunking_processor import ChunkingProcessor

from src.isne.pipeline import ISNEPipeline, PipelineConfig

__all__ = [
    # Data models
    "IngestDocument",
    "IngestDataset",
    "DocumentRelation",
    "RelationType",
    "EmbeddingVector",
    "EmbeddingConfig",
    "ISNEConfig",
    
    # Neural network components
    "ISNEModel",
    "ISNELayer",
    "ISNEFeaturePropagation",
    
    # Loaders
    "BaseLoader",
    "LoaderResult",
    "LoaderConfig",
    "TextDirectoryLoader",
    "JSONLoader",
    "CSVLoader",
    
    # Processors
    "BaseProcessor",
    "ProcessorResult",
    "ProcessorConfig",
    "EmbeddingProcessor",
    "GraphProcessor",
    "ChunkingProcessor",
    
    # Pipeline
    "ISNEPipeline",
    "PipelineConfig"
]

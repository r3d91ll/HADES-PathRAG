"""
Batch types module.

This module defines the data structures used to pass batches between pipeline stages.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
import numpy as np


@dataclass
class DocumentBatch:
    """A batch of documents to be processed."""
    
    docs: List[Dict[str, Any]]
    batch_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ChunkBatch:
    """A batch of chunks produced by the chunking stage."""
    
    chunks: List[Dict[str, Any]]
    batch_id: str
    parent_batch_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EmbeddingBatch:
    """A batch of embeddings produced by the embedding stage."""
    
    embeddings: List[np.ndarray]
    chunk_ids: List[str]
    batch_id: str
    parent_batch_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ISNEBatch:
    """A batch of ISNE graph embeddings."""
    
    graph_embeddings: Dict[str, np.ndarray]
    relations: List[Dict[str, Any]]
    batch_id: str
    parent_batch_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineBatch:
    """
    A unified batch that flows through the pipeline.
    
    This class contains all data produced by each stage of the pipeline.
    Each stage updates the relevant field with its output.
    """
    
    # Batch identification
    batch_id: str
    
    # Document processing stage output
    docs: List[Dict[str, Any]]
    
    # Chunking stage output
    chunks: Optional[List[Dict[str, Any]]] = None
    
    # Embedding stage output
    embeddings: Optional[Dict[str, np.ndarray]] = None
    
    # ISNE stage output
    graph: Optional[Dict[str, Any]] = None
    
    # Metadata for tracking and debugging
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Error information
    errors: List[Dict[str, Any]] = field(default_factory=list)
    
    def add_error(self, stage: str, error: Exception, details: Optional[Dict[str, Any]] = None) -> None:
        """
        Add an error to the batch.
        
        Args:
            stage: The pipeline stage where the error occurred
            error: The exception that was raised
            details: Additional details about the error
        """
        self.errors.append({
            "stage": stage,
            "error": str(error),
            "error_type": type(error).__name__,
            "details": details or {},
        })
    
    @property
    def has_errors(self) -> bool:
        """Check if the batch has any errors."""
        return len(self.errors) > 0
    
    def get_stage_errors(self, stage: str) -> List[Dict[str, Any]]:
        """Get all errors for a specific stage."""
        return [e for e in self.errors if e["stage"] == stage]

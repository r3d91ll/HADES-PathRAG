"""Configuration for the ingestion orchestrator.

This module contains configuration settings for the repository ingestor,
including device settings, batch sizes, and performance options.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Literal

# Default environment variable for controlling device usage
DEFAULT_DEVICE_ENV_VAR = "HADES_DEFAULT_DEVICE"
DEFAULT_DEVICE = os.environ.get(DEFAULT_DEVICE_ENV_VAR, "cpu")

# Chunking specific configuration
@dataclass
class ChunkingConfig:
    """Configuration for the chunking process."""
    
    # Whether to use GPU acceleration for chunking
    use_gpu: bool = False
    
    # Device to use for chunking (cpu, cuda:0, etc.)
    device: str = "cpu"
    
    # Maximum tokens per chunk
    max_tokens: int = 2048
    
    # Model ID for semantic chunking
    model_id: str = "mirth/chonky_modernbert_large_1"
    
    # Whether to use semantic chunking or basic chunking
    use_semantic_chunking: bool = True
    
    # Number of workers for parallel processing when using CPU
    num_cpu_workers: int = 4
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "use_gpu": self.use_gpu,
            "device": self.device,
            "max_tokens": self.max_tokens,
            "model_id": self.model_id,
            "use_semantic_chunking": self.use_semantic_chunking,
            "num_cpu_workers": self.num_cpu_workers,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ChunkingConfig:
        """Create from dictionary."""
        return cls(
            use_gpu=data.get("use_gpu", False),
            device=data.get("device", "cpu"),
            max_tokens=data.get("max_tokens", 2048),
            model_id=data.get("model_id", "mirth/chonky_modernbert_large_1"),
            use_semantic_chunking=data.get("use_semantic_chunking", True),
            num_cpu_workers=data.get("num_cpu_workers", 4),
        )


# Embedding specific configuration
@dataclass
class EmbeddingConfig:
    """Configuration for the embedding process."""
    
    # Whether to use GPU acceleration for embeddings
    use_gpu: bool = False
    
    # Device to use for embeddings (cpu, cuda:0, etc.)
    device: str = "cpu"
    
    # Embedding adapter to use (vllm, openai, etc.)
    adapter: str = "vllm"
    
    # Embedding model ID
    model_id: str = "thenlper/gte-large"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "use_gpu": self.use_gpu,
            "device": self.device,
            "adapter": self.adapter,
            "model_id": self.model_id,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> EmbeddingConfig:
        """Create from dictionary."""
        return cls(
            use_gpu=data.get("use_gpu", False),
            device=data.get("device", "cpu"),
            adapter=data.get("adapter", "vllm"),
            model_id=data.get("model_id", "thenlper/gte-large"),
        )


# Main ingestion configuration
@dataclass
class IngestionConfig:
    """Configuration for the ingestion process."""
    
    # Chunking configuration
    chunking: ChunkingConfig = field(default_factory=lambda: ChunkingConfig())
    
    # Embedding configuration
    embedding: EmbeddingConfig = field(default_factory=lambda: EmbeddingConfig())
    
    # Batch size for processing
    batch_size: int = 32
    
    # Maximum concurrency
    max_concurrency: int = 8
    
    # Whether to initialize the database
    initialize_db: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "chunking": self.chunking.to_dict(),
            "embedding": self.embedding.to_dict(),
            "batch_size": self.batch_size,
            "max_concurrency": self.max_concurrency,
            "initialize_db": self.initialize_db,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> IngestionConfig:
        """Create from dictionary."""
        return cls(
            chunking=ChunkingConfig.from_dict(data.get("chunking", {})),
            embedding=EmbeddingConfig.from_dict(data.get("embedding", {})),
            batch_size=data.get("batch_size", 32),
            max_concurrency=data.get("max_concurrency", 8),
            initialize_db=data.get("initialize_db", False),
        )
    
    @classmethod
    def create_cpu_only(cls) -> IngestionConfig:
        """Create a CPU-only configuration."""
        return cls(
            chunking=ChunkingConfig(use_gpu=False, device="cpu"),
            embedding=EmbeddingConfig(use_gpu=False, device="cpu"),
        )
    
    @classmethod
    def create_gpu_enabled(cls, device: str = "cuda:0") -> IngestionConfig:
        """Create a GPU-enabled configuration."""
        return cls(
            chunking=ChunkingConfig(use_gpu=True, device=device),
            embedding=EmbeddingConfig(use_gpu=True, device=device),
        )
    
    @classmethod
    def from_env(cls) -> IngestionConfig:
        """Create configuration based on environment variables."""
        device = os.environ.get(DEFAULT_DEVICE_ENV_VAR, "cpu")
        use_gpu = device.startswith("cuda")
        
        return cls(
            chunking=ChunkingConfig(use_gpu=use_gpu, device=device),
            embedding=EmbeddingConfig(use_gpu=use_gpu, device=device),
        )


# Default configuration
DEFAULT_CONFIG = IngestionConfig.from_env()

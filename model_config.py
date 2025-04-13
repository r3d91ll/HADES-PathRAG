#!/usr/bin/env python
"""Model configuration for HADES-PathRAG.

This script configures the Hybrid Embedding System that combines ModernBERT
semantic embeddings with ISNE structural embeddings for PathRAG.
"""
import os
import logging
from pathlib import Path
from typing import Optional

import torch

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Global flag to determine if ISNE is available
ISNE_AVAILABLE = True

from hades_pathrag.ingestion.enhanced_embeddings import HybridEmbeddingProcessor
from hades_pathrag.embeddings.semantic import ModernBERTEmbedder

# Try to import ISNE but handle case where dependencies aren't available
try:
    from hades_pathrag.ingestion.embeddings import ISNEEmbeddingProcessor
except ImportError:
    logger.warning("ISNE dependencies not available. Will use semantic-only embeddings.")
    ISNE_AVAILABLE = False
    
from hades_pathrag.core.config import PathRAGConfig


def setup_models(
    semantic_weight: float = 0.5,
    isne_embedding_dim: int = 128,
    semantic_embedding_dim: int = 768,
    final_embedding_dim: Optional[int] = None,
    device: Optional[str] = None,
    timeout: int = 30  # Add timeout parameter to prevent hanging
) -> HybridEmbeddingProcessor:
    """
    Set up the hybrid embedding processor with ModernBERT and ISNE.
    
    Args:
        semantic_weight: Weight of semantic embeddings in final combination
        isne_embedding_dim: Dimension of ISNE embeddings
        semantic_embedding_dim: Dimension of semantic embeddings 
        final_embedding_dim: Dimension of final combined embeddings
        device: Device to run models on (cpu, cuda, mps)
        timeout: Timeout in seconds for model loading
        
    Returns:
        Configured HybridEmbeddingProcessor
    """
    # Determine if CUDA is available for GPU operations
    if device is None:
        cuda_available = torch.cuda.is_available()
        gpu_count = torch.cuda.device_count() if cuda_available else 0
        logger.info(f"CUDA available: {cuda_available}, GPU count: {gpu_count}")
        device = "cpu"  # Force CPU to avoid potential CUDA issues
        logger.info(f"Using CPU for model loading to avoid timeouts")
    
    # Create the hybrid embedding processor first (this doesn't load any models yet)
    logger.info("Creating hybrid embedding processor structure...")
    processor = HybridEmbeddingProcessor(
        semantic_weight=semantic_weight,
        isne_embedding_dim=isne_embedding_dim,
        semantic_embedding_dim=semantic_embedding_dim,
        final_embedding_dim=final_embedding_dim,
        device=device,
        weight_threshold=0.5
    )
    
    # Create the semantic embedder (ModernBERT) with error handling
    logger.info("Initializing ModernBERT embedder...")
    try:
        cache_dir = os.path.join(str(Path.home()), ".cache", "hades_pathrag", "models")
        os.makedirs(cache_dir, exist_ok=True)  # Ensure cache directory exists
        
        semantic_embedder = ModernBERTEmbedder(
            embedding_dim=semantic_embedding_dim,
            max_length=512,  # Standard context length
            device=device,
            cache_dir=cache_dir
        )
        processor.semantic_embedder = semantic_embedder
        logger.info("ModernBERT embedder initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize ModernBERT embedder: {e}")
        logger.warning("Using fallback for semantic embedder")
        # Create a simple fallback embedder that returns zeros
        # This ensures the system doesn't crash if the transformer model fails to load
        from hades_pathrag.embeddings.fallback import FallbackEmbedder
        processor.semantic_embedder = FallbackEmbedder(dim=semantic_embedding_dim)
    
    # Create the structural embedder (ISNE) with error handling
    if ISNE_AVAILABLE:
        logger.info("Initializing ISNE embedder...")
        try:
            isne_processor = ISNEEmbeddingProcessor(
                embedding_dim=isne_embedding_dim,
                device=device
            )
            processor.isne_processor = isne_processor
            logger.info("ISNE embedder initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize ISNE embedder: {e}")
            logger.warning("ISNE will not be available")
    else:
        logger.warning("ISNE not available due to missing dependencies")
    
    logger.info(f"Hybrid embedding processor initialized on {device}")
    logger.info(f"Semantic weight: {semantic_weight}, ISNE weight: {1.0 - semantic_weight}")
    
    return processor


if __name__ == "__main__":
    # When run directly, set up models and print status
    processor = setup_models()
    logger.info("Model setup complete!")

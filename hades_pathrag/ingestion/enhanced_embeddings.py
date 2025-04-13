"""
Enhanced embedding processor that combines semantic and structural embeddings.

This module extends the basic ISNE embedding processor to incorporate
semantic embeddings from ModernBERT models, creating a hybrid approach.
"""
import logging
from typing import Dict, List, Optional, Tuple, Any, Union, cast

import numpy as np
import torch
from torch import Tensor

from hades_pathrag.ingestion.models import IngestDataset, IngestDocument
from hades_pathrag.ingestion.embeddings import ISNEEmbeddingProcessor
from hades_pathrag.embeddings.semantic import ModernBERTEmbedder

logger = logging.getLogger(__name__)


class HybridEmbeddingProcessor:
    """
    Enhanced embedding processor that combines semantic and structural embeddings.
    
    This processor uses ModernBERT for semantic understanding and ISNE for
    structural understanding, combining them into unified embeddings.
    """
    
    def __init__(
        self,
        semantic_weight: float = 0.5,
        isne_embedding_dim: int = 128,
        semantic_embedding_dim: int = 768,
        final_embedding_dim: Optional[int] = None,
        device: Optional[str] = None,
        weight_threshold: float = 0.5,
    ) -> None:
        """
        Initialize the hybrid embedding processor.
        
        Args:
            semantic_weight: Weight of semantic embeddings in the final combination (0-1)
            isne_embedding_dim: Dimension of ISNE embeddings
            semantic_embedding_dim: Dimension of semantic embeddings
            final_embedding_dim: Dimension of final combined embeddings
            device: Device to run models on (cpu, cuda, mps)
            weight_threshold: Minimum weight for a relationship in ISNE
        """
        self.semantic_weight = max(0.0, min(1.0, semantic_weight))  # Clamp to [0, 1]
        self.isne_weight = 1.0 - self.semantic_weight
        
        self.isne_embedding_dim = isne_embedding_dim
        self.semantic_embedding_dim = semantic_embedding_dim
        self.final_embedding_dim = final_embedding_dim or max(isne_embedding_dim, semantic_embedding_dim)
        
        # Initialize the component processors
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.isne_processor = ISNEEmbeddingProcessor(
            embedding_dim=isne_embedding_dim,
            device=self.device,
            weight_threshold=weight_threshold
        )
        self.semantic_embedder = ModernBERTEmbedder(
            embedding_dim=semantic_embedding_dim,
            device=self.device
        )
        
        logger.info("Initialized hybrid embedding processor")
        logger.info(f"Semantic weight: {self.semantic_weight}, ISNE weight: {self.isne_weight}")
        logger.info(f"Final embedding dimension: {self.final_embedding_dim}")
    
    def _combine_embeddings(
        self, 
        semantic_embedding: Optional[List[float]], 
        isne_embedding: Optional[List[float]]
    ) -> List[float]:
        """
        Combine semantic and structural embeddings.
        
        Args:
            semantic_embedding: Semantic embedding vector
            isne_embedding: ISNE structural embedding vector
            
        Returns:
            Combined embedding vector
        """
        # Handle missing embeddings
        if semantic_embedding is None and isne_embedding is None:
            # If both are missing, return zeros
            return [0.0] * self.final_embedding_dim
        elif semantic_embedding is None:
            # If only semantic is missing, use normalized ISNE
            isne_embedding = cast(List[float], isne_embedding)
            isne_array = np.array(isne_embedding)
            normalized = isne_array / (np.linalg.norm(isne_array) + 1e-6)
            isne_resized: np.ndarray = self._resize_vector(normalized, self.final_embedding_dim)
            return cast(List[float], isne_resized.tolist())
        elif isne_embedding is None:
            # If only ISNE is missing, use normalized semantic
            semantic_array = np.array(semantic_embedding)
            normalized = semantic_array / (np.linalg.norm(semantic_array) + 1e-6)
            semantic_resized: np.ndarray = self._resize_vector(normalized, self.final_embedding_dim)
            return cast(List[float], semantic_resized.tolist())
        
        # Both embeddings present - combine with weights
        semantic_array = np.array(semantic_embedding)
        isne_array = np.array(isne_embedding)
        
        # Normalize both embeddings
        semantic_norm = np.linalg.norm(semantic_array) + 1e-6
        isne_norm = np.linalg.norm(isne_array) + 1e-6
        
        semantic_normalized = semantic_array / semantic_norm
        isne_normalized = isne_array / isne_norm
        
        # Resize to target dimension if needed
        semantic_resized = self._resize_vector(semantic_normalized, self.final_embedding_dim)
        isne_resized = self._resize_vector(isne_normalized, self.final_embedding_dim)
        
        # Weighted combination
        combined = (
            self.semantic_weight * semantic_resized + 
            self.isne_weight * isne_resized
        )
        
        # Final normalization
        combined_norm = np.linalg.norm(combined) + 1e-6
        normalized_combined = combined / combined_norm
        
        return cast(List[float], normalized_combined.tolist())
    
    def _resize_vector(self, vector: np.ndarray, target_dim: int) -> np.ndarray:
        """
        Resize a vector to a target dimension.
        
        If target_dim > original_dim: Pad with zeros
        If target_dim < original_dim: Truncate or use dimensionality reduction
        
        Args:
            vector: Vector to resize
            target_dim: Target dimension
            
        Returns:
            Resized vector
        """
        current_dim = vector.shape[0]
        
        if current_dim == target_dim:
            return vector
        elif current_dim > target_dim:
            # Truncate to smaller dimension
            return vector[:target_dim]
        else:
            # Pad with zeros
            result = np.zeros(target_dim)
            result[:current_dim] = vector
            return result
    
    def process(self, dataset: IngestDataset) -> IngestDataset:
        """
        Process a dataset to compute and add hybrid embeddings.
        
        Args:
            dataset: The dataset to process
            
        Returns:
            The processed dataset with hybrid embeddings
        """
        logger.info(f"Processing dataset {dataset.name} with hybrid embedder...")
        
        # Step 1: Compute semantic embeddings
        logger.info("Computing semantic embeddings...")
        for doc in dataset.documents:
            # Use proper type annotations for the semantic embeddings
            semantic_result: Optional[List[float]] = self.semantic_embedder.embed(doc)
            # Store with proper type safety
            setattr(doc, "semantic_embedding", semantic_result)
        
        # Step 2: Compute ISNE structural embeddings
        logger.info("Computing ISNE structural embeddings...")
        dataset = self.isne_processor.process(dataset)
        
        # Step 3: Combine embeddings
        logger.info("Combining semantic and structural embeddings...")
        for doc in dataset.documents:
            # Get embeddings with proper type handling
            semantic_embed: Optional[List[float]] = cast(Optional[List[float]], getattr(doc, "semantic_embedding", None))
            isne_embed: Optional[List[float]] = doc.embedding
            
            # Combine embeddings with proper type handling
            doc.embedding = self._combine_embeddings(
                semantic_embed,
                isne_embed
            )
            
            # Clean up intermediate embeddings to save memory
            if hasattr(doc, "semantic_embedding"):
                delattr(doc, "semantic_embedding")
        
        logger.info(f"Computed hybrid embeddings for {len(dataset.documents)} documents.")
        return dataset
    
    def embed_document(self, document: IngestDocument) -> IngestDocument:
        """
        Embed a single document using the hybrid approach.
        
        This is useful for embedding new documents after initial processing.
        
        Args:
            document: The document to embed
            
        Returns:
            Document with hybrid embedding added
        """
        # Compute semantic embedding with proper type annotation
        semantic_embedding: Optional[List[float]] = self.semantic_embedder.embed(document)
        
        # For structural embedding, we need graph context
        # Here we use an approximation or direct connection to existing nodes
        # This is complex in practice and may require specialized logic
        isne_embedding: Optional[List[float]] = None  # This would be computed based on relationships
        
        # Combine embeddings with proper type handling
        document.embedding = self._combine_embeddings(
            semantic_embedding,
            isne_embedding
        )
        
        return document

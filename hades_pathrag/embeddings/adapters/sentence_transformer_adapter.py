"""
Adapter for SentenceTransformer embedding models.

This adapter wraps the SentenceTransformer implementation and makes it compatible
with the HADES-PathRAG BaseEmbedder interface.
"""
from typing import Dict, List, Optional, Any, Union, Tuple, cast
import logging
import os
import pickle
from pathlib import Path
import json

import numpy as np

# Import BaseEmbedder and other interfaces
from hades_pathrag.embeddings.base import BaseEmbedder
from hades_pathrag.embeddings.interfaces import EmbeddingStats

logger = logging.getLogger(__name__)

# Try to import SentenceTransformer, but don't fail if not available
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMER_AVAILABLE = True
except ImportError:
    logger.debug("sentence_transformers package not available")
    SENTENCE_TRANSFORMER_AVAILABLE = False
    # Create dummy class for type checking
    class SentenceTransformer:
        pass


class SentenceTransformerAdapter(BaseEmbedder):
    """
    Adapter for SentenceTransformer embedding models.
    
    This class wraps SentenceTransformer from the sentence_transformers package
    and adapts it to the BaseEmbedder interface.
    """
    
    def __init__(
        self, 
        model_name: str = "all-MiniLM-L6-v2",
        embedding_dim: Optional[int] = None,
        device: Optional[str] = None,
        cache_folder: Optional[str] = None,
        **kwargs: Any
    ) -> None:
        """
        Initialize the SentenceTransformer adapter.
        
        Args:
            model_name: Name of the pretrained model to use
            embedding_dim: Dimension of embeddings (inferred from model if None)
            device: Device to use (cuda, cpu, mps)
            cache_folder: Folder to cache models
            **kwargs: Additional keyword arguments
        """
        if not SENTENCE_TRANSFORMER_AVAILABLE:
            raise ImportError("sentence_transformers package is required but not installed")
        
        self.model_name = model_name
        
        # Initialize model with optional cache folder
        model_kwargs = {}
        if cache_folder:
            model_kwargs["cache_folder"] = cache_folder
        
        if device:
            model_kwargs["device"] = device
            
        self._model = SentenceTransformer(model_name, **model_kwargs)
        
        # Get embedding dimension from model
        self.embedding_dim = embedding_dim or self._model.get_sentence_embedding_dimension()
        
        # Node ID to text mapping
        self._node_texts: Dict[str, str] = {}
        self._embeddings: Dict[str, np.ndarray] = {}
    
    @property
    def name(self) -> str:
        """Get the name of the embedder."""
        return f"sentence-transformer-{self.model_name}"
    
    def add_texts(self, texts_dict: Dict[str, str]) -> None:
        """
        Add texts to the model.
        
        Args:
            texts_dict: Dictionary mapping node IDs to text content
        """
        # Store text mapping
        self._node_texts.update(texts_dict)
        
        # Batch encode all texts
        if texts_dict:
            nodes = list(texts_dict.keys())
            texts = list(texts_dict.values())
            
            # Encode in batches to avoid memory issues
            batch_size = 32
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                batch_nodes = nodes[i:i+batch_size]
                
                embeddings = self._model.encode(batch_texts)
                
                for node, embedding in zip(batch_nodes, embeddings):
                    self._embeddings[node] = embedding
    
    def encode(self, node_id: str, neighbors: List[str]) -> np.ndarray:
        """
        Encode a node with its neighbors.
        
        Args:
            node_id: ID of the node to encode
            neighbors: List of neighbor node IDs
            
        Returns:
            Embedding vector for the node
        """
        # Check if node already has an embedding
        if node_id in self._embeddings:
            return self._embeddings[node_id]
        
        # If node has text, encode it
        if node_id in self._node_texts:
            text = self._node_texts[node_id]
            embedding = self._model.encode(text)
            self._embeddings[node_id] = embedding
            return embedding
        
        # If node doesn't have text but neighbors do, average neighbor embeddings
        neighbor_embeddings = []
        for neighbor in neighbors:
            if neighbor in self._embeddings:
                neighbor_embeddings.append(self._embeddings[neighbor])
        
        if neighbor_embeddings:
            avg_embedding = np.mean(neighbor_embeddings, axis=0)
            self._embeddings[node_id] = avg_embedding
            return avg_embedding
        
        # Fallback to zeros
        return np.zeros(self.embedding_dim)
    
    def get_embedding(self, node_id: str) -> Optional[np.ndarray]:
        """
        Get the embedding for a node.
        
        Args:
            node_id: ID of the node
            
        Returns:
            Embedding vector if available, None otherwise
        """
        return self._embeddings.get(node_id)
    
    def encode_text(self, text: str) -> np.ndarray:
        """
        Encode a text string directly.
        
        Args:
            text: Text to encode
            
        Returns:
            Embedding vector
        """
        return self._model.encode(text)
    
    def get_similarities(self, query_embedding: np.ndarray, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Get most similar nodes to the query embedding.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of similar nodes to return
            
        Returns:
            List of (node_id, similarity) tuples
        """
        if not self._embeddings:
            return []
        
        # Compute cosine similarities
        similarities = []
        for node_id, embedding in self._embeddings.items():
            sim = self._cosine_similarity(query_embedding, embedding)
            similarities.append((node_id, sim))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return np.dot(a, b) / (norm_a * norm_b)
    
    def get_stats(self) -> EmbeddingStats:
        """
        Get statistics about the embeddings.
        
        Returns:
            Embedding statistics
        """
        return {
            "num_embeddings": len(self._embeddings),
            "embedding_dim": self.embedding_dim,
            "model_type": f"sentence-transformer-{self.model_name}",
            "average_norm": np.mean([np.linalg.norm(emb) for emb in self._embeddings.values()]) if self._embeddings else 0.0
        }
    
    def save(self, path: str) -> None:
        """
        Save the model to a file.
        
        Args:
            path: Path to save the model to
        """
        # Create directory if it doesn't exist
        save_dir = Path(path)
        os.makedirs(save_dir.parent, exist_ok=True)
        
        # Save embeddings and text mappings
        model_data = {
            "model_name": self.model_name,
            "embedding_dim": self.embedding_dim
        }
        
        # Save model metadata
        with open(str(save_dir) + ".json", "w") as f:
            json.dump(model_data, f)
        
        # Save embeddings
        with open(str(save_dir) + ".embeddings", "wb") as f:
            pickle.dump({
                "embeddings": self._embeddings,
                "node_texts": self._node_texts
            }, f)
    
    @classmethod
    def load(cls, path: str) -> "SentenceTransformerAdapter":
        """
        Load a model from a file.
        
        Args:
            path: Path to load the model from
            
        Returns:
            Loaded model
        """
        # Load model metadata
        with open(path + ".json", "r") as f:
            model_data = json.load(f)
        
        # Create instance
        instance = cls(
            model_name=model_data["model_name"],
            embedding_dim=model_data["embedding_dim"]
        )
        
        # Load embeddings and texts
        with open(path + ".embeddings", "rb") as f:
            data = pickle.load(f)
            instance._embeddings = data["embeddings"]
            instance._node_texts = data["node_texts"]
        
        return instance

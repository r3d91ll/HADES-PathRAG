"""
Adapter for Node2Vec embedding models.

This adapter wraps the Node2Vec implementation and makes it compatible
with the HADES-PathRAG BaseEmbedder interface.
"""
from typing import Dict, List, Optional, Any, Union, Tuple, cast
import logging
import os
import pickle
from pathlib import Path

import numpy as np
import networkx as nx
from gensim.models import KeyedVectors

# Import BaseEmbedder and other interfaces
from hades_pathrag.embeddings.base import BaseEmbedder
from hades_pathrag.embeddings.interfaces import EmbeddingStats

logger = logging.getLogger(__name__)

# Try to import Node2Vec, but don't fail if not available
try:
    from node2vec import Node2Vec
    NODE2VEC_AVAILABLE = True
except ImportError:
    logger.debug("node2vec package not available")
    NODE2VEC_AVAILABLE = False
    # Create dummy class for type checking
    class Node2Vec:
        pass


class Node2VecAdapter(BaseEmbedder):
    """
    Adapter for Node2Vec embedding models.
    
    This class wraps Node2Vec from the node2vec package and adapts it
    to the BaseEmbedder interface.
    """
    
    def __init__(
        self, 
        embedding_dim: int = 128,
        walk_length: int = 80,
        num_walks: int = 10,
        p: float = 1.0,
        q: float = 1.0,
        workers: int = 1,
        window: int = 10,
        min_count: int = 1,
        batch_words: int = 4,
        epochs: int = 5,
        **kwargs: Any
    ) -> None:
        """
        Initialize the Node2Vec adapter.
        
        Args:
            embedding_dim: Dimension of embeddings
            walk_length: Length of each random walk
            num_walks: Number of random walks per node
            p: Return parameter (1 = neutral)
            q: In-out parameter (1 = neutral)
            workers: Number of workers for parallel processing
            window: Window size for word2vec
            min_count: Minimum count for word2vec
            batch_words: Batch size for word2vec
            epochs: Number of epochs to train
            **kwargs: Additional keyword arguments
        """
        if not NODE2VEC_AVAILABLE:
            raise ImportError("node2vec package is required but not installed")
        
        self.embedding_dim = embedding_dim
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.p = p
        self.q = q
        self.workers = workers
        self.window = window
        self.min_count = min_count
        self.batch_words = batch_words
        self.epochs = epochs
        
        self._model = None
        self._embeddings: Dict[str, np.ndarray] = {}
        self._graph = nx.Graph()
    
    @property
    def name(self) -> str:
        """Get the name of the embedder."""
        return "node2vec"
    
    def fit(self, graph: nx.Graph) -> None:
        """
        Fit the model to the graph.
        
        Args:
            graph: NetworkX graph to fit the model to
        """
        self._graph = graph
        
        # Initialize Node2Vec model
        node2vec = Node2Vec(
            graph=graph,
            dimensions=self.embedding_dim,
            walk_length=self.walk_length,
            num_walks=self.num_walks,
            p=self.p,
            q=self.q,
            workers=self.workers
        )
        
        # Train model
        model = node2vec.fit(
            window=self.window,
            min_count=self.min_count,
            batch_words=self.batch_words,
            epochs=self.epochs
        )
        
        self._model = model
        
        # Store embeddings
        for node in graph.nodes():
            node_str = str(node)
            if node_str in model.wv:
                self._embeddings[node_str] = model.wv[node_str]
    
    def encode(self, node_id: str, neighbors: List[str]) -> np.ndarray:
        """
        Encode a node with its neighbors.
        
        Args:
            node_id: ID of the node to encode
            neighbors: List of neighbor node IDs
            
        Returns:
            Embedding vector for the node
        """
        if self._model is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        if node_id in self._embeddings:
            return self._embeddings[node_id]
        
        # If node not in embeddings, try to infer based on neighbors
        if neighbors and any(n in self._embeddings for n in neighbors):
            # Average embeddings of neighbors
            neighbor_embeddings = [
                self._embeddings[n] for n in neighbors if n in self._embeddings
            ]
            if neighbor_embeddings:
                return np.mean(neighbor_embeddings, axis=0)
        
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
            "model_type": "node2vec",
            "average_norm": np.mean([np.linalg.norm(emb) for emb in self._embeddings.values()]) if self._embeddings else 0.0
        }
    
    def save(self, path: str) -> None:
        """
        Save the model to a file.
        
        Args:
            path: Path to save the model to
        """
        if self._model is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        
        # Save model and embeddings
        with open(path, "wb") as f:
            pickle.dump({
                "embeddings": self._embeddings,
                "params": {
                    "embedding_dim": self.embedding_dim,
                    "walk_length": self.walk_length,
                    "num_walks": self.num_walks,
                    "p": self.p,
                    "q": self.q,
                    "workers": self.workers,
                    "window": self.window,
                    "min_count": self.min_count,
                    "batch_words": self.batch_words,
                    "epochs": self.epochs
                }
            }, f)
    
    @classmethod
    def load(cls, path: str) -> "Node2VecAdapter":
        """
        Load a model from a file.
        
        Args:
            path: Path to load the model from
            
        Returns:
            Loaded model
        """
        with open(path, "rb") as f:
            data = pickle.load(f)
        
        # Create instance with saved parameters
        instance = cls(**data["params"])
        
        # Load embeddings
        instance._embeddings = data["embeddings"]
        
        return instance

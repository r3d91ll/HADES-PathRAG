"""
Adapter for Node2Vec embedding models.

This adapter wraps the Node2Vec implementation and makes it compatible
with the HADES-PathRAG BaseEmbedder interface.
"""
from typing import Dict, List, Optional, Any, Union, Tuple, cast, Type
import logging
import os
import pickle
from pathlib import Path

import numpy as np
import networkx as nx
from .gensim_models import KeyedVectors

# Import BaseEmbedder and other interfaces
from hades_pathrag.embeddings.base import BaseEmbedder
from hades_pathrag.embeddings.interfaces import EmbeddingStats
from hades_pathrag.typings import (
    EmbeddingArray, NodeIDType, Graph, EmbeddingDict, PathType
)

logger = logging.getLogger(__name__)

# Try to import Node2Vec, but don't fail if not available
try:
    from node2vec import Node2Vec  # type: ignore
    NODE2VEC_AVAILABLE = True
except ImportError:
    logger.debug("node2vec package not available")
    NODE2VEC_AVAILABLE = False
    # Create dummy class for type checking
    class DummyNode2Vec:
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
        self._embeddings: EmbeddingDict = {}
        self._graph: Graph = nx.Graph()
    
    @property
    def name(self) -> str:
        """Get the name of the embedder."""
        return "node2vec"
    
    def fit(self, graph: Graph) -> None:
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
    
    def encode(self, node_id: NodeIDType, neighbors: List[NodeIDType]) -> EmbeddingArray:
        """
        Encode a node with its neighbors.
        
        Args:
            node_id: ID of the node to encode
            neighbors: List of neighbor node IDs
            
        Returns:
            Embedding vector for the node
        """
        # Implemented with separate methods to help type checker
        return self._encode_impl(node_id, neighbors)
        
    def _encode_impl(self, node_id: NodeIDType, neighbors: List[NodeIDType]) -> EmbeddingArray:
        """Implementation of encode logic to avoid mypy unreachable code errors."""
        if self._model is None:
            # Handle missing model with fallback
            return np.zeros(self.embedding_dim, dtype=np.float32)
            
        # Check if node embedding is already computed
        existing_embedding = self._embeddings.get(node_id)  # type: ignore[unreachable]
        if existing_embedding is not None:
            return np.asarray(existing_embedding, dtype=np.float32)
        
        # If node not in embeddings, try to infer based on neighbors
        if neighbors:
            valid_neighbors = [n for n in neighbors if n in self._embeddings]
            if valid_neighbors:
                # Average embeddings of neighbors
                neighbor_embeddings = [
                    np.asarray(self._embeddings[n], dtype=np.float32) 
                    for n in valid_neighbors
                ]
                avg_embedding = np.mean(neighbor_embeddings, axis=0)
                result = np.asarray(avg_embedding, dtype=np.float32)
                self._embeddings[node_id] = result
                return result
        
        # Fallback to zeros
        fallback = np.zeros(self.embedding_dim, dtype=np.float32)
        return fallback
    
    def get_embedding(self, node_id: NodeIDType) -> Optional[EmbeddingArray]:
        """
        Get the embedding for a node.
        
        Args:
            node_id: ID of the node
            
        Returns:
            Embedding vector if available, None otherwise
        """
        return self._embeddings.get(node_id)
    
    def get_similarities(self, query_embedding: EmbeddingArray, top_k: int = 10) -> List[Tuple[NodeIDType, float]]:
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
    
    def _cosine_similarity(self, a: EmbeddingArray, b: EmbeddingArray) -> float:
        """Compute cosine similarity between two vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return float(np.dot(a, b) / (norm_a * norm_b))  # type: ignore[return-value]
    
    def batch_encode(self, nodes: Union[List[Tuple[NodeIDType, List[NodeIDType], Optional[str]]], Tuple[List[NodeIDType], List[List[NodeIDType]]]]) -> EmbeddingArray:
        """
        Encode multiple nodes in a batch.
        
        Args:
            nodes: Either a list of (node_id, neighbors, text) tuples or a tuple of (node_ids, neighbor_lists)
            
        Returns:
            Matrix of node embeddings
        """
        # Delegate to implementation method to make type checking cleaner
        return self._batch_encode_impl(nodes)
        
    def _batch_encode_impl(self, nodes: Union[List[Tuple[NodeIDType, List[NodeIDType], Optional[str]]], Tuple[List[NodeIDType], List[List[NodeIDType]]]]) -> EmbeddingArray:
        """Implementation helper for batch_encode to avoid mypy unreachable code errors."""
        if self._model is None:
            # Return empty array with correct dimension rather than raising an exception
            return np.zeros((0, self.embedding_dim), dtype=np.float32)
        
        # Process input according to its type
        node_ids: List[NodeIDType] = []  # type: ignore[unreachable]
        neighbor_lists: List[List[NodeIDType]] = []
        
        if isinstance(nodes, tuple):
            # Handle tuple format: (node_ids, neighbor_lists)
            nodes_tuple = cast(Tuple[List[NodeIDType], List[List[NodeIDType]]], nodes)
            if len(nodes_tuple) != 2:
                raise ValueError("Tuple input must have exactly 2 elements: node_ids and neighbor_lists")
            node_ids = nodes_tuple[0]
            neighbor_lists = nodes_tuple[1]
        else:
            # Handle list format: [(node_id, neighbors, text), ...]
            nodes_list = cast(List[Tuple[NodeIDType, List[NodeIDType], Optional[str]]], nodes)
            if not all(len(node) >= 2 for node in nodes_list):
                raise ValueError("Each tuple in list must have at least 2 elements")
            node_ids = [node[0] for node in nodes_list]
            neighbor_lists = [node[1] for node in nodes_list]
        
        # Create result matrix with appropriate error handling
        dim = self.embedding_dim  # This should always be an int in this adapter
        results = np.zeros((len(node_ids), dim), dtype=np.float32)
        
        # Fill in embeddings one by one
        for i, (node_id, neighbors) in enumerate(zip(node_ids, neighbor_lists)):
            embedding = self.encode(node_id, neighbors)
            if embedding is not None:
                # Ensure the embedding is the right shape and type
                results[i] = np.asarray(embedding, dtype=np.float32)
            else:
                # If no embedding could be computed, leave as zeros
                continue
            
        return results
    
    def get_stats(self) -> EmbeddingStats:
        """
        Get statistics about the embeddings.
        
        Returns:
            Embedding statistics
        """
        return EmbeddingStats(
            model_name="node2vec",
            embedding_dim=self.embedding_dim,
            node_count=len(self._embeddings),
            training_parameters={
                "dimensions": self.embedding_dim,
                "walk_length": getattr(self, "walk_length", 80),
                "num_walks": getattr(self, "num_walks", 10),
                "p": getattr(self, "p", 1.0),
                "q": getattr(self, "q", 1.0)
            }
        )
    
    def save(self, path: str) -> None:
        """
        Save the model to a file.
        
        Args:
            path: Path to save the model to
        """
        # Delegate to implementation method to make type checking cleaner
        self._save_impl(path)
    
    def _save_impl(self, path: str) -> None:
        """Implementation helper for save to avoid mypy unreachable code errors."""
        if self._model is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        # Create directory structure safely
        path_obj = Path(path)  # type: ignore[unreachable]
        dir_path = path_obj.parent
        dir_path.mkdir(parents=True, exist_ok=True)
        
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

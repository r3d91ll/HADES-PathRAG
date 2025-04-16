"""
Adapter for SentenceTransformer embedding models.

This adapter wraps the SentenceTransformer implementation and makes it compatible
with the HADES-PathRAG BaseEmbedder interface.
"""
from typing import Dict, List, Optional, Any, Union, Tuple, cast, Type
import logging
import os
import pickle
from pathlib import Path
import json

import numpy as np
import networkx as nx

# Import BaseEmbedder and other interfaces
from hades_pathrag.embeddings.base import BaseEmbedder
from hades_pathrag.embeddings.interfaces import EmbeddingStats
from hades_pathrag.typings import (
    EmbeddingArray, NodeIDType, Graph, EmbeddingDict, PathType
)

logger = logging.getLogger(__name__)

# Try to import SentenceTransformer, but don't fail if not available
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMER_AVAILABLE = True
except ImportError:
    logger.debug("sentence_transformers package not available")
    SENTENCE_TRANSFORMER_AVAILABLE = False
    # Create dummy class for type checking
    class _SentenceTransformerDummy:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass
        
        def to(self, device: str) -> '_SentenceTransformerDummy':
            return self
            
        def get_sentence_embedding_dimension(self) -> int:
            return 0
            
        def encode(self, sentences: List[str], **kwargs: Any) -> np.ndarray:
            return np.array([])
    
    # For type checking purposes
    SentenceTransformer = _SentenceTransformerDummy  # type: ignore


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
        # Pass cache_folder as a keyword argument, not as a second positional argument
        if cache_folder:
            self._model = SentenceTransformer(model_name, cache_folder=cache_folder)
        else:
            self._model = SentenceTransformer(model_name)
        
        # Handle device separately if provided
        if device:
            self._model = self._model.to(device)
        
        # Get embedding dimension from model
        self.embedding_dim = embedding_dim or self._model.get_sentence_embedding_dimension()
        
        # Node ID to text mapping
        self._node_texts: Dict[NodeIDType, str] = {}
        self._embeddings: EmbeddingDict = {}
    
    @property
    def name(self) -> str:
        """Get the name of the embedder."""
        return f"sentence-transformer-{self.model_name}"
    
    def add_texts(self, texts_dict: Dict[NodeIDType, str]) -> None:
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
                
                tensor_embeddings = self._model.encode(batch_texts)
                embeddings = tensor_embeddings.numpy() if hasattr(tensor_embeddings, 'numpy') else tensor_embeddings.detach().cpu().numpy()
                
                for node, embedding in zip(batch_nodes, embeddings):
                    self._embeddings[node] = embedding
    
    def encode(self, node_id: NodeIDType, neighbors: List[NodeIDType]) -> EmbeddingArray:
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
            # Ensure correct type with explicit cast
            return cast(EmbeddingArray, np.asarray(self._embeddings[node_id], dtype=np.float32))
        
        # If node has text, encode it
        if node_id in self._node_texts:
            text = self._node_texts[node_id]
            tensor_embedding = self._model.encode(text)
            embedding = tensor_embedding.numpy() if hasattr(tensor_embedding, 'numpy') else tensor_embedding.detach().cpu().numpy()
            # Ensure correct type with explicit conversion
            embedding_array = np.asarray(embedding, dtype=np.float32)
            self._embeddings[node_id] = embedding_array
            # Return explicitly typed array instead of using cast
            return embedding_array
        
        # If node doesn't have text but neighbors do, average neighbor embeddings
        neighbor_embeddings: List[EmbeddingArray] = []
        for neighbor in neighbors:
            if neighbor in self._embeddings:
                # Ensure each embedding is of the correct type
                emb = np.asarray(self._embeddings[neighbor], dtype=np.float32)
                neighbor_embeddings.append(emb)
        
        if neighbor_embeddings:
            # Create explicitly typed array and store it
            avg_embedding_array = np.asarray(np.mean(neighbor_embeddings, axis=0), dtype=np.float32)
            self._embeddings[node_id] = avg_embedding_array
            return avg_embedding_array
        
        # Fallback to zeros - ensure embedding_dim is not None
        dim = self.embedding_dim if self.embedding_dim is not None else 768
        # Create a float32 array directly
        fallback_embedding = np.zeros(dim, dtype=np.float32)
        return fallback_embedding

    def get_embedding(self, node_id: NodeIDType) -> Optional[EmbeddingArray]:
        """
        Get the embedding for a node.
        
        Args:
            node_id: ID of the node
            
        Returns:
            Embedding vector if available, None otherwise
        """
        embedding = self._embeddings.get(node_id)
        if embedding is None:
            return None
        # Ensure correct type with explicit cast
        return cast(EmbeddingArray, np.asarray(embedding, dtype=np.float32))

    def encode_text(self, text: str) -> EmbeddingArray:
        """
        Encode a text string directly.
        
        Args:
            text: Text to encode
            
        Returns:
            Embedding vector
        """
        tensor_embedding = self._model.encode(text)
        return tensor_embedding.numpy() if hasattr(tensor_embedding, 'numpy') else tensor_embedding.detach().cpu().numpy()

    def batch_encode(self, nodes: Union[List[Tuple[NodeIDType, List[NodeIDType], Optional[str]]], Tuple[List[NodeIDType], List[List[NodeIDType]]]]) -> EmbeddingArray:
        """
        Encode multiple nodes in a batch.
        
        Args:
            nodes: Either a list of (node_id, neighbors, text) tuples or a tuple of (node_ids, neighbor_lists)
            
        Returns:
            Matrix of node embeddings
        """
        if isinstance(nodes, tuple):
            node_ids, neighbor_lists = nodes
            texts = [self._node_texts.get(nid, "") for nid in node_ids]
        else:
            node_ids = [n[0] for n in nodes]
            neighbor_lists = [n[1] for n in nodes]
            texts = [n[2] for n in nodes if len(n) > 2]
        
        # Encode all texts at once
        if texts and any(t for t in texts):
            tensor_embeddings = self._model.encode(texts)
            # Ensure correct type with explicit cast
            embeddings = np.asarray(
                tensor_embeddings.numpy() if hasattr(tensor_embeddings, 'numpy') else tensor_embeddings.detach().cpu().numpy(),
                dtype=np.float32
            )
            
            # Store embeddings for nodes with text
            for nid, emb in zip(node_ids, embeddings):
                if nid in self._node_texts:
                    self._embeddings[nid] = emb
        else:
            # Ensure embedding_dim is not None
            dim = self.embedding_dim if self.embedding_dim is not None else 768
            embeddings = np.zeros((len(node_ids), dim), dtype=np.float32)
            
            # Handle nodes without text by averaging neighbor embeddings
            for i, (nid, neighbors) in enumerate(zip(node_ids, neighbor_lists)):
                if nid not in self._node_texts:
                    # Get embeddings for neighbors, ensuring they're not None
                    neighbor_embs: List[EmbeddingArray] = []
                    for n in neighbors:
                        emb = self.get_embedding(n)
                        if emb is not None:
                            neighbor_embs.append(emb)
                    
                    if neighbor_embs:
                        # Calculate mean and ensure float32 type
                        avg_emb = np.mean(neighbor_embs, axis=0).astype(np.float32)
                        embeddings[i] = avg_emb
                        self._embeddings[nid] = avg_emb
        
        # Ensure the returned value is explicitly of type EmbeddingArray
        return np.asarray(embeddings, dtype=np.float32)

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
        
        # Ensure we return a float, not Any
        return float(np.dot(a, b) / (norm_a * norm_b))

    def get_stats(self) -> EmbeddingStats:
        """
        Get statistics about the embeddings.
        
        Returns:
            Embedding statistics
        """
        # Create EmbeddingStats object with correct typing
        return EmbeddingStats(
            model_name=f"sentence-transformer-{self.model_name}",
            embedding_dim=self.embedding_dim if self.embedding_dim is not None else 768,
            node_count=len(self._embeddings),
            training_parameters={
                "model": self.model_name,
                "average_norm": float(np.mean([np.linalg.norm(emb) for emb in self._embeddings.values()]) if self._embeddings else 0.0)
            }
        )

    def fit(self, graph: Graph) -> None:
        """
        Fit the model to the graph by encoding all nodes with text content.
        
        Args:
            graph: NetworkX graph to fit on
        """
        # Get node count safely without calling list()
        node_count = sum(1 for _ in graph.nodes())
        logger.info(f"Fitting {self.name} to graph with {node_count} nodes")
        
        # Extract text content from nodes and create a mapping
        texts_dict = {}
        for node in graph.nodes():
            # Get node data safely
            node_data = dict(graph.nodes[node]) if node in graph.nodes else {}
            if 'text' in node_data and isinstance(node_data['text'], str):
                texts_dict[node] = node_data['text']
        
        logger.info(f"Found {len(texts_dict)} nodes with text content")
        self.add_texts(texts_dict)

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

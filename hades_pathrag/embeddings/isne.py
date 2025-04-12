"""
Implementation of Inductive Shallow Node Embedding (ISNE) for PathRAG.

ISNE extends shallow embeddings to inductive learning by capturing
local neighborhood structure of each node, enabling effective generalization
to unseen nodes in the graph.
"""
import os
import json
import logging
from typing import Dict, List, Optional, Tuple, Union, Any, cast, Type

import numpy as np
import networkx as nx
from sklearn.preprocessing import normalize
from sentence_transformers import SentenceTransformer

from .base import BaseEmbedder

logger = logging.getLogger(__name__)


class ISNEEmbedder(BaseEmbedder):
    """
    ISNE (Inductive Shallow Node Embedding) implementation.
    
    This embedder creates node representations based on the average of neighbor parameters,
    allowing for inductive learning where new nodes can be embedded based on their neighbors.
    """
    
    def __init__(
        self, 
        embedding_dim: int = 128,
        learning_rate: float = 0.01,
        epochs: int = 10,
        batch_size: int = 32,
        negative_samples: int = 5,
        text_model_name: str = "all-MiniLM-L6-v2",
        random_seed: Optional[int] = None
    ) -> None:
        """
        Initialize the ISNE embedder.
        
        Args:
            embedding_dim: Dimension of node embeddings
            learning_rate: Learning rate for parameter updates
            epochs: Number of training epochs
            batch_size: Batch size for training
            negative_samples: Number of negative samples per positive sample
            text_model_name: Name of the text embedding model to use
            random_seed: Optional seed for reproducibility
        """
        self.embedding_dim: int = embedding_dim
        self.learning_rate: float = learning_rate
        self.epochs: int = epochs
        self.batch_size: int = batch_size
        self.negative_samples: int = negative_samples
        self.text_model_name: str = text_model_name
        self.random_seed: Optional[int] = random_seed
        
        # Set random seed if specified
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Initialize node parameters dictionary
        self.node_parameters: Dict[str, np.ndarray] = {}
        
        # Initialize text encoder for query encoding
        self.text_encoder: Optional[SentenceTransformer] = None
        try:
            self.text_encoder = SentenceTransformer(text_model_name)
            logger.info(f"Initialized text encoder: {text_model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize text encoder: {str(e)}")
            raise
    
    def fit(self, graph: nx.Graph) -> None:
        """
        Fit the ISNE model to the graph.
        
        This trains the node parameters using a contrastive learning approach
        where node embeddings are encouraged to be similar to their neighbors
        and dissimilar to randomly sampled non-neighbors.
        
        Args:
            graph: NetworkX graph representing the code structure
        """
        logger.info(f"Fitting ISNE embedder to graph with {len(graph.nodes)} nodes and {len(graph.edges)} edges")
        
        # Initialize parameters for all nodes
        for node in graph.nodes():
            node_id: str = str(node)  # Ensure node ID is a string
            self.node_parameters[node_id] = np.random.normal(
                0, 0.1, self.embedding_dim
            ).astype(np.float32)
        
        # Training loop
        for epoch in range(self.epochs):
            total_loss: float = 0.0
            
            # Process each node in the graph
            for node in graph.nodes():
                node_id: str = str(node)
                neighbors: List[str] = [str(n) for n in graph.neighbors(node)]
                
                if not neighbors:
                    continue
                
                # Update parameters based on contrastive learning objective
                loss: float = self._update_parameters(node_id, neighbors, graph)
                total_loss += loss
            
            # Log training progress
            avg_loss: float = total_loss / len(graph.nodes)
            logger.info(f"Epoch {epoch+1}/{self.epochs}, Average Loss: {avg_loss:.4f}")
        
        # Normalize embeddings after training
        for node_id in self.node_parameters:
            self.node_parameters[node_id] = normalize(
                self.node_parameters[node_id].reshape(1, -1)
            ).flatten().astype(np.float32)
    
    def encode(self, node_id: str, neighbors: List[str]) -> np.ndarray:
        """
        Generate embedding for a node using the ISNE formula.
        
        For seen nodes, returns the stored parameter vector.
        For unseen nodes, computes the embedding as the average of neighbor parameters.
        
        Args:
            node_id: ID of the node to embed
            neighbors: List of neighbor node IDs
            
        Returns:
            Node embedding vector
        """
        # If node is already in parameters, return its embedding
        if node_id in self.node_parameters:
            return self.node_parameters[node_id]
        
        # If no neighbors, return zero vector
        if not neighbors:
            return cast(np.ndarray, np.zeros(self.embedding_dim, dtype=np.float32))
        
        # Filter for valid neighbors (those in our parameter dictionary)
        valid_neighbors: List[str] = [n for n in neighbors if n in self.node_parameters]
        if not valid_neighbors:
            return cast(np.ndarray, np.zeros(self.embedding_dim, dtype=np.float32))
        
        # ISNE formula: h(v) = (1/|N_v|) * ∑(θ_n) for all n ∈ N_v
        neighbor_embeddings: np.ndarray = np.array([
            self.node_parameters[n] for n in valid_neighbors
        ])
        embedding: np.ndarray = np.mean(neighbor_embeddings, axis=0, dtype=np.float32)
        
        # Normalize the embedding
        embedding = normalize(embedding.reshape(1, -1)).flatten()
        
        return cast(np.ndarray, embedding.astype(np.float32))
    
    def encode_text(self, text: str) -> np.ndarray:
        """
        Generate an embedding for the given text.
        
        This is used for encoding queries or other text for comparison with node embeddings.
        
        Args:
            text: Text to embed
            
        Returns:
            Text embedding vector
        """
        # Check if text encoder is initialized
        if self.text_encoder is None:
            raise RuntimeError("Text encoder has not been initialized")
            
        # Use the text encoder to generate embeddings
        embedding: np.ndarray = self.text_encoder.encode(text, convert_to_numpy=True)
        
        # If dimensions don't match, we need to project
        if embedding.shape[0] != self.embedding_dim:
            # Simple projection strategy - could be made more sophisticated
            if embedding.shape[0] > self.embedding_dim:
                # Truncate if larger
                embedding = embedding[:self.embedding_dim]
            else:
                # Pad with zeros if smaller
                padding: np.ndarray = np.zeros(
                    self.embedding_dim - embedding.shape[0], 
                    dtype=np.float32
                )
                embedding = np.concatenate([embedding, padding])
        
        # Normalize the embedding
        embedding = normalize(embedding.reshape(1, -1)).flatten()
        
        return cast(np.ndarray, embedding.astype(np.float32))
    
    def batch_encode(self, nodes: Union[List[Tuple[str, List[str], Optional[str]]], Tuple[List[str], List[List[str]]]]) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Generate embeddings for multiple nodes at once.
        
        This method accepts either:
        - A list of (node_id, neighbors, text) tuples for enhanced embedders
        - A tuple of (node_ids, neighbor_lists) for basic embedders
        
        Args:
            nodes: Either a list of node tuples or a tuple of (node_ids, neighbor_lists)
            
        Returns:
            Matrix of node embeddings, one per row
        """
        # Handle the two different input formats
        if isinstance(nodes, tuple) and len(nodes) == 2:
            # Basic format: (node_ids, neighbor_lists)
            node_ids, neighbor_lists = nodes
            if len(node_ids) != len(neighbor_lists):
                raise ValueError("Number of node IDs must match number of neighbor lists")
            
            embeddings: np.ndarray = np.zeros(
                (len(node_ids), self.embedding_dim), 
                dtype=np.float32
            )
            
            for i, (node_id, neighbors) in enumerate(zip(node_ids, neighbor_lists)):
                embeddings[i] = self.encode(node_id, neighbors)
                
            return embeddings
            
        else:
            # Enhanced format: List[Tuple[node_id, neighbors, text]]
            # For basic embedder, ignore the text
            embeddings: np.ndarray = np.zeros(
                (len(nodes), self.embedding_dim), 
                dtype=np.float32
            )
            
            for i, (node_id, neighbors, _) in enumerate(nodes):
                embeddings[i] = self.encode(node_id, neighbors)
                
            return embeddings
    
    def save(self, path: str) -> None:
        """
        Save the embedder model to disk.
        
        Args:
            path: Path to save the model to
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_params: Dict[str, List[float]] = {
            node_id: params.tolist() 
            for node_id, params in self.node_parameters.items()
        }
        
        # Save model configuration and parameters
        model_data: Dict[str, Any] = {
            "config": {
                "embedding_dim": self.embedding_dim,
                "learning_rate": self.learning_rate,
                "epochs": self.epochs,
                "batch_size": self.batch_size,
                "negative_samples": self.negative_samples,
                "text_model_name": self.text_model_name,
                "random_seed": self.random_seed
            },
            "parameters": serializable_params
        }
        
        with open(path, 'w') as f:
            json.dump(model_data, f)
        
        logger.info(f"Saved ISNE model to {path}")
    
    @classmethod
    def load(cls: Type['ISNEEmbedder'], path: str) -> 'ISNEEmbedder':
        """
        Load the embedder model from disk.
        
        Args:
            path: Path to load the model from
            
        Returns:
            Loaded embedder model
        """
        with open(path, 'r') as f:
            model_data: Dict[str, Any] = json.load(f)
        
        # Extract configuration
        config: Dict[str, Any] = model_data["config"]
        
        # Initialize model with saved config
        model: ISNEEmbedder = cls(
            embedding_dim=config["embedding_dim"],
            learning_rate=config["learning_rate"],
            epochs=config["epochs"],
            batch_size=config["batch_size"],
            negative_samples=config["negative_samples"],
            text_model_name=config["text_model_name"],
            random_seed=config["random_seed"]
        )
        
        # Load parameters
        model.node_parameters = {
            node_id: np.array(params, dtype=np.float32)
            for node_id, params in model_data["parameters"].items()
        }
        
        logger.info(f"Loaded ISNE model from {path}")
        return model
    
    def _update_parameters(self, node_id: str, neighbors: List[str], graph: nx.Graph) -> float:
        """
        Update parameters using contrastive learning objective.
        
        Args:
            node_id: ID of the node being updated
            neighbors: List of neighbor node IDs
            graph: NetworkX graph for sampling negative examples
            
        Returns:
            Loss value for this update
        """
        # Get current node's parameters
        node_params: np.ndarray = self.node_parameters[node_id]
        
        # Sample negative nodes (non-neighbors)
        neg_samples: List[str] = self._sample_negatives(node_id, neighbors, list(graph.nodes()))
        
        # Calculate gradients
        grad: np.ndarray = np.zeros_like(node_params)
        loss: float = 0.0
        
        # Positive samples (neighbors)
        for neighbor in neighbors:
            neighbor_params: np.ndarray = self.node_parameters[neighbor]
            similarity: float = float(np.dot(node_params, neighbor_params))
            loss -= np.log(self._sigmoid(similarity))
            grad += (self._sigmoid(similarity) - 1.0) * neighbor_params
        
        # Negative samples
        for neg in neg_samples:
            neg_params: np.ndarray = self.node_parameters[neg]
            similarity: float = float(np.dot(node_params, neg_params))
            loss -= np.log(1.0 - self._sigmoid(similarity))
            grad += self._sigmoid(similarity) * neg_params
        
        # Update parameters with gradient
        self.node_parameters[node_id] = node_params - self.learning_rate * grad
        
        return float(loss)
    
    def _sample_negatives(
        self, 
        node_id: str, 
        neighbors: List[str], 
        all_nodes: List[Any]
    ) -> List[str]:
        """
        Sample negative examples (non-neighbors) for contrastive learning.
        
        Args:
            node_id: ID of the current node
            neighbors: List of neighbor node IDs
            all_nodes: List of all node IDs in the graph
            
        Returns:
            List of negative sample node IDs
        """
        # Convert all nodes to strings for consistency
        all_nodes_str: List[str] = [str(n) for n in all_nodes]
        
        # Create a set of nodes to exclude (self and neighbors)
        exclude: set[str] = set(neighbors + [node_id])
        
        # Filter candidate nodes
        candidates: List[str] = [n for n in all_nodes_str if n not in exclude]
        
        # If not enough candidates, sample with replacement
        if len(candidates) < self.negative_samples:
            if not candidates:  # If no candidates at all, return empty list
                return []
            return cast(List[str], np.random.choice(
                candidates, 
                size=self.negative_samples, 
                replace=True
            ).tolist())
        
        # Sample without replacement
        return cast(List[str], np.random.choice(
            candidates, 
            size=self.negative_samples, 
            replace=False
        ).tolist())
    
    @staticmethod
    def _sigmoid(x: float) -> float:
        """
        Sigmoid activation function.
        
        Args:
            x: Input value
            
        Returns:
            Sigmoid of x
        """
        return float(1.0 / (1.0 + np.exp(-x)))

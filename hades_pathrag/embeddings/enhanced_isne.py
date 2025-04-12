"""
Enhanced Inductive Shallow Node Embedding (ISNE) implementation for PathRAG.

This module provides an optimized implementation of ISNE that supports
inductive learning and neighborhood-based embedding generation for new nodes.
It implements the extended embedding interfaces for comprehensive functionality.
"""
import os
import json
import time
import logging
from typing import Dict, List, Optional, Tuple, Union, Any, cast, Type, Set
from datetime import datetime

import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import normalize
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from .base import BaseEmbedder
from .interfaces import (
    TrainableEmbedder, ComparableEmbedder, EmbeddingStats, 
    EmbeddingMetrics, EmbeddingCache, NodeID, Embedding
)

logger = logging.getLogger(__name__)


class NodeNeighborDataset(Dataset):
    """Dataset for batched training of node-neighbor pairs."""
    
    def __init__(
        self, 
        graph: nx.Graph, 
        neg_samples: int = 5,
        node_subset: Optional[List[str]] = None
    ):
        """
        Initialize node-neighbor dataset.
        
        Args:
            graph: NetworkX graph
            neg_samples: Number of negative samples per positive sample
            node_subset: Optional subset of nodes to include (for partial training)
        """
        self.graph = graph
        self.neg_samples = neg_samples
        
        # Convert all node IDs to strings
        self.nodes = [str(n) for n in graph.nodes()]
        
        # Filter to subset if provided
        if node_subset:
            self.nodes = [n for n in self.nodes if n in set(node_subset)]
        
        # Build node ID to index mapping
        self.node_to_idx = {node: i for i, node in enumerate(self.nodes)}
        
        # For each node, store its neighbor indices
        self.node_neighbors = {}
        self.node_non_neighbors = {}
        
        for node in self.nodes:
            neighbors = [str(n) for n in graph.neighbors(node)]
            neighbor_indices = [self.node_to_idx[n] for n in neighbors if n in self.node_to_idx]
            self.node_neighbors[node] = neighbor_indices
            
            # Pre-compute non-neighbors for negative sampling
            non_neighbor_indices = [
                i for i, n in enumerate(self.nodes) 
                if n not in neighbors and n != node
            ]
            self.node_non_neighbors[node] = non_neighbor_indices
    
    def __len__(self) -> int:
        """Return number of nodes in the dataset."""
        return len(self.nodes)
    
    def __getitem__(self, idx: int) -> Tuple[int, List[int], List[int]]:
        """
        Get a node with its positive and negative samples.
        
        Args:
            idx: Index of the node
            
        Returns:
            Tuple of (node_idx, pos_samples, neg_samples)
        """
        node = self.nodes[idx]
        
        # Get neighbor indices
        pos_indices = self.node_neighbors[node]
        
        # If no neighbors, return empty
        if not pos_indices:
            return idx, [], []
        
        # Sample positive neighbors if too many
        if len(pos_indices) > self.neg_samples:
            pos_indices = np.random.choice(
                pos_indices, 
                size=self.neg_samples,
                replace=False
            ).tolist()
        
        # Sample negative neighbors
        non_neighbors = self.node_non_neighbors[node]
        if len(non_neighbors) < self.neg_samples:
            # If not enough non-neighbors, sample with replacement or use all
            if non_neighbors:
                neg_indices = np.random.choice(
                    non_neighbors, 
                    size=self.neg_samples,
                    replace=True
                ).tolist()
            else:
                neg_indices = []
        else:
            # Sample without replacement
            neg_indices = np.random.choice(
                non_neighbors, 
                size=self.neg_samples,
                replace=False
            ).tolist()
        
        return idx, pos_indices, neg_indices


class ISNEModel(nn.Module):
    """PyTorch model for ISNE training."""
    
    def __init__(
        self, 
        num_nodes: int, 
        embedding_dim: int, 
        neighbor_aggregation: str = "mean"
    ):
        """
        Initialize ISNE model.
        
        Args:
            num_nodes: Number of nodes in the graph
            embedding_dim: Dimension of embeddings
            neighbor_aggregation: Method to aggregate neighbor embeddings 
                                  (mean, max, sum)
        """
        super().__init__()
        self.num_nodes = num_nodes
        self.embedding_dim = embedding_dim
        self.neighbor_aggregation = neighbor_aggregation
        
        # Create node parameters
        self.node_parameters = nn.Parameter(
            torch.randn(num_nodes, embedding_dim) * 0.1
        )
        
        # Layer for combining node and neighbor parameters
        self.combine_layer = nn.Linear(embedding_dim * 2, embedding_dim)
    
    def forward(
        self, 
        node_indices: torch.Tensor, 
        pos_neighbor_indices: List[List[int]], 
        neg_neighbor_indices: List[List[int]]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of the model.
        
        Args:
            node_indices: Indices of nodes
            pos_neighbor_indices: Lists of positive neighbor indices per node
            neg_neighbor_indices: Lists of negative neighbor indices per node
            
        Returns:
            Tuple of (node_embeddings, pos_embeddings, neg_embeddings)
        """
        # Get node parameters
        node_embeds = self.node_parameters[node_indices]
        
        # Process each node's neighbors
        pos_embeds = []
        neg_embeds = []
        
        for i, (pos_indices, neg_indices) in enumerate(zip(pos_neighbor_indices, neg_neighbor_indices)):
            # Get embeddings for positive neighbors
            if pos_indices:
                pos_params = self.node_parameters[pos_indices]
                if self.neighbor_aggregation == "mean":
                    pos_agg = torch.mean(pos_params, dim=0)
                elif self.neighbor_aggregation == "max":
                    pos_agg = torch.max(pos_params, dim=0)[0]
                else:  # sum
                    pos_agg = torch.sum(pos_params, dim=0)
                pos_embeds.append(pos_agg)
            else:
                pos_embeds.append(torch.zeros_like(node_embeds[i]))
            
            # Get embeddings for negative neighbors
            if neg_indices:
                neg_params = self.node_parameters[neg_indices]
                if self.neighbor_aggregation == "mean":
                    neg_agg = torch.mean(neg_params, dim=0)
                elif self.neighbor_aggregation == "max":
                    neg_agg = torch.max(neg_params, dim=0)[0]
                else:  # sum
                    neg_agg = torch.sum(neg_params, dim=0)
                neg_embeds.append(neg_agg)
            else:
                neg_embeds.append(torch.zeros_like(node_embeds[i]))
        
        if pos_embeds:
            pos_embeds = torch.stack(pos_embeds)
        else:
            pos_embeds = torch.zeros_like(node_embeds)
            
        if neg_embeds:
            neg_embeds = torch.stack(neg_embeds)
        else:
            neg_embeds = torch.zeros_like(node_embeds)
        
        return node_embeds, pos_embeds, neg_embeds
    
    def get_embeddings(self) -> np.ndarray:
        """Get all node embeddings."""
        return cast(np.ndarray, self.node_parameters.detach().cpu().numpy())


class EnhancedISNEEmbedder(BaseEmbedder):
    """
    Enhanced ISNE (Inductive Shallow Node Embedding) implementation.
    
    This implementation provides optimized training with PyTorch and full support
    for inductive learning, where new nodes can be embedded based on their 
    neighborhood structure without retraining the entire model.
    
    It implements both TrainableEmbedder and ComparableEmbedder protocols for
    incremental training and model comparison capabilities.
    """
    
    def __init__(
        self, 
        embedding_dim: int = 128,
        learning_rate: float = 0.01,
        epochs: int = 10,
        batch_size: int = 32,
        negative_samples: int = 5,
        text_model_name: str = "all-MiniLM-L6-v2",
        neighbor_aggregation: str = "mean",  # Options: mean, max, sum
        device: Optional[str] = None,
        cache_size: int = 10000,
        random_seed: Optional[int] = None
    ) -> None:
        """
        Initialize the Enhanced ISNE embedder.
        
        Args:
            embedding_dim: Dimension of node embeddings
            learning_rate: Learning rate for parameter updates
            epochs: Number of training epochs
            batch_size: Batch size for training
            negative_samples: Number of negative samples per positive sample
            text_model_name: Name of the text embedding model to use
            neighbor_aggregation: Method to aggregate neighbor embeddings
            device: PyTorch device (cuda, cpu), auto-detected if None
            cache_size: Maximum number of embeddings to cache
            random_seed: Optional seed for reproducibility
        """
        self.embedding_dim: int = embedding_dim
        self.learning_rate: float = learning_rate
        self.epochs: int = epochs
        self.batch_size: int = batch_size
        self.negative_samples: int = negative_samples
        self.text_model_name: str = text_model_name
        self.neighbor_aggregation: str = neighbor_aggregation
        self.random_seed: Optional[int] = random_seed
        
        # Auto-detect device if not specified
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # Set random seed if specified
        if random_seed is not None:
            np.random.seed(random_seed)
            torch.manual_seed(random_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(random_seed)
        
        # Initialize model
        self.model: Optional[ISNEModel] = None
        
        # Initialize node to index mapping
        self.node_to_idx: Dict[str, int] = {}
        self.idx_to_node: Dict[int, str] = {}
        
        # Initialize text encoder
        self.text_encoder: Optional[SentenceTransformer] = None
        try:
            self.text_encoder = SentenceTransformer(text_model_name)
            logger.info(f"Initialized text encoder: {text_model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize text encoder: {str(e)}")
            raise
        
        # Initialize embedding cache
        self.cache = EmbeddingCache(max_size=cache_size)
        
        # Initialize empty model parameters for type safety
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
        
        # Training statistics
        self.training_stats: Dict[str, Any] = {
            "last_training_time": None,
            "training_duration": 0.0,
            "epochs_completed": 0,
            "final_loss": 0.0,
        }
        
        # Model evaluation metrics
        self.metrics = EmbeddingMetrics()
        
        logger.info(f"Enhanced ISNE embedder initialized with {embedding_dim} dimensions")
    
    def fit(self, graph: nx.Graph) -> None:
        """
        Fit the ISNE model to the graph.
        
        This trains the node parameters using PyTorch and batch processing
        for efficient training on large graphs.
        
        Args:
            graph: NetworkX graph representing the knowledge graph
        """
        start_time = time.time()
        logger.info(f"Fitting Enhanced ISNE embedder to graph with {len(graph.nodes)} nodes and {len(graph.edges)} edges")
        
        # Convert node IDs to strings and build mappings
        nodes = [str(node) for node in graph.nodes()]
        self.node_to_idx = {node: i for i, node in enumerate(nodes)}
        self.idx_to_node = {i: node for i, node in enumerate(nodes)}
        
        # Initialize model
        self.model = ISNEModel(
            num_nodes=len(nodes),
            embedding_dim=self.embedding_dim,
            neighbor_aggregation=self.neighbor_aggregation
        ).to(self.device)
        
        # Create dataset and data loader
        dataset = NodeNeighborDataset(
            graph=graph,
            neg_samples=self.negative_samples
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0  # Increase for multi-processing if needed
        )
        
        # Initialize optimizer
        if self.model is None:
            raise ValueError("Model must be initialized before optimizer")
            
        optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=self.learning_rate
        )
        
        # Training loop
        best_loss = float('inf')
        
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{self.epochs}")
            
            for node_indices, pos_neighbor_indices, neg_neighbor_indices in progress_bar:
                # Convert to tensors
                node_indices = node_indices.to(self.device)
                
                # Forward pass
                if self.model is None:
                    raise ValueError("Model must be initialized before forward pass")
                    
                node_embeds, pos_embeds, neg_embeds = self.model(
                    node_indices, 
                    pos_neighbor_indices,
                    neg_neighbor_indices
                )
                
                # Compute loss
                batch_loss = 0.0
                
                # Only compute loss for nodes with neighbors
                for i, (pos_indices, neg_indices) in enumerate(zip(pos_neighbor_indices, neg_neighbor_indices)):
                    if not pos_indices:  # Skip nodes without neighbors
                        continue
                        
                    # Positive loss: -log(sigmoid(node·pos))
                    pos_similarity = torch.sum(node_embeds[i] * pos_embeds[i])
                    pos_loss = -torch.log(torch.sigmoid(pos_similarity) + 1e-6)
                    
                    # Negative loss: -log(1-sigmoid(node·neg))
                    if neg_indices:
                        neg_similarity = torch.sum(node_embeds[i] * neg_embeds[i])
                        neg_loss = -torch.log(1.0 - torch.sigmoid(neg_similarity) + 1e-6)
                    else:
                        neg_loss = 0.0
                    
                    batch_loss += pos_loss + neg_loss
                
                # If no valid nodes, skip optimization
                if batch_loss == 0.0:
                    continue
                
                # Backward pass and optimization
                optimizer.zero_grad()
                
                # Type check for tensor operations
                if isinstance(batch_loss, torch.Tensor) and batch_loss.requires_grad:
                    batch_loss.backward()
                    optimizer.step()
                    
                    # Update statistics safely
                    with torch.no_grad():
                        loss_value = batch_loss.item()
                    epoch_loss += loss_value
                    num_batches += 1
                    
                    # Update progress bar safely
                    progress_bar.set_postfix({"loss": loss_value})
                else:
                    # Handle non-tensor case
                    logger.warning(f"Batch loss is not a valid tensor: {type(batch_loss)}")
                    if isinstance(batch_loss, (int, float)):
                        epoch_loss += float(batch_loss)
                        num_batches += 1
                        progress_bar.set_postfix({"loss": float(batch_loss)})
            
            # Compute average loss for epoch
            avg_epoch_loss = epoch_loss / max(num_batches, 1)
            logger.info(f"Epoch {epoch+1}/{self.epochs}, Average Loss: {avg_epoch_loss:.4f}")
            
            # Track best model
            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
        
        # Normalize embeddings after training
        if self.model is not None and hasattr(self.model, 'node_parameters'):
            with torch.no_grad():
                # Safe access to node_parameters
                if isinstance(self.model.node_parameters, torch.nn.Parameter):
                    norms = torch.norm(self.model.node_parameters, dim=1, keepdim=True)
                    self.model.node_parameters.data = self.model.node_parameters.data / (norms + 1e-8)
                else:
                    logger.warning("Model node_parameters is not a torch Parameter")
        
        # Clear cache after training
        self.cache.clear()
        
        # Update training statistics
        training_duration = time.time() - start_time
        self.training_stats = {
            "last_training_time": datetime.now().isoformat(),
            "training_duration": training_duration,
            "epochs_completed": self.epochs,
            "final_loss": best_loss,
            "num_nodes": len(nodes),
            "num_edges": len(graph.edges)
        }
        
        # Update metrics
        self.metrics.training_loss = best_loss
        self.metrics.training_time_seconds = training_duration
        
        logger.info(f"Training completed in {training_duration:.2f} seconds")
    
    def partial_fit(self, graph: nx.Graph, new_nodes: List[NodeID]) -> None:
        """
        Update the embedder model with new nodes while preserving existing embeddings.
        
        This implements incremental training where we only train on new nodes
        and their connections, preserving the existing embeddings for other nodes.
        
        Args:
            graph: NetworkX graph with both existing and new nodes
            new_nodes: List of new node IDs to train on
        """
        if self.model is None:
            # If no existing model, do a full training
            return self.fit(graph)
        
        start_time = time.time()
        logger.info(f"Incrementally training on {len(new_nodes)} new nodes")
        
        # Create mappings for existing nodes
        existing_nodes = list(self.node_to_idx.keys())
        all_nodes = existing_nodes.copy()
        
        # Find truly new nodes that aren't in our index yet
        new_nodes_set = set(str(n) for n in new_nodes)
        actual_new_nodes = [n for n in new_nodes_set if n not in self.node_to_idx]
        
        if not actual_new_nodes:
            logger.info("No new nodes to train on")
            return
        
        # Add new nodes to mappings
        for node in actual_new_nodes:
            idx = len(all_nodes)
            self.node_to_idx[node] = idx
            self.idx_to_node[idx] = node
            all_nodes.append(node)
        
        # Create new model with expanded parameters
        old_params = self.model.get_embeddings()  # Get existing parameters
        
        new_model = ISNEModel(
            num_nodes=len(all_nodes),
            embedding_dim=self.embedding_dim,
            neighbor_aggregation=self.neighbor_aggregation
        ).to(self.device)
        
        # Copy existing parameters
        with torch.no_grad():
            new_model.node_parameters.data[:len(old_params)] = torch.tensor(
                old_params, device=self.device
            )
        
        # Create dataset for new nodes and their connections
        dataset = NodeNeighborDataset(
            graph=graph,
            neg_samples=self.negative_samples,
            node_subset=[n for n in all_nodes if n in new_nodes_set 
                        or any(neighbor in new_nodes_set 
                              for neighbor in graph.neighbors(n))]
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0
        )
        
        # Freeze existing node parameters and only train new ones
        for param in new_model.parameters():
            param.requires_grad = False
        new_model.node_parameters.requires_grad = True
        new_model.node_parameters.data[len(old_params):].requires_grad = True
        
        # Initialize optimizer for only new parameters
        optimizer = torch.optim.Adam(
            [new_model.node_parameters], 
            lr=self.learning_rate
        )
        
        # Training loop - fewer epochs for incremental updates
        incremental_epochs = max(1, self.epochs // 2)
        
        for epoch in range(incremental_epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{incremental_epochs}")
            
            for node_indices, pos_neighbor_indices, neg_neighbor_indices in progress_bar:
                # Convert to tensors
                node_indices = node_indices.to(self.device)
                
                # Forward pass
                node_embeds, pos_embeds, neg_embeds = new_model(
                    node_indices, 
                    pos_neighbor_indices,
                    neg_neighbor_indices
                )
                
                # Compute loss
                batch_loss = 0.0
                
                # Only compute loss for nodes with neighbors
                for i, (pos_indices, neg_indices) in enumerate(zip(pos_neighbor_indices, neg_neighbor_indices)):
                    if not pos_indices:  # Skip nodes without neighbors
                        continue
                        
                    # Positive loss
                    pos_similarity = torch.sum(node_embeds[i] * pos_embeds[i])
                    pos_loss = -torch.log(torch.sigmoid(pos_similarity) + 1e-6)
                    
                    # Negative loss
                    if neg_indices:
                        neg_similarity = torch.sum(node_embeds[i] * neg_embeds[i])
                        neg_loss = -torch.log(1.0 - torch.sigmoid(neg_similarity) + 1e-6)
                    else:
                        neg_loss = 0.0
                    
                    batch_loss += pos_loss + neg_loss
                
                # If no valid nodes, skip optimization
                if batch_loss == 0.0:
                    continue
                
                # Backward pass and optimization
                optimizer.zero_grad()
                
                # Type check for tensor operations
                if isinstance(batch_loss, torch.Tensor) and batch_loss.requires_grad:
                    batch_loss.backward()
                    optimizer.step()
                    
                    # Update statistics safely
                    with torch.no_grad():
                        loss_value = batch_loss.item()
                    epoch_loss += loss_value
                    num_batches += 1
                    
                    # Update progress bar safely
                    progress_bar.set_postfix({"loss": loss_value})
                else:
                    # Handle non-tensor case
                    logger.warning(f"Batch loss is not a valid tensor: {type(batch_loss)}")
                    if isinstance(batch_loss, (int, float)):
                        epoch_loss += float(batch_loss)
                        num_batches += 1
                        progress_bar.set_postfix({"loss": float(batch_loss)})
            
            # Compute average loss for epoch
            avg_epoch_loss = epoch_loss / max(num_batches, 1)
            logger.info(f"Epoch {epoch+1}/{incremental_epochs}, Average Loss: {avg_epoch_loss:.4f}")
        
        # Normalize embeddings after training
        with torch.no_grad():
            norms = torch.norm(new_model.node_parameters, dim=1, keepdim=True)
            new_model.node_parameters.data = new_model.node_parameters.data / (norms + 1e-8)
        
        # Update model and clear cache
        self.model = new_model
        self.cache.clear()
        
        # Update training statistics
        training_duration = time.time() - start_time
        self.training_stats.update({
            "last_update_time": datetime.now().isoformat(),
            "incremental_training_duration": training_duration,
            "incremental_epochs": incremental_epochs,
            "new_nodes_added": len(actual_new_nodes)
        })
        
        logger.info(f"Incremental training completed in {training_duration:.2f} seconds")
    
    def encode(self, node_id: NodeID, neighbors: Optional[List[NodeID]] = None, text: Optional[str] = None) -> np.ndarray:
        """
        Generate an embedding for a node based on its neighborhood.
        
        This is the core inductive functionality of ISNE, allowing embeddings to be
        generated for nodes that weren't present during training, based solely on
        their neighborhood structure.
        
        Args:
            node_id: ID of the node to encode
            neighbors: Optional list of neighbor node IDs. If None, attempts to use
                       cached neighbors or returns a text-only embedding
            text: Optional text associated with the node, used for cold-start
            
        Returns:
            Embedding vector as numpy array
        """
        # Check cache first
        node_id_str = str(node_id)
        cached = self.cache.get(node_id_str)
        if cached is not None:
            return cached
        
        # If model not initialized, we can only rely on text
        if self.model is None:
            if text and self.text_encoder:
                text_embedding = self.text_encoder.encode(text)
                if isinstance(text_embedding, list):
                    text_embedding = np.array(text_embedding)
                # Normalize if needed
                if len(text_embedding.shape) > 1:
                    text_embedding = text_embedding.mean(axis=0)
                text_embedding = normalize(text_embedding.reshape(1, -1))[0]
                
                # Resize if necessary
                if text_embedding.shape[0] != self.embedding_dim:
                    # Use PCA-like approach to project to target dimension
                    if text_embedding.shape[0] > self.embedding_dim:
                        # Downsample: take first N dimensions
                        text_embedding = text_embedding[:self.embedding_dim]
                    else:
                        # Upsample: pad with zeros
                        padding = np.zeros(self.embedding_dim)
                        padding[:text_embedding.shape[0]] = text_embedding
                        text_embedding = padding
                
                # Save in cache and return
                normalized = normalize(text_embedding.reshape(1, -1))[0]
                self.cache.put(node_id_str, normalized)
                return cast(np.ndarray, normalized)
            else:
                logger.warning(f"No trained model, and no text provided for node {node_id}")
                # Return a random embedding as fallback
                random_embed = np.random.randn(self.embedding_dim)
                normalized = normalize(random_embed.reshape(1, -1))[0]
                return cast(np.ndarray, normalized)
        
        # If node exists in model, return its embedding directly
        if node_id_str in self.node_to_idx:
            node_idx = self.node_to_idx[node_id_str]
            embedding = self.model.get_embeddings()[node_idx]
            self.cache.put(node_id_str, embedding)
            return embedding
        
        # Inductive case: generate embedding from neighbors
        if neighbors:
            # Filter neighbors that exist in our model
            valid_neighbors = [n for n in neighbors if str(n) in self.node_to_idx]
            
            if valid_neighbors:
                # Get embeddings for all valid neighbors
                neighbor_embeddings = np.stack([
                    self.model.get_embeddings()[self.node_to_idx[str(n)]]
                    for n in valid_neighbors
                ])
                
                # Aggregate neighbor embeddings based on configured method
                if self.neighbor_aggregation == "mean":
                    aggregated = np.mean(neighbor_embeddings, axis=0)
                elif self.neighbor_aggregation == "max":
                    aggregated = np.max(neighbor_embeddings, axis=0)
                else:  # sum
                    aggregated = np.sum(neighbor_embeddings, axis=0)
                
                # Normalize
                normalized = normalize(aggregated.reshape(1, -1))[0]
                
                # If we have text, combine with text embedding
                if text and self.text_encoder:
                    text_embedding = self.text_encoder.encode(text)
                    if isinstance(text_embedding, list):
                        text_embedding = np.array(text_embedding)
                    if len(text_embedding.shape) > 1:
                        text_embedding = text_embedding.mean(axis=0)
                    
                    # Resize text embedding if needed
                    if text_embedding.shape[0] != self.embedding_dim:
                        if text_embedding.shape[0] > self.embedding_dim:
                            text_embedding = text_embedding[:self.embedding_dim]
                        else:
                            padding = np.zeros(self.embedding_dim)
                            padding[:text_embedding.shape[0]] = text_embedding
                            text_embedding = padding
                    
                    text_embedding = normalize(text_embedding.reshape(1, -1))[0]
                    
                    # Combine the two embeddings (0.7 neighborhood, 0.3 text)
                    combined = 0.7 * normalized + 0.3 * text_embedding
                    normalized = normalize(combined.reshape(1, -1))[0]
                
                # Store in cache and return
                self.cache.put(node_id_str, normalized)
                return cast(np.ndarray, normalized)
        
        # If no valid neighbors, fall back to text embedding
        if text and self.text_encoder:
            text_embedding = self.text_encoder.encode(text)
            if isinstance(text_embedding, list):
                text_embedding = np.array(text_embedding)
            if len(text_embedding.shape) > 1:
                text_embedding = text_embedding.mean(axis=0)
                
            # Resize if necessary
            if text_embedding.shape[0] != self.embedding_dim:
                if text_embedding.shape[0] > self.embedding_dim:
                    text_embedding = text_embedding[:self.embedding_dim]
                else:
                    padding = np.zeros(self.embedding_dim)
                    padding[:text_embedding.shape[0]] = text_embedding
                    text_embedding = padding
            
            normalized = normalize(text_embedding.reshape(1, -1))[0]
            self.cache.put(node_id_str, normalized)
            return cast(np.ndarray, normalized)
        
        # Last resort: generate a random vector
        logger.warning(f"No neighbors or text for node {node_id}, generating random embedding")
        random_embed = np.random.randn(self.embedding_dim)
        normalized = normalize(random_embed.reshape(1, -1))[0]
        return cast(np.ndarray, normalized)
    
    def batch_encode(self, nodes: Union[List[Tuple[str, List[str], Optional[str]]], Tuple[List[str], List[List[str]]]]) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Generate embeddings for multiple nodes at once.
        
        This method accepts either:
        - A list of (node_id, neighbors, text) tuples for enhanced embedders (preferred)
        - A tuple of (node_ids, neighbor_lists) for backward compatibility
        
        Args:
            nodes: Either a list of node tuples or a tuple of (node_ids, neighbor_lists)
            
        Returns:
            List of embedding vectors
        """
        results: List[Embedding] = []
        
        # Handle the two different input formats
        if isinstance(nodes, tuple) and len(nodes) == 2:
            # Convert basic format (node_ids, neighbor_lists) to enhanced format
            node_ids, neighbor_lists = nodes
            if len(node_ids) != len(neighbor_lists):
                raise ValueError("Number of node IDs must match number of neighbor lists")
                
            # Convert to enhanced format with None for text
            enhanced_nodes = [(node_id, neighbors, None) for node_id, neighbors in zip(node_ids, neighbor_lists)]
        else:
            # Already in enhanced format: List[Tuple[node_id, neighbors, text]]
            enhanced_nodes = nodes
        
        # Check cache first for all nodes
        need_encoding = []
        node_indices = []
        
        for i, (node_id, neighbors, text) in enumerate(enhanced_nodes):
            node_id_str = str(node_id)
            cached = self.cache.get(node_id_str)
            
            if cached is not None:
                results.append(cached)
            else:
                # Create a zero vector as placeholder instead of None
                results.append(np.zeros(self.embedding_dim))
                need_encoding.append((node_id, neighbors, text))
                node_indices.append(i)
        
        # If all found in cache, return early
        if not need_encoding:
            return results
        
        # Process nodes that need encoding
        for (node_id, neighbors, text), orig_idx in zip(need_encoding, node_indices):
            # Encode individual node
            encoding = self.encode(node_id, neighbors, text)
            results[orig_idx] = encoding
        
        return results
    
    def _process_single_comparison(self, source: np.ndarray, target: Embedding) -> float:
        """Process a single comparison between source and target embeddings."""
        if target is not None and isinstance(target, np.ndarray):
            try:
                normalized_target = normalize(target.reshape(1, -1))[0]
                return float(np.dot(source, normalized_target))
            except Exception as e:
                logger.warning(f"Error comparing embeddings: {e}")
        return 0.0
    
    def _create_default_similarities(self, count: int) -> np.ndarray:
        """Create a default array of zeros for invalid comparisons."""
        return cast(np.ndarray, np.zeros(count, dtype=np.float32))
    
    def _is_valid_embedding(self, embedding: Any) -> bool:
        """Check if an embedding is valid for comparison operations."""
        return embedding is not None and isinstance(embedding, np.ndarray)
        
    def compare(self, source_embedding: Embedding, target_embeddings: List[Embedding]) -> np.ndarray:
        """
        Compare a source embedding with multiple target embeddings.
        
        Args:
            source_embedding: Source embedding vector
            target_embeddings: List of target embedding vectors
            
        Returns:
            Array of similarity scores (higher is more similar)
        """
        # First validate the source embedding
        if not self._is_valid_embedding(source_embedding):
            logger.warning(f"Source embedding has invalid type: {type(source_embedding)}")
            return self._create_default_similarities(len(target_embeddings))

        # We know the source embedding is valid at this point
        # Use explicit cast to avoid type errors
        source = normalize(cast(np.ndarray, source_embedding).reshape(1, -1))[0]
        
        # Calculate similarities
        similarities = [self._process_single_comparison(source, target) for target in target_embeddings]
        similarities_array = np.array(similarities, dtype=np.float32)
        
        # Update metrics
        self.metrics.num_comparisons += len(target_embeddings)
        
        return cast(np.ndarray, similarities_array)
    
    def inductive_embedding(self, node_id: NodeID, neighbors: List[NodeID], text: Optional[str] = None) -> np.ndarray:
        """
        Generate an embedding for a node that wasn't seen during training.
        
        This is an explicit alias for encode() that emphasizes the inductive capability.
        
        Args:
            node_id: ID of the node
            neighbors: Neighbor node IDs
            text: Optional text associated with the node
            
        Returns:
            Embedding vector
        """
        return self.encode(node_id, neighbors, text)
    
    def save(self, path: str) -> None:
        """
        Save the embedder model to disk.
        
        Args:
            path: Path to save the model to
        """
        if self.model is None:
            raise ValueError("Cannot save model that hasn't been trained")
        
        # Prepare model parameters
        model_params = self.model.get_embeddings()
        
        # Serialize node embeddings
        parameters = {}
        for node_id, idx in self.node_to_idx.items():
            parameters[node_id] = model_params[idx].tolist()
        
        # Prepare model data
        model_data = {
            "config": {
                "embedding_dim": self.embedding_dim,
                "learning_rate": self.learning_rate,
                "epochs": self.epochs,
                "batch_size": self.batch_size,
                "negative_samples": self.negative_samples,
                "text_model_name": self.text_model_name,
                "neighbor_aggregation": self.neighbor_aggregation,
                "random_seed": self.random_seed
            },
            "parameters": parameters,
            "node_mapping": self.node_to_idx,
            "training_stats": self.training_stats,
            "metrics": self.metrics.dict(),
            "version": "1.0.0"
        }
        
        # Save to file
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(model_data, f)
        
        logger.info(f"Saved Enhanced ISNE model to {path}")
    
    @classmethod
    def load(cls: Type['EnhancedISNEEmbedder'], path: str) -> 'EnhancedISNEEmbedder':
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
        embedder = cls(
            embedding_dim=config["embedding_dim"],
            learning_rate=config["learning_rate"],
            epochs=config["epochs"],
            batch_size=config["batch_size"],
            negative_samples=config["negative_samples"],
            text_model_name=config["text_model_name"],
            neighbor_aggregation=config.get("neighbor_aggregation", "mean"),
            random_seed=config.get("random_seed", None)
        )
        
        # Load node mappings
        embedder.node_to_idx = model_data["node_mapping"]
        embedder.idx_to_node = {int(idx): node_id for node_id, idx in embedder.node_to_idx.items()}
        
        # Get number of nodes and initialize model
        num_nodes = len(embedder.node_to_idx)
        embedder.model = ISNEModel(
            num_nodes=num_nodes,
            embedding_dim=embedder.embedding_dim,
            neighbor_aggregation=embedder.neighbor_aggregation
        ).to(embedder.device)
        
        # Load parameters
        all_params = np.zeros((num_nodes, embedder.embedding_dim), dtype=np.float32)
        for node_id, params in model_data["parameters"].items():
            idx = embedder.node_to_idx[node_id]
            all_params[idx] = np.array(params, dtype=np.float32)
        
        # Set model parameters
        with torch.no_grad():
            # Safe access with null check
            if embedder.model is not None and hasattr(embedder.model, 'node_parameters'):
                embedder.model.node_parameters.data = torch.tensor(
                    all_params, device=embedder.device
                )
            else:
                logger.error("Model or node_parameters not available for parameter loading")
                # Create a new model if none exists
                embedder.model = ISNEModel(
                    num_nodes=num_nodes,
                    embedding_dim=embedder.embedding_dim,
                    device=embedder.device
                )
                embedder.model.node_parameters.data = torch.tensor(
                    all_params, device=embedder.device
                )
        
        # Load training stats
        if "training_stats" in model_data:
            embedder.training_stats = model_data["training_stats"]
        
        # Load metrics
        if "metrics" in model_data:
            metrics_dict = model_data["metrics"]
            embedder.metrics = EmbeddingMetrics(
                training_loss=metrics_dict.get("training_loss", 0.0),
                training_time_seconds=metrics_dict.get("training_time_seconds", 0.0),
                num_comparisons=metrics_dict.get("num_comparisons", 0),
                num_inductive_embeddings=metrics_dict.get("num_inductive_embeddings", 0),
                cache_hits=metrics_dict.get("cache_hits", 0),
                cache_misses=metrics_dict.get("cache_misses", 0)
            )
        
        logger.info(f"Loaded Enhanced ISNE model from {path} with {num_nodes} nodes")
        return embedder
    
    def get_stats(self) -> EmbeddingStats:
        """
        Get current embedding model statistics.
        
        Returns:
            EmbeddingStats object with current metrics
        """
        num_nodes = len(self.node_to_idx) if self.node_to_idx else 0
        
        return EmbeddingStats(
            model_type="Enhanced ISNE",
            embedding_dim=self.embedding_dim,
            num_nodes=num_nodes,
            training_stats=self.training_stats,
            metrics=self.metrics,
            cache_stats={
                "size": len(self.cache.cache) if hasattr(self.cache, 'cache') else 0,
                "hits": self.metrics.cache_hits,
                "misses": self.metrics.cache_misses,
                "hit_rate": self.metrics.cache_hits / max(1, (self.metrics.cache_hits + self.metrics.cache_misses))
            }
        )

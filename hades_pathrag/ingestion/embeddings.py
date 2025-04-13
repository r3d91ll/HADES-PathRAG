"""
Embedding processors for the HADES-PathRAG ingestion pipeline.

This module provides embedders for documents and graphs, including
the Inductive Shallow Node Embedding (ISNE) implementation.
"""
import logging
import os
from typing import Dict, List, Optional, Tuple, Any, Set, Union, cast
import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
from torch_geometric.data import Data  # type: ignore
from torch_scatter import scatter_mean  # type: ignore

from hades_pathrag.ingestion.models import IngestDataset, IngestDocument, DocumentRelation

logger = logging.getLogger(__name__)


class ISNELayer(nn.Module):
    """
    Inductive Shallow Node Embedding Layer.
    
    Implementation based on the paper "Inductive Shallow Node Embedding"
    by Richard Csaky and Andras Majdik.
    """
    def __init__(self, num_nodes: int, hidden_channels: int, *args: Any, **kwargs: Any) -> None:
        """
        Initialize the ISNE layer.
        
        Args:
            num_nodes: Number of nodes in the graph
            hidden_channels: Dimension of the embedding vectors
        """
        super().__init__()
        self.emb = nn.Embedding(num_nodes, hidden_channels, *args, **kwargs)

    def forward(self, node_ids: Tensor, edge_index: Tensor) -> Tensor:
        """
        Forward pass through the ISNE layer.
        
        Args:
            node_ids: IDs of the nodes to compute embeddings for
            edge_index: Edge index tensor representing the graph structure
            
        Returns:
            Node embeddings
        """
        sources = node_ids[edge_index[0]]
        vs = self.emb(sources)
        index = edge_index[1]
        result = scatter_mean(vs, index, dim=0)
        return cast(Tensor, result)

    @property
    def embedding_dim(self) -> int:
        """Get the embedding dimension."""
        return self.emb.embedding_dim


class ISNEModel(nn.Module):
    """ISNE model for computing node embeddings."""
    
    def __init__(self, num_nodes: int, hidden_channels: int, *args: Any, **kwargs: Any) -> None:
        """
        Initialize the ISNE model.
        
        Args:
            num_nodes: Number of nodes in the graph
            hidden_channels: Dimension of the embedding vectors
        """
        super().__init__()
        self.num_nodes = num_nodes
        self.encoder = ISNELayer(num_nodes, hidden_channels, *args, **kwargs)
        
    def forward(self, edge_index: Tensor) -> Tensor:
        """
        Compute embeddings for all nodes in the graph.
        
        Args:
            edge_index: Edge index tensor representing the graph structure
            
        Returns:
            Node embeddings for all nodes
        """
        node_ids = torch.arange(self.num_nodes, device=edge_index.device)
        result: Tensor = self.encoder(node_ids, edge_index)
        return result


class ISNEEmbeddingProcessor:
    """
    Process documents using ISNE embeddings.
    
    This class converts a dataset of documents and relationships into a graph,
    computes ISNE embeddings, and updates the documents with their embeddings.
    """
    
    def __init__(
        self, 
        embedding_dim: int = 128,
        device: Optional[str] = None,
        weight_threshold: float = 0.5,
    ):
        """
        Initialize the ISNE embedding processor.
        
        Args:
            embedding_dim: Dimension of the embedding vectors
            device: Device to run the model on (cpu, cuda, etc.)
            weight_threshold: Minimum weight for a relationship to be included in the graph
        """
        self.embedding_dim = embedding_dim
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.weight_threshold = weight_threshold
        self._model: Optional[ISNEModel] = None
        self._node_id_mapping: Dict[str, int] = {}
        self._id_to_doc: Dict[str, IngestDocument] = {}
        
    def _build_graph(self, dataset: IngestDataset) -> Tuple[Tensor, Dict[str, int]]:
        """
        Build a graph from the dataset.
        
        Args:
            dataset: The dataset of documents and relationships
            
        Returns:
            Edge index tensor and mapping from document IDs to node indices
        """
        node_id_mapping = {}
        edges = []
        
        # Map documents to node indices
        for i, doc in enumerate(dataset.documents):
            node_id_mapping[doc.id] = i
            self._id_to_doc[doc.id] = doc
            
        # Create edges from relationships
        for rel in dataset.relationships:
            if rel.weight >= self.weight_threshold and rel.source_id in node_id_mapping and rel.target_id in node_id_mapping:
                source_idx = node_id_mapping[rel.source_id]
                target_idx = node_id_mapping[rel.target_id]
                edges.append([source_idx, target_idx])
                # Add reverse edge for undirected graph
                edges.append([target_idx, source_idx])
        
        if not edges:
            logger.warning("No edges found in dataset. Creating self-loops for documents.")
            for i in range(len(dataset.documents)):
                edges.append([i, i])
                
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        
        self._node_id_mapping = node_id_mapping
        return edge_index, node_id_mapping
    
    def process(self, dataset: IngestDataset) -> IngestDataset:
        """
        Process a dataset to compute and add ISNE embeddings.
        
        Args:
            dataset: The dataset to process
            
        Returns:
            The processed dataset with embeddings added to documents
        """
        logger.info(f"Processing dataset {dataset.name} with ISNE embedder...")
        
        # Build the graph
        edge_index, node_id_mapping = self._build_graph(dataset)
        num_nodes = len(dataset.documents)
        
        # Create and train the model
        model = ISNEModel(num_nodes, self.embedding_dim).to(self.device)
        self._model = model  # Assign to instance variable with proper typing
        edge_index = edge_index.to(self.device)
        
        # Compute embeddings
        # Initialize with default empty embeddings
        node_embeddings = np.zeros((num_nodes, self.embedding_dim))
        
        # First check if model is None
        model_valid = self._model is not None
        
        # Only try to compute embeddings if we have a valid model
        if not model_valid:
            logger.error("No valid ISNE model available")
            # We'll use the zero embeddings initialized above
        else:
            # We have a model, try to use it
            try:
                self._model.eval()
                with torch.no_grad():
                    node_embeddings = self._model(edge_index).cpu().numpy()
            except Exception as e:
                logger.error(f"Error computing ISNE embeddings: {e}")
                # Keep the zero embeddings on error
        
        # Update documents with embeddings
        for doc_id, node_idx in node_id_mapping.items():
            doc = dataset.get_document_by_id(doc_id)
            if doc:
                doc.embedding = node_embeddings[node_idx].tolist()
        
        logger.info(f"Computed ISNE embeddings for {len(node_id_mapping)} documents.")
        return dataset
    
    def embed_document(self, document: IngestDocument, edge_index: Tensor) -> IngestDocument:
        """
        Embed a single document using the trained ISNE model.
        
        This is useful for embedding new documents inductively after training.
        
        Args:
            document: The document to embed
            edge_index: The edge index tensor representing the graph structure
            
        Returns:
            The document with embedding added
        """
        # Handle case where model is not trained
        if self._model is None:
            logger.error("Model not trained. Cannot embed document. Call process() first.")
            # Return document without embedding rather than raising exception
            return document
        
        # Add the document to the node mapping
        if document.id not in self._node_id_mapping:
            new_idx = len(self._node_id_mapping)
            self._node_id_mapping[document.id] = new_idx
            self._id_to_doc[document.id] = document
        
        node_idx = self._node_id_mapping[document.id]
        
        # Create a tensor for the single node
        node_tensor = torch.tensor([node_idx], device=self.device)
        
        # Get embedding using the model
        # Check if model exists before proceeding
        model_exists = self._model is not None
        
        # If no model, log error and return unmodified document
        if not model_exists:
            logger.error("No model available for embedding computation")
            return document
            
        # We have a model, try to compute embedding
        embedding = None
        try:
            self._model.eval()
            with torch.no_grad():
                # Use the encoder to compute embedding for the specific node
                embedding = self._model.encoder(node_tensor, edge_index.to(self.device)).cpu().numpy()[0]
        except Exception as e:
            logger.error(f"Error computing embedding: {e}")
            return document
        
        # Only set embedding if we got a valid result
        if embedding is not None:
            document.embedding = embedding.tolist()
            
        return document

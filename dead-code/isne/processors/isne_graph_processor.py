"""
ISNE-specific graph processor for the ISNE pipeline.

This module provides a specialized processor for building graph representations
tailored for the Inductive Shallow Node Embedding (ISNE) algorithm.
"""

from typing import Dict, List, Optional, Union, Any, Set, Tuple, Callable, cast
import logging
import time
import numpy as np
from datetime import datetime
import torch
from torch import Tensor
import torch.nn.functional as F

from src.isne.types.models import IngestDocument, IngestDataset, DocumentRelation, RelationType
from src.isne.processors.graph_processor import GraphProcessor
from src.isne.processors.base_processor import ProcessorConfig, ProcessorResult

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ISNEGraphProcessor(GraphProcessor):
    """
    Specialized processor for building graph representations for ISNE processing.
    
    This processor extends the base GraphProcessor with additional functionality
    specifically designed for the Inductive Shallow Node Embedding (ISNE) algorithm,
    including specialized edge weighting, feature normalization, and graph structure
    optimizations for ISNE.
    """
    
    def __init__(
        self,
        processor_config: Optional[ProcessorConfig] = None,
        min_edge_weight: float = 0.1,
        include_self_loops: bool = True,
        bidirectional_edges: bool = True,
        max_distance: int = 3,
        normalize_features: bool = True,
        edge_dropout: float = 0.0,
        use_attention_weights: bool = True,
        aggregation_type: str = "mean",
        feature_dim: Optional[int] = None
    ) -> None:
        """
        Initialize the ISNE graph processor.
        
        Args:
            processor_config: Configuration for the processor
            min_edge_weight: Minimum weight for edges to include
            include_self_loops: Whether to include self-loops in the graph
            bidirectional_edges: Whether to create bidirectional edges
            max_distance: Maximum path distance for transitive relationships
            normalize_features: Whether to normalize node features
            edge_dropout: Dropout probability for edges during training
            use_attention_weights: Whether to use attention-based edge weights
            aggregation_type: Type of neighborhood aggregation ('mean', 'sum', 'max')
            feature_dim: Optional dimension to reshape features to
        """
        super().__init__(
            processor_config=processor_config,
            min_edge_weight=min_edge_weight,
            include_self_loops=include_self_loops,
            bidirectional_edges=bidirectional_edges,
            max_distance=max_distance,
            normalize_features=normalize_features
        )
        
        self.edge_dropout = edge_dropout
        self.use_attention_weights = use_attention_weights
        self.aggregation_type = aggregation_type
        self.feature_dim = feature_dim
    
    def process(
        self, 
        documents: List[IngestDocument],
        relations: Optional[List[DocumentRelation]] = None,
        dataset: Optional[IngestDataset] = None
    ) -> ProcessorResult:
        """
        Process documents and relationships into an ISNE-optimized graph representation.
        
        This method extends the base GraphProcessor.process() method with additional
        processing steps specific to ISNE requirements.
        
        Args:
            documents: List of documents to process
            relations: Optional list of relationships between documents
            dataset: Optional dataset containing documents and relationships
            
        Returns:
            ProcessorResult containing document and relationship graphs with ISNE metadata
        """
        start_time = time.time()
        logger.info(f"Building ISNE graph from {len(documents)} documents and {len(relations or [])} relationships")
        
        # Call the parent class's process method to get the base graph
        result = super().process(documents, relations, dataset)
        
        # Extract graph data from the result
        if 'node_features' in result.metadata and 'edge_index' in result.metadata:
            node_features = torch.tensor(result.metadata['node_features'], dtype=torch.float32)
            edge_index = torch.tensor(result.metadata['edge_index'], dtype=torch.int64) if result.metadata['edge_index'] else None
            edge_weights = torch.tensor(result.metadata['edge_weights'], dtype=torch.float32) if result.metadata['edge_weights'] else None
            
            # Apply ISNE-specific processing
            node_features, edge_index, edge_weights = self._process_for_isne(
                node_features, edge_index, edge_weights
            )
            
            # Update the result metadata with ISNE-specific information
            result.metadata.update({
                "processor": "ISNEGraphProcessor",
                "isne_node_features": node_features.tolist(),
                "isne_edge_index": edge_index.tolist() if edge_index is not None else None,
                "isne_edge_weights": edge_weights.tolist() if edge_weights is not None else None,
                "isne_processing_time": time.time() - start_time,
                "isne_edge_dropout": self.edge_dropout,
                "isne_aggregation_type": self.aggregation_type
            })
            
            # Update document metadata with ISNE-specific information
            processed_documents = []
            for i, doc in enumerate(result.documents):
                if i < len(node_features):
                    isne_metadata = {
                        "isne_node_index": i,
                        "isne_processing_time": time.time() - start_time
                    }
                    
                    # Merge with existing metadata
                    updated_metadata = doc.metadata.copy()
                    updated_metadata.update(isne_metadata)
                    
                    # Create updated document
                    updated_doc = IngestDocument(
                        id=doc.id,
                        content=doc.content,
                        source=doc.source,
                        document_type=doc.document_type,
                        title=doc.title,
                        author=doc.author,
                        created_at=doc.created_at,
                        updated_at=doc.updated_at,
                        metadata=updated_metadata,
                        embedding=doc.embedding,
                        embedding_model=doc.embedding_model,
                        chunks=doc.chunks,
                        tags=doc.tags
                    )
                    processed_documents.append(updated_doc)
                else:
                    processed_documents.append(doc)
            
            # Update the result with the processed documents
            result.documents = processed_documents
        
        logger.info(f"ISNE graph processing completed in {time.time() - start_time:.2f}s")
        return result
    
    def _process_for_isne(
        self, 
        node_features: Tensor, 
        edge_index: Optional[Tensor], 
        edge_weights: Optional[Tensor]
    ) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
        """
        Apply ISNE-specific processing to graph data.
        
        Args:
            node_features: Tensor of node features
            edge_index: Tensor of edge indices
            edge_weights: Tensor of edge weights
            
        Returns:
            Tuple of processed (node_features, edge_index, edge_weights)
        """
        # If edge_index is None, we don't have a graph
        if edge_index is None or edge_index.shape[1] == 0:
            logger.warning("No edges found for ISNE processing, using identity graph")
            # Create identity graph (only self-loops)
            num_nodes = node_features.shape[0]
            edge_index = torch.stack([
                torch.arange(num_nodes),
                torch.arange(num_nodes)
            ], dim=0)
            edge_weights = torch.ones(num_nodes)
        
        # Reshape features if a specific dimension is required
        if self.feature_dim is not None and node_features.shape[1] != self.feature_dim:
            logger.info(f"Reshaping node features from {node_features.shape[1]} to {self.feature_dim}")
            
            # If target dimension is smaller, use dimensionality reduction (PCA-like)
            if self.feature_dim < node_features.shape[1]:
                # Simple linear projection
                projection = torch.nn.Linear(node_features.shape[1], self.feature_dim)
                with torch.no_grad():
                    node_features = projection(node_features)
            else:
                # If target dimension is larger, pad with zeros
                padding = torch.zeros(
                    node_features.shape[0], 
                    self.feature_dim - node_features.shape[1]
                )
                node_features = torch.cat([node_features, padding], dim=1)
        
        # Apply normalization if needed
        if self.normalize_features:
            node_features = F.normalize(node_features, p=2, dim=1)
        
        # Compute attention-based edge weights if requested
        if self.use_attention_weights and edge_weights is not None:
            # Get source and target nodes for each edge
            src, dst = edge_index
            
            # Compute dot product similarity between node features
            sim_scores = torch.sum(node_features[src] * node_features[dst], dim=1)
            
            # Apply softmax to get attention weights
            attention_scores = F.softmax(sim_scores, dim=0)
            
            # Combine with existing edge weights
            edge_weights = edge_weights * attention_scores
        
        # Apply edge dropout during training if requested
        if self.edge_dropout > 0 and self.training:
            # Create a dropout mask
            dropout_mask = torch.rand(edge_index.shape[1]) >= self.edge_dropout
            
            # Apply the mask to edges and weights
            edge_index = edge_index[:, dropout_mask]
            if edge_weights is not None:
                edge_weights = edge_weights[dropout_mask]
        
        return node_features, edge_index, edge_weights
    
    def calculate_neighborhood_embeddings(
        self,
        node_features: Tensor,
        edge_index: Tensor,
        edge_weights: Optional[Tensor] = None
    ) -> Tensor:
        """
        Calculate neighborhood embeddings using ISNE aggregation.
        
        Args:
            node_features: Node feature tensor
            edge_index: Edge index tensor
            edge_weights: Optional edge weight tensor
            
        Returns:
            Tensor of neighborhood embeddings
        """
        # Extract source and target nodes
        src, dst = edge_index
        
        # Initialize neighborhood embeddings
        num_nodes = node_features.shape[0]
        neighborhood_embeddings = torch.zeros_like(node_features)
        
        # Use scatter operations for efficient aggregation
        if self.aggregation_type == "mean":
            # Weighted mean aggregation
            if edge_weights is not None:
                # Multiply features by weights
                weighted_features = node_features[src] * edge_weights.unsqueeze(1)
                
                # Sum weighted features for each target node
                neighborhood_embeddings.index_add_(0, dst, weighted_features)
                
                # Calculate sum of weights for each node for normalization
                weight_sums = torch.zeros(num_nodes, device=node_features.device)
                weight_sums.index_add_(0, dst, edge_weights)
                
                # Avoid division by zero
                weight_sums = torch.clamp(weight_sums, min=1e-8).unsqueeze(1)
                
                # Normalize by weight sum
                neighborhood_embeddings = neighborhood_embeddings / weight_sums
            else:
                # Unweighted mean
                for i in range(num_nodes):
                    # Get neighbors of node i
                    mask = dst == i
                    neighbors = src[mask]
                    
                    if len(neighbors) > 0:
                        # Simple mean of neighbor features
                        neighborhood_embeddings[i] = torch.mean(node_features[neighbors], dim=0)
        
        elif self.aggregation_type == "sum":
            # Sum aggregation
            if edge_weights is not None:
                # Multiply features by weights
                weighted_features = node_features[src] * edge_weights.unsqueeze(1)
                
                # Sum weighted features for each target node
                neighborhood_embeddings.index_add_(0, dst, weighted_features)
            else:
                # Simple sum
                neighborhood_embeddings.index_add_(0, dst, node_features[src])
        
        elif self.aggregation_type == "max":
            # Max aggregation (more complex with scatter)
            # We'll use a manual approach for simplicity
            for i in range(num_nodes):
                # Get neighbors of node i
                mask = dst == i
                neighbors = src[mask]
                
                if len(neighbors) > 0:
                    # Max of neighbor features
                    neighborhood_embeddings[i] = torch.max(node_features[neighbors], dim=0)[0]
        
        else:
            raise ValueError(f"Unsupported aggregation type: {self.aggregation_type}")
        
        return neighborhood_embeddings

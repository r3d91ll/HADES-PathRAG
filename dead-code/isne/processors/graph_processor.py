"""
Graph processor for the ISNE pipeline.

This module provides a processor for building graph representations from
documents and their relationships in the ISNE pipeline.
"""

from typing import Dict, List, Optional, Union, Any, Set, Tuple, Callable, cast
import logging
import time
import numpy as np
from datetime import datetime
import torch
from torch import Tensor

from src.isne.types.models import IngestDocument, IngestDataset, DocumentRelation, RelationType
from src.isne.processors.base_processor import BaseProcessor, ProcessorConfig, ProcessorResult

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GraphProcessor(BaseProcessor):
    """
    Processor for building graph representations from documents and relationships.
    
    This processor transforms documents and their relationships into graph data
    structures suitable for processing with the ISNE model.
    """
    
    def __init__(
        self,
        processor_config: Optional[ProcessorConfig] = None,
        min_edge_weight: float = 0.1,
        include_self_loops: bool = True,
        bidirectional_edges: bool = True,
        max_distance: int = 3,
        normalize_features: bool = True
    ) -> None:
        """
        Initialize the graph processor.
        
        Args:
            processor_config: Configuration for the processor
            min_edge_weight: Minimum weight for edges to include
            include_self_loops: Whether to include self-loops in the graph
            bidirectional_edges: Whether to create bidirectional edges
            max_distance: Maximum path distance for transitive relationships
            normalize_features: Whether to normalize node features
        """
        super().__init__(processor_config)
        
        self.min_edge_weight = min_edge_weight
        self.include_self_loops = include_self_loops
        self.bidirectional_edges = bidirectional_edges
        self.max_distance = max_distance
        self.normalize_features = normalize_features
    
    def process(
        self, 
        documents: List[IngestDocument],
        relations: Optional[List[DocumentRelation]] = None,
        dataset: Optional[IngestDataset] = None
    ) -> ProcessorResult:
        """
        Process documents and relationships into a graph representation.
        
        Args:
            documents: List of documents to process
            relations: Optional list of relationships between documents
            dataset: Optional dataset containing documents and relationships
            
        Returns:
            ProcessorResult containing document and relationship graphs
        """
        start_time = time.time()
        logger.info(f"Building graph from {len(documents)} documents and {len(relations or [])} relationships")
        
        # Validate documents have embeddings
        docs_with_embeddings = [doc for doc in documents if doc.embedding is not None]
        if len(docs_with_embeddings) < len(documents):
            logger.warning(f"{len(documents) - len(docs_with_embeddings)} documents do not have embeddings")
        
        # Build node and edge data structures
        node_features, edge_index, edge_weights = self._build_graph(docs_with_embeddings, relations or [])
        
        # Add graph data to document metadata
        processed_documents = []
        for i, doc in enumerate(documents):
            # Create a copy of the document with graph metadata
            if i < len(docs_with_embeddings):
                graph_metadata = {
                    "graph_node_index": i,
                    "graph_processing_time": time.time() - start_time
                }
                
                # Merge with existing metadata
                updated_metadata = doc.metadata.copy()
                updated_metadata.update(graph_metadata)
                
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
                # Keep document without graph metadata
                processed_documents.append(doc)
        
        # Create updated dataset if provided
        updated_dataset = None
        if dataset:
            updated_dataset = IngestDataset(
                id=dataset.id,
                name=dataset.name,
                description=dataset.description,
                metadata={
                    **dataset.metadata,
                    "graph_node_count": len(node_features),
                    "graph_edge_count": edge_index.shape[1] if edge_index is not None else 0,
                    "graph_processing_time": time.time() - start_time
                },
                created_at=dataset.created_at,
                updated_at=datetime.now()
            )
            
            # Add processed documents
            for doc in processed_documents:
                updated_dataset.add_document(doc)
            
            # Add relationships
            if relations:
                for rel in relations:
                    updated_dataset.add_relation(rel)
        
        # Return result with graph data in metadata
        elapsed_time = time.time() - start_time
        logger.info(f"Graph built in {elapsed_time:.2f}s with {len(node_features)} nodes and "
                   f"{edge_index.shape[1] if edge_index is not None else 0} edges")
        
        return ProcessorResult(
            documents=processed_documents,
            relations=relations or [],
            dataset=updated_dataset,
            errors=[],
            metadata={
                "processor": "GraphProcessor",
                "node_count": len(node_features),
                "edge_count": edge_index.shape[1] if edge_index is not None else 0,
                "node_features_shape": node_features.shape,
                "processing_time": elapsed_time,
                "node_features": node_features.tolist() if isinstance(node_features, np.ndarray) else node_features,
                "edge_index": edge_index.tolist() if edge_index is not None else None,
                "edge_weights": edge_weights.tolist() if edge_weights is not None else None
            }
        )
    
    def _build_graph(
        self, 
        documents: List[IngestDocument],
        relations: List[DocumentRelation]
    ) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        """
        Build a graph from documents and relationships.
        
        Args:
            documents: List of documents to build a graph from
            relations: List of relationships between documents
            
        Returns:
            Tuple of (node_features, edge_index, edge_weights)
        """
        # Map document IDs to node indices
        doc_index: Dict[str, int] = {doc.id: i for i, doc in enumerate(documents)}
        
        # Create node features matrix from document embeddings
        node_features = []
        for doc in documents:
            if doc.embedding is not None:
                # Convert to numpy array if it's a list
                if isinstance(doc.embedding, list):
                    embedding = np.array(doc.embedding, dtype=np.float32)
                else:
                    embedding = doc.embedding
                
                node_features.append(embedding)
            else:
                # Use random embedding as fallback (not recommended)
                logger.warning(f"Document {doc.id} has no embedding, using random fallback")
                random_embedding = np.random.randn(768)  # Common embedding dimension
                node_features.append(random_embedding)
        
        # Convert to numpy array and normalize if requested
        node_features_np = np.array(node_features, dtype=np.float32)
        if self.normalize_features:
            row_norms = np.linalg.norm(node_features_np, axis=1, keepdims=True)
            node_features_np = node_features_np / np.maximum(row_norms, 1e-8)
        
        # Create edge index and weights from relationships
        edges = []
        weights = []
        
        # Process explicit relationships
        for rel in relations:
            # Skip if source or target is not in our documents
            if rel.source_id not in doc_index or rel.target_id not in doc_index:
                continue
            
            # Get node indices
            src_idx = doc_index[rel.source_id]
            dst_idx = doc_index[rel.target_id]
            
            # Add edge if weight is above threshold
            if rel.weight >= self.min_edge_weight:
                edges.append((src_idx, dst_idx))
                weights.append(rel.weight)
                
                # Add bidirectional edge if specified
                if self.bidirectional_edges or rel.bidirectional:
                    edges.append((dst_idx, src_idx))
                    weights.append(rel.weight)
        
        # Add self-loops if requested
        if self.include_self_loops:
            for i in range(len(documents)):
                edges.append((i, i))
                weights.append(1.0)
        
        # Convert to tensor format expected by PyTorch Geometric
        edge_index = None
        edge_weights = None
        
        if edges:
            # Convert to numpy array
            edges_np = np.array(edges, dtype=np.int64).T
            weights_np = np.array(weights, dtype=np.float32)
            
            # Convert to PyTorch tensors
            edge_index = torch.from_numpy(edges_np)
            edge_weights = torch.from_numpy(weights_np)
        else:
            # Create empty tensors with the right shape
            edge_index = torch.zeros((2, 0), dtype=torch.int64)
            edge_weights = torch.zeros(0, dtype=torch.float32)
        
        # Convert node features to tensor
        node_features_tensor = torch.from_numpy(node_features_np)
        
        return node_features_tensor, edge_index, edge_weights
    
    def get_graph_data(
        self, 
        documents: List[IngestDocument],
        relations: List[DocumentRelation]
    ) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        """
        Extract graph data from documents and relationships.
        
        Args:
            documents: List of documents to extract graph data from
            relations: List of relationships between documents
            
        Returns:
            Tuple of (node_features, edge_index, edge_weights)
        """
        return self._build_graph(documents, relations)
    
    def calculate_similarity_relationships(
        self,
        documents: List[IngestDocument],
        similarity_threshold: float = 0.7,
        max_relationships_per_doc: int = 5
    ) -> List[DocumentRelation]:
        """
        Calculate document relationships based on embedding similarity.
        
        Args:
            documents: List of documents to calculate relationships for
            similarity_threshold: Minimum similarity score for relationships
            max_relationships_per_doc: Maximum relationships per document
            
        Returns:
            List of generated DocumentRelation objects
        """
        # Filter documents with embeddings
        docs_with_embeddings = [doc for doc in documents if doc.embedding is not None]
        
        if not docs_with_embeddings:
            logger.warning("No documents with embeddings found")
            return []
        
        # Extract embeddings
        embeddings = []
        for doc in docs_with_embeddings:
            if isinstance(doc.embedding, list):
                embeddings.append(np.array(doc.embedding))
            else:
                embeddings.append(doc.embedding)
        
        # Convert to numpy array and normalize
        embeddings_np = np.array(embeddings)
        norms = np.linalg.norm(embeddings_np, axis=1, keepdims=True)
        normalized_embeddings = embeddings_np / np.maximum(norms, 1e-8)
        
        # Calculate similarity matrix
        similarity_matrix = np.matmul(normalized_embeddings, normalized_embeddings.T)
        
        # Create relationships from similarities
        relations = []
        
        for i, doc in enumerate(docs_with_embeddings):
            # Get top similar documents (excluding self)
            similarities = similarity_matrix[i].copy()
            similarities[i] = -1  # Exclude self
            
            # Get indices of top similar documents
            top_indices = np.argsort(-similarities)[:max_relationships_per_doc]
            
            # Create relationships for similar documents
            for idx in top_indices:
                similarity = similarities[idx]
                if similarity >= similarity_threshold:
                    target_doc = docs_with_embeddings[idx]
                    
                    # Create relationship
                    relation = DocumentRelation(
                        source_id=doc.id,
                        target_id=target_doc.id,
                        relation_type=RelationType.SIMILAR_TO,
                        weight=float(similarity),
                        bidirectional=True,
                        metadata={
                            "relationship_source": "embedding_similarity",
                            "similarity_score": float(similarity)
                        }
                    )
                    relations.append(relation)
        
        logger.info(f"Generated {len(relations)} similarity relationships")
        return relations

"""
PathRAG core implementation.

This module contains the main PathRAG class that implements the flow-based
path pruning algorithm for retrieval augmented generation.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any, Union, Type, Iterator, cast
from contextlib import contextmanager

import logging
import time
import numpy as np
import networkx as nx

from ..embeddings.base import BaseEmbedder
from ..graph.base import BaseGraph, Path, NodeID
from ..storage.base import BaseVectorStorage, BaseDocumentStorage, BaseGraphStorage
from ..storage.interfaces import MetadataQuery, MetadataCondition, QueryOperator
from ..utils.text import extract_entities
from .config import PathRAGConfig
from .path_pruning import PathPruningConfig, extract_paths_with_pruning

logger = logging.getLogger(__name__)


@dataclass
class PathRAGResult:
    """Results from a PathRAG query."""
    query: str
    nodes: List[Dict[str, Any]]
    paths: List[Dict[str, Any]]
    execution_time_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class PathRAG:
    """PathRAG implementation with modular components for embedding, graph, and storage.
    
    PathRAG uses a flow-based path pruning algorithm to extract relevant paths
    between nodes in a knowledge graph for retrieval augmented generation.
    
    This class integrates the embedder, graph, and storage components into a
    cohesive system that supports both inductive learning with ISNE and
    traditional embedding approaches.
    """
    
    def __init__(
        self,
        embedder: BaseEmbedder,
        graph: BaseGraph,
        vector_storage: BaseVectorStorage,
        document_storage: BaseDocumentStorage,
        graph_storage: BaseGraphStorage,
        config: Optional[PathRAGConfig] = None
    ) -> None:
        """
        Initialize PathRAG with components and configuration.
        
        Args:
            embedder: Embedder implementation for node embeddings
            graph: Graph implementation for path extraction
            vector_storage: Vector storage for embeddings
            document_storage: Document storage for text content
            graph_storage: Graph storage for node and edge data
            config: Optional configuration, uses default if not provided
        """
        self.embedder = embedder
        self.graph = graph
        self.vector_storage = vector_storage
        self.document_storage = document_storage
        self.graph_storage = graph_storage
        self.config = config or PathRAGConfig()
        
        # Initialize components
        self.vector_storage.initialize()
        self.document_storage.initialize()
        self.graph_storage.initialize()
        
        logger.info("PathRAG initialized with %s embedder", self.config.embedding_model)
    
    def add_document(self, doc_id: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> List[NodeID]:
        """
        Add a document to the system, creating nodes for its content.
        
        This method processes a document into entities, creates nodes for them,
        and updates the graph structure. In inductive mode, it generates embeddings
        immediately. In retrained mode, embeddings must be explicitly generated
        later by calling retrain_embeddings().
        
        Args:
            doc_id: Unique identifier for the document
            content: Text content of the document
            metadata: Optional metadata to store with the document
            
        Returns:
            List of created node IDs
        """
        start_time = time.time()
        logger.info(f"Adding document {doc_id} with {len(content)} characters")
        
        # Combine document metadata
        doc_metadata = metadata or {}
        doc_metadata.update({
            "doc_id": doc_id,
            "content_length": len(content),
            "processing_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        })
        
        # Store document
        self.document_storage.store_document(doc_id, content, doc_metadata)
        
        # Process document into nodes
        nodes = self._process_document_to_nodes(doc_id, content, doc_metadata)
        
        # Process relationships between nodes
        edges = self._extract_relationships(nodes, doc_id)
        
        # Add nodes and edges to graph
        node_ids: List[NodeID] = []
        nodes_created = 0
        nodes_updated = 0
        
        for node_id, node_attrs in nodes.items():
            # Check if node already exists
            existing_node = self.graph_storage.get_node(node_id)
            if existing_node:
                # Update existing node
                merged_attrs = {**existing_node, **node_attrs}
                
                # Add source documents list if not present
                if "source_docs" not in merged_attrs:
                    merged_attrs["source_docs"] = [doc_id]
                elif doc_id not in merged_attrs["source_docs"]:
                    merged_attrs["source_docs"].append(doc_id)
                
                # Update node by removing and re-adding with updated attributes
                self.graph.add_node(node_id, merged_attrs)  # NetworkX will update existing nodes
                self.graph_storage.store_node(node_id, merged_attrs)  # Use store_node as update mechanism
                nodes_updated += 1
            else:
                # Add source documents list
                if "source_docs" not in node_attrs:
                    node_attrs["source_docs"] = [doc_id]
                
                # Create new node
                self.graph.add_node(node_id, node_attrs)
                self.graph_storage.store_node(node_id, node_attrs)
                nodes_created += 1
            
            node_ids.append(node_id)
        
        # Add edges to graph
        edges_created = 0
        for edge in edges:
            source = edge["source"]
            target = edge["target"]
            edge_type = edge["type"]
            edge_attrs = {**edge, "doc_id": doc_id}
            
            # Generate edge ID if not provided
            if "id" not in edge_attrs:
                edge_attrs["id"] = f"{source}_{edge_type}_{target}"
            
            # Add to graph
            relation_type = edge_attrs.get("type", "related_to")
            self.graph.add_edge(
                source, 
                target, 
                relation_type,
                weight=edge_attrs.get("weight", 1.0),
                attributes=edge_attrs
            )
            self.graph_storage.store_edge(
                source, 
                target, 
                relation_type,
                weight=edge_attrs.get("weight", 1.0),
                attributes=edge_attrs
            )
            edges_created += 1
        
        # Generate embeddings in inductive mode
        if self.config.mode == "inductive":
            self._generate_inductive_embeddings(node_ids)
        
        # Log processing stats
        elapsed = time.time() - start_time
        logger.info(f"Document {doc_id} processed: {nodes_created} nodes created, {nodes_updated} updated, {edges_created} edges created in {elapsed:.2f}s")
        
        return node_ids
    
    def _generate_inductive_embeddings(self, node_ids: List[NodeID]) -> None:
        """
        Generate embeddings for nodes in inductive mode.
        
        Args:
            node_ids: List of node IDs to generate embeddings for
        """
        for node_id in node_ids:
            # Get node attributes
            node_attrs = self.graph.get_node(node_id)
            if not node_attrs:
                continue
            
            # Get neighbors for embedding
            neighbors = self.graph.get_neighbors(node_id)
            
            # Generate embedding
            embedding = self.embedder.encode(node_id, neighbors)
            
            # Store embedding
            self.vector_storage.store_embedding(node_id, embedding, node_attrs)
    
    def retrain_embeddings(self) -> None:
        """
        Retrain all embeddings from scratch.
        
        This method extracts all nodes and their connections from the graph,
        then retrains the embedding model on the entire graph. This should be
        called periodically in "retrained" mode to update embeddings as the
        graph evolves.
        """
        start_time = time.time()
        logger.info("Retraining embeddings for all nodes")
        
        # Extract networkx graph for training
        nx_graph = self.graph.to_networkx()
        node_count = len(nx_graph.nodes)
        
        logger.info(f"Graph extracted for training: {node_count} nodes, {len(nx_graph.edges)} edges")
        
        # Retrain embedder
        self.embedder.fit(nx_graph)
        
        # Update all embeddings in vector storage in batches
        batch_size = 100
        batch_items = []
        processed = 0
        
        for node in nx_graph.nodes:
            node_id = str(node)
            neighbors = self.graph.get_neighbors(node_id)
            
            # Generate new embedding
            embedding = self.embedder.encode(node_id, neighbors)
            
            # Get node attributes
            node_attrs = self.graph.get_node(node_id)
            
            # Update embedding in storage
            if node_attrs:
                batch_items.append((node_id, embedding, node_attrs))
            
            # Process in batches
            if len(batch_items) >= batch_size:
                # Store embeddings individually if bulk method not available
                for node_id, embedding, metadata in batch_items:
                    self.vector_storage.store_embedding(node_id, embedding, metadata)
                processed += len(batch_items)
                logger.info(f"Progress: {processed}/{node_count} embeddings updated")
                batch_items = []
        
        # Process remaining items
        if batch_items:
            # Store embeddings individually if bulk method not available
            for node_id, embedding, metadata in batch_items:
                self.vector_storage.store_embedding(node_id, embedding, metadata)
            processed += len(batch_items)
        
        elapsed = time.time() - start_time
        logger.info(f"Embedding retraining completed: {processed} embeddings updated in {elapsed:.2f}s")
    
    def _extract_relationships(self, nodes: Dict[NodeID, Dict[str, Any]], doc_id: str) -> List[Dict[str, Any]]:
        """
        Extract relationships between nodes.
        
        This method analyzes nodes extracted from a document and infers
        relationships between them. This is a simplified implementation that
        creates basic connections based on co-occurrence.
        
        Args:
            nodes: Dictionary mapping node IDs to node attributes
            doc_id: Document ID that these nodes came from
            
        Returns:
            List of edge attributes representing relationships
        """
        edges = []
        node_ids = list(nodes.keys())
        
        # Simple co-occurrence relationship creation
        # In a real implementation, this would use NLP to extract actual relationships
        for i, source_id in enumerate(node_ids):
            source_type = nodes[source_id].get("type", "ENTITY")
            
            for target_id in node_ids[i+1:]:
                target_type = nodes[target_id].get("type", "ENTITY")
                
                # Skip self-connections
                if source_id == target_id:
                    continue
                
                # Create a generic relationship
                edge = {
                    "source": source_id,
                    "target": target_id,
                    "type": "RELATED_TO",
                    "weight": 1.0,
                    "doc_id": doc_id,
                    "source_type": source_type,
                    "target_type": target_type,
                }
                
                edges.append(edge)
        
        return edges
    
    def query(self, query_text: str, top_k: Optional[int] = None, metadata_filter: Optional[Dict[str, Any]] = None) -> PathRAGResult:
        """
        Process a query through PathRAG to retrieve relevant paths.
        
        This method generates an embedding for the query text, retrieves similar nodes,
        and then extracts and scores paths between those nodes using the flow-based
        pruning algorithm. The results include both the individual nodes and the
        extracted paths with reliability scores.
        
        Args:
            query_text: Query text to process
            top_k: Optional override for number of results
            metadata_filter: Optional metadata filter for nodes
            
        Returns:
            PathRAGResult with retrieved nodes, paths, and metrics
        """
        start_time = time.time()
        logger.info(f"Processing query: '{query_text}'")
        
        k = top_k or self.config.top_k_nodes
        
        # Process metadata filter if provided
        filter_conditions = None
        if metadata_filter:
            filter_conditions = []
            for key, value in metadata_filter.items():
                filter_conditions.append(MetadataCondition(
                    field=key,
                    operator=QueryOperator.EQUALS,
                    value=value
                ))
        
        # Generate query embedding
        query_embedding = self.embedder.encode_text(query_text)
        
        # Track query performance
        embedding_time = time.time() - start_time
        
        # Retrieve similar nodes
        if filter_conditions:
            # Use hybrid search with metadata filtering
            metadata_query = MetadataQuery(
                conditions=filter_conditions,
                combine_operator="AND"
            )
            # Fall back to standard search if hybrid search not available
            if hasattr(self.vector_storage, 'hybrid_search'):
                # Use hybrid search if available (for enhanced storage implementations)
                similar_nodes = self.vector_storage.hybrid_search(
                    query_embedding=query_embedding,
                    metadata_query=metadata_query,
                    k=k
                )
            else:
                # Fallback to regular search
                similar_nodes = self.vector_storage.retrieve_similar(
                    query_embedding=query_embedding,
                    k=k,
                    filter_metadata={"conditions": filter_conditions, "combine_operator": "AND"}
                )
            similar_nodes = [(node_id, score, metadata) for node_id, score, metadata in similar_nodes]
        else:
            # Use standard vector search
            similar_nodes = self.vector_storage.retrieve_similar(
                query_embedding, 
                k=k
            )
        
        # Track retrieval performance
        retrieval_time = time.time() - start_time - embedding_time
        
        # Create initial resource levels based on similarity scores
        node_resources = {}
        for node_id, score, _ in similar_nodes:
            # Use similarity score as initial resource level (normalized to 0-1)
            node_resources[node_id] = score
        
        # Create path pruning config from main config
        pruning_config = PathPruningConfig(
            max_path_length=self.config.max_path_length,
            decay_rate=self.config.decay_rate,
            pruning_threshold=self.config.pruning_threshold,
            max_paths_per_node_pair=self.config.max_paths_per_node_pair,
            use_diversity=self.config.use_diverse_paths
        )
        
        # Extract paths using flow-based pruning
        nx_graph = self.graph.to_networkx()
        paths = extract_paths_with_pruning(nx_graph, node_resources, pruning_config)
        
        # Track path extraction performance
        path_time = time.time() - start_time - embedding_time - retrieval_time
        
        # Limit to top_k_paths
        top_paths = paths[:self.config.top_k_paths]
        
        # Enrich paths with additional node and edge information
        enriched_paths = []
        for path in top_paths:
            # Fetch full node details for the path
            nodes_detail = []
            for node_id in path.nodes:
                node_detail = self.graph_storage.get_node(node_id) or {}
                nodes_detail.append({
                    "id": node_id,
                    "attributes": node_detail
                })
            
            # Create enriched path record
            enriched_path = {
                "nodes": nodes_detail,
                "edges": path.edges,
                "reliability": path.reliability
            }
            enriched_paths.append(enriched_path)
        
        # Enrich node results with additional details
        enriched_nodes = []
        for node_id, score, attrs in similar_nodes:
            # Get full node details if needed
            if not attrs or len(attrs) < 2:  # Only has _id/_key
                attrs = self.graph_storage.get_node(node_id) or {}
            
            enriched_nodes.append({
                "id": node_id,
                "similarity": score,
                "attributes": attrs
            })
        
        # Calculate total execution time
        total_time = time.time() - start_time
        
        # Create performance metadata
        metadata = {
            "timing": {
                "embedding_ms": embedding_time * 1000,
                "retrieval_ms": retrieval_time * 1000,
                "path_extraction_ms": path_time * 1000,
                "total_ms": total_time * 1000
            },
            "stats": {
                "node_count": len(similar_nodes),
                "path_count": len(paths),
                "returned_paths": len(top_paths)
            },
            "config": {
                "max_path_length": self.config.max_path_length,
                "decay_rate": self.config.decay_rate,
                "pruning_threshold": self.config.pruning_threshold
            }
        }
        
        # Log query performance
        logger.info(f"Query processed in {total_time:.2f}s: {len(similar_nodes)} nodes, {len(paths)} paths extracted")
        
        return PathRAGResult(
            query=query_text,
            nodes=enriched_nodes,
            paths=enriched_paths,
            execution_time_ms=total_time * 1000,
            metadata=metadata
        )
    
    def close(self) -> None:
        """
        Close all connections and release resources.
        """
        logger.info("Closing PathRAG resources")
        self.vector_storage.close()
        self.document_storage.close()
        self.graph_storage.close()
    
    def _process_document_to_nodes(
        self, 
        doc_id: str, 
        content: str, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[NodeID, Dict[str, Any]]:
        """
        Process a document into nodes for the graph.
        
        This method extracts entities from the document content and creates
        nodes for each entity, as well as a document node that connects to all entities.
        
        Args:
            doc_id: Document ID
            content: Document content
            metadata: Optional document metadata
            
        Returns:
            Dictionary mapping node IDs to node attributes
        """
        nodes: Dict[NodeID, Dict[str, Any]] = {}
        
        # Create document node
        doc_node_id = f"doc-{doc_id}"
        
        # Get document metadata
        doc_metadata = metadata or {}
        
        # Create document node attributes
        doc_node_attrs = {
            "type": "DOCUMENT",
            "content": content[:1000],  # Store truncated content
            "doc_id": doc_id,
            "content_length": len(content),
            **doc_metadata
        }
        
        # Add document node
        nodes[doc_node_id] = doc_node_attrs
        
        # Extract entities from content
        entities = extract_entities(content)
        
        # Create entity nodes
        for entity in entities:
            entity_id = f"entity-{doc_id}-{len(nodes)}"
            if "id" in entity:
                entity_id = entity["id"]
            
            # Create entity node attributes
            entity_attrs = {
                "type": entity.get("type", "ENTITY"),
                "name": entity.get("name", ""),
                "text": entity.get("text", ""),
                "doc_id": doc_id,
                "mention_count": entity.get("mention_count", 1),
            }
            
            # Add additional entity attributes
            for key, value in entity.items():
                if key not in ["id", "type", "name", "text"]:
                    entity_attrs[key] = value
            
            # Add entity node
            nodes[entity_id] = entity_attrs
        
        return nodes
    
    @contextmanager
    def session(self) -> Iterator['PathRAG']:
        """
        Context manager for PathRAG session.
        
        This provides a convenient way to ensure resources are closed
        after using PathRAG.
        
        Returns:
            An iterator yielding the PathRAG instance
        
        Usage:
            with pathrag.session():
                result = pathrag.query("...")        
        """
        try:
            yield self
        finally:
            self.close()

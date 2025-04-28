"""
Path ranking algorithm for HADES-PathRAG.

This module implements the PathRAG algorithm for ranking paths in code graphs
based on semantic relevance, path length, and edge strength.
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple, Set, cast
import networkx as nx
from pathlib import Path

from src.isne.types.models import IngestDocument, DocumentRelation
from src.types.common import NodeData, EdgeData, PathRankingConfig

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PathRanker:
    """
    Implementation of the PathRAG algorithm for ranking paths in code graphs.
    
    This class computes path rankings based on a weighted combination of
    semantic relevance, path length, and edge strength.
    """
    
    def __init__(
        self,
        semantic_weight: float = 0.7,
        path_length_weight: float = 0.1,
        edge_strength_weight: float = 0.2,
        max_path_length: int = 5,
        max_paths: int = 20
    ) -> None:
        """
        Initialize the path ranker.
        
        Args:
            semantic_weight: Weight for semantic relevance (default: 0.7)
            path_length_weight: Weight for path length (default: 0.1)
            edge_strength_weight: Weight for edge strength (default: 0.2)
            max_path_length: Maximum length of paths to consider (default: 5)
            max_paths: Maximum number of paths to return (default: 20)
        """
        self.semantic_weight = semantic_weight
        self.path_length_weight = path_length_weight
        self.edge_strength_weight = edge_strength_weight
        self.max_path_length = max_path_length
        self.max_paths = max_paths
        
        # Validate weights sum to 1.0
        total_weight = semantic_weight + path_length_weight + edge_strength_weight
        if not np.isclose(total_weight, 1.0):
            logger.warning(f"Weights do not sum to 1.0: {total_weight}")
            # Normalize weights
            self.semantic_weight /= total_weight
            self.path_length_weight /= total_weight
            self.edge_strength_weight /= total_weight
            logger.info(f"Normalized weights: semantic={self.semantic_weight}, "
                       f"path_length={self.path_length_weight}, "
                       f"edge_strength={self.edge_strength_weight}")
    
    @classmethod
    def from_config(cls, config: Optional[PathRankingConfig] = None) -> "PathRanker":
        """
        Create a PathRanker from configuration.
        
        Args:
            config: Path ranking configuration
            
        Returns:
            Initialized PathRanker
        """
        if config is None:
            return cls()
        
        # Extract config values with defaults
        semantic_weight = config.get("semantic_weight", 0.7)
        path_length_weight = config.get("path_length_weight", 0.1)
        edge_strength_weight = config.get("edge_strength_weight", 0.2)
        max_path_length = config.get("max_path_length", 5)
        max_paths = config.get("max_paths", 20)
        
        return cls(
            semantic_weight=semantic_weight,
            path_length_weight=path_length_weight,
            edge_strength_weight=edge_strength_weight,
            max_path_length=max_path_length,
            max_paths=max_paths
        )
    
    def build_graph(
        self, 
        documents: List[IngestDocument],
        relations: List[DocumentRelation]
    ) -> nx.DiGraph:
        """
        Build a directed graph from documents and relations.
        
        Args:
            documents: List of documents
            relations: List of relations between documents
            
        Returns:
            NetworkX DiGraph representing the document graph
        """
        graph = nx.DiGraph()
        
        # Add nodes
        for doc in documents:
            node_data = {
                "id": doc.id,
                "content": doc.content,
                "title": doc.title,
                "source": doc.source,
                "document_type": doc.document_type,
                "embedding": doc.embedding,
                "metadata": doc.metadata
            }
            graph.add_node(doc.id, **node_data)
        
        # Add edges
        for rel in relations:
            edge_data = {
                "type": rel.relation_type.value,
                "weight": rel.weight,
                "bidirectional": rel.bidirectional,
                "metadata": rel.metadata
            }
            graph.add_edge(rel.source_id, rel.target_id, **edge_data)
            
            # Add reverse edge if bidirectional
            if rel.bidirectional:
                graph.add_edge(rel.target_id, rel.source_id, **edge_data)
        
        return graph
    
    def compute_semantic_similarity(
        self,
        query_embedding: List[float],
        document_embedding: Optional[List[float]]
    ) -> float:
        """
        Compute semantic similarity between query and document.
        
        Args:
            query_embedding: Query embedding vector
            document_embedding: Document embedding vector
            
        Returns:
            Similarity score (cosine similarity)
        """
        if document_embedding is None:
            return 0.0
        
        # Convert to numpy arrays
        query_np = np.array(query_embedding)
        doc_np = np.array(document_embedding)
        
        # Compute cosine similarity
        dot_product = np.dot(query_np, doc_np)
        query_norm = np.linalg.norm(query_np)
        doc_norm = np.linalg.norm(doc_np)
        
        # Avoid division by zero
        if query_norm == 0 or doc_norm == 0:
            return 0.0
        
        return dot_product / (query_norm * doc_norm)
    
    def normalize_path_length_score(self, path_length: int) -> float:
        """
        Normalize path length to a score between 0 and 1.
        
        Shorter paths get higher scores.
        
        Args:
            path_length: Length of the path
            
        Returns:
            Normalized score
        """
        if path_length <= 1:
            return 1.0
        
        # Discount factor increases with path length
        return 1.0 / path_length
    
    def compute_edge_strength(self, path: List[str], graph: nx.DiGraph) -> float:
        """
        Compute average edge strength for a path.
        
        Args:
            path: List of node IDs in the path
            graph: NetworkX DiGraph
            
        Returns:
            Average edge strength
        """
        if len(path) <= 1:
            return 0.0
        
        # Get edge weights along the path
        edge_weights = []
        for i in range(len(path) - 1):
            source = path[i]
            target = path[i + 1]
            weight = graph.edges[source, target].get("weight", 1.0)
            edge_weights.append(weight)
        
        # Return average edge weight
        return sum(edge_weights) / len(edge_weights)
    
    def rank_paths(
        self,
        query_embedding: List[float],
        source_node: str,
        target_nodes: List[str],
        graph: nx.DiGraph
    ) -> List[Dict[str, Any]]:
        """
        Rank paths from source to target nodes using the PathRAG algorithm.
        
        Args:
            query_embedding: Query embedding for semantic similarity
            source_node: Source node ID
            target_nodes: List of target node IDs
            graph: NetworkX DiGraph
            
        Returns:
            List of ranked paths with scores
        """
        ranked_paths = []
        
        # Find paths to each target node
        for target in target_nodes:
            # Skip if source and target are the same
            if source_node == target:
                continue
            
            # Find paths up to max_path_length
            try:
                # Use simple_paths to get all paths up to max_path_length
                paths = list(nx.all_simple_paths(
                    graph, source_node, target, cutoff=self.max_path_length
                ))
                
                # Skip if no paths found
                if not paths:
                    continue
                
                # Compute scores for each path
                for path in paths:
                    # 1. Semantic similarity of nodes along the path
                    semantic_scores = []
                    for node in path:
                        node_data = graph.nodes[node]
                        embedding = node_data.get("embedding")
                        if embedding is not None:
                            sim = self.compute_semantic_similarity(query_embedding, embedding)
                            semantic_scores.append(sim)
                    
                    avg_semantic = sum(semantic_scores) / len(semantic_scores) if semantic_scores else 0.0
                    
                    # 2. Path length score (shorter is better)
                    path_length_score = self.normalize_path_length_score(len(path))
                    
                    # 3. Edge strength along the path
                    edge_strength = self.compute_edge_strength(path, graph)
                    
                    # Combine scores using weights
                    combined_score = (
                        self.semantic_weight * avg_semantic +
                        self.path_length_weight * path_length_score +
                        self.edge_strength_weight * edge_strength
                    )
                    
                    # Create path info
                    path_info = {
                        "path": path,
                        "score": combined_score,
                        "semantic_score": avg_semantic,
                        "path_length_score": path_length_score,
                        "edge_strength": edge_strength,
                        "source": source_node,
                        "target": target,
                        "length": len(path)
                    }
                    
                    ranked_paths.append(path_info)
            
            except nx.NetworkXNoPath:
                # No path exists between source and target
                continue
            except Exception as e:
                logger.error(f"Error finding paths from {source_node} to {target}: {e}")
                continue
        
        # Sort by score in descending order
        ranked_paths.sort(key=lambda x: x["score"], reverse=True)
        
        # Limit to max_paths
        return ranked_paths[:self.max_paths]
    
    def get_node_details(
        self,
        paths: List[Dict[str, Any]],
        graph: nx.DiGraph
    ) -> List[Dict[str, Any]]:
        """
        Add node details to ranked paths.
        
        Args:
            paths: List of ranked paths
            graph: NetworkX DiGraph
            
        Returns:
            Paths with node details
        """
        enriched_paths = []
        
        for path_info in paths:
            path = path_info["path"]
            enriched_nodes = []
            
            for node_id in path:
                node_data = graph.nodes[node_id]
                enriched_node = {
                    "id": node_id,
                    "title": node_data.get("title"),
                    "document_type": node_data.get("document_type"),
                    "content_preview": node_data.get("content", "")[:200] + "..." if len(node_data.get("content", "")) > 200 else node_data.get("content", ""),
                    "source": node_data.get("source")
                }
                enriched_nodes.append(enriched_node)
            
            # Create enriched path
            enriched_path = dict(path_info)
            enriched_path["nodes"] = enriched_nodes
            
            enriched_paths.append(enriched_path)
        
        return enriched_paths
    
    def find_and_rank_paths(
        self,
        query_embedding: List[float],
        documents: List[IngestDocument],
        relations: List[DocumentRelation],
        source_id: Optional[str] = None,
        target_ids: Optional[List[str]] = None,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Find and rank paths between documents.
        
        Args:
            query_embedding: Query embedding for semantic similarity
            documents: List of documents
            relations: List of relations between documents
            source_id: Optional source document ID (if None, use most similar to query)
            target_ids: Optional list of target document IDs (if None, use all documents)
            top_k: Number of top paths to return
            
        Returns:
            List of ranked paths with scores and node details
        """
        # Build graph
        graph = self.build_graph(documents, relations)
        
        # Determine source node if not provided
        if source_id is None:
            # Find most semantically similar document
            max_sim = -1.0
            best_doc = None
            
            for doc in documents:
                if doc.embedding is not None:
                    sim = self.compute_semantic_similarity(query_embedding, doc.embedding)
                    if sim > max_sim:
                        max_sim = sim
                        best_doc = doc
            
            if best_doc is None:
                logger.warning("No document with embedding found for similarity comparison")
                return []
            
            source_id = best_doc.id
        
        # Determine target nodes if not provided
        if target_ids is None:
            # Use all nodes as potential targets
            target_ids = [doc.id for doc in documents if doc.id != source_id]
        
        # Rank paths
        ranked_paths = self.rank_paths(query_embedding, source_id, target_ids, graph)
        
        # Add node details
        enriched_paths = self.get_node_details(ranked_paths, graph)
        
        # Return top-k paths
        return enriched_paths[:top_k]

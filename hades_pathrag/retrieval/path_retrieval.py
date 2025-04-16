"""
Path-based retrieval for HADES-PathRAG.

This module implements the core path-based retrieval features of the PathRAG
architecture, providing functions for combining semantic and structural relevance.
"""
from typing import Dict, List, Optional, Tuple, Any, Set, Union, cast
import logging
import numpy as np
from dataclasses import dataclass, field

from hades_pathrag.typings import (
    NodeIDType, NodeData, EmbeddingArray
)
from hades_pathrag.storage.path_traversal import PathQuery, PathResult, execute_path_query
from hades_pathrag.storage.edge_types import EdgeCategory, EDGE_TYPES

logger = logging.getLogger(__name__)


@dataclass
class PathRankingConfig:
    """Configuration for path ranking algorithms."""
    # Weights for scoring components
    semantic_weight: float = 0.7  # Weight for semantic relevance
    structural_weight: float = 0.3  # Weight for structural relevance
    
    # Structural scoring components
    path_length_weight: float = 0.1  # Weight for path length component
    edge_strength_weight: float = 0.2  # Weight for edge strength component
    
    # Decay factor for distance from query node (α^d where d = distance)
    distance_decay_factor: float = 0.85
    
    # Constraints
    max_path_length: int = 3
    min_edge_weight: float = 0.2
    
    # Edge type filters
    allowed_edge_types: Optional[List[str]] = None
    excluded_edge_types: List[str] = field(default_factory=list)
    
    # Result limits
    max_expansion_paths: int = 20
    max_results: int = 10


@dataclass
class RetrievalResult:
    """Result of a path-based retrieval query."""
    node_id: NodeIDType
    data: NodeData
    score: float
    path_from_query: Optional[List[Dict[str, Any]]] = None
    semantic_score: float = 0.0
    structural_score: float = 0.0


def calculate_semantic_relevance(
    query_embedding: EmbeddingArray,
    node_embedding: EmbeddingArray
) -> float:
    """
    Calculate semantic relevance between query and node embeddings.
    
    Args:
        query_embedding: Embedding array for the query
        node_embedding: Embedding array for the node
        
    Returns:
        Similarity score (0.0-1.0)
    """
    # Convert embeddings to numpy arrays (ensures all code paths are reachable)
    query_embedding = np.asarray(query_embedding)
    node_embedding = np.asarray(node_embedding)

    # Normalize embeddings
    query_norm = np.linalg.norm(query_embedding)
    node_norm = np.linalg.norm(node_embedding)
    
    if query_norm == 0 or node_norm == 0:
        return 0.0
    
    query_normalized = query_embedding / query_norm
    node_normalized = node_embedding / node_norm
    
    # Calculate cosine similarity
    similarity = np.dot(query_normalized, node_normalized)
    
    # Ensure value is in valid range
    return float(max(0.0, min(1.0, similarity)))


def calculate_structural_relevance(
    path_result: PathResult,
    config: PathRankingConfig
) -> float:
    """
    Calculate structural relevance score for a path.
    
    Args:
        path_result: Result of path traversal
        config: Path ranking configuration
        
    Returns:
        Structural relevance score (0.0-1.0)
    """
    # Calculate path length component (shorter is better)
    # Normalize to 0.1-1.0 range (1.0 for length=1, 0.1 for length=10+)
    path_length = path_result.length
    length_score = max(0.1, min(1.0, 1.0 / path_length))
    
    # Calculate edge strength component (average edge weight)
    edge_strength = path_result.avg_weight
    
    # Apply weights from config
    length_component = length_score * config.path_length_weight
    edge_component = edge_strength * config.edge_strength_weight
    
    # Combine components into structural score
    structural_score = length_component + edge_component
    
    # Normalize to 0.0-1.0 range
    normalized_score = structural_score / (config.path_length_weight + config.edge_strength_weight)
    
    return float(normalized_score)


def apply_distance_decay(
    score: float,
    distance: int,
    decay_factor: float
) -> float:
    """
    Apply distance decay to a score based on distance from query node.
    
    Args:
        score: Original score
        distance: Distance from query node
        decay_factor: Decay factor (α in α^d formula)
        
    Returns:
        Decayed score
    """
    # Apply exponential decay: score * (decay_factor ^ distance)
    if distance <= 0:
        return score
    
    decayed_score = score * (decay_factor ** distance)
    return decayed_score


def combine_scores(
    semantic_score: float,
    structural_score: float,
    config: PathRankingConfig
) -> float:
    """
    Combine semantic and structural scores into a final ranking score.
    
    Args:
        semantic_score: Semantic relevance score (0.0-1.0)
        structural_score: Structural relevance score (0.0-1.0)
        config: Path ranking configuration
        
    Returns:
        Combined score (0.0-1.0)
    """
    combined = (
        (semantic_score * config.semantic_weight) + 
        (structural_score * config.structural_weight)
    )
    
    # Ensure value is in valid range
    return max(0.0, min(1.0, combined))


def path_based_retrieval(
    db: Any,  # Database connection
    query_embedding: EmbeddingArray,
    initial_nodes: List[Tuple[NodeIDType, NodeData, float]],
    config: Optional[PathRankingConfig] = None,
) -> List[RetrievalResult]:
    """
    Perform path-based retrieval using both semantic and structural relevance.
    
    This is the core retrieval algorithm of the PathRAG architecture, combining
    semantic search results with graph path traversal for comprehensive retrieval.
    
    Args:
        db: Database connection
        query_embedding: Embedding of the query
        initial_nodes: Initial nodes from semantic search (id, data, score)
        config: Optional configuration for path ranking
        
    Returns:
        List of retrieval results with combined scores
    """
    # Use default config if none provided
    if config is None:
        config = PathRankingConfig()
    
    # Extract node IDs and scores from initial results
    initial_node_ids = [node_id for node_id, _, _ in initial_nodes]
    initial_nodes_map = {node_id: (data, score) for node_id, data, score in initial_nodes}
    
    # Create a set to track processed nodes
    processed_nodes: Set[NodeIDType] = set(initial_node_ids)
    
    # Initialize results with initial nodes
    results: List[RetrievalResult] = []
    for node_id, data, semantic_score in initial_nodes:
        results.append(RetrievalResult(
            node_id=node_id,
            data=data,
            score=semantic_score,  # Will be updated later
            semantic_score=semantic_score,
            structural_score=1.0,  # Initial nodes have perfect structural score
        ))
    
    # Build path traversal query
    path_query = PathQuery(
        start_vertex="",  # Will be set for each initial node
        direction="OUTBOUND",
        min_depth=1,
        max_depth=config.max_path_length,
        edge_collections=["edges"],  # Default edge collection
        edge_types=config.allowed_edge_types,
        min_weight=config.min_edge_weight,
        sort_by="combined",
        limit=config.max_expansion_paths,
    )
    
    # Expand paths from each initial node
    expanded_paths: List[Tuple[NodeIDType, PathResult]] = []
    for node_id in initial_node_ids:
        # Update starting node for query
        path_query.start_vertex = node_id
        
        # Execute path query
        paths = execute_path_query(db, path_query)
        for path in paths:
            # Get last node in path
            if not path.path_vertices:
                continue
            
            target_node_id = path.path_vertices[-1]["_id"]
            
            # Skip if already processed
            if target_node_id in processed_nodes:
                continue
            
            expanded_paths.append((target_node_id, path))
            processed_nodes.add(target_node_id)
    
    # Process expanded paths
    for target_node_id, path in expanded_paths:
        # Extract node data (it's in the last vertex of the path)
        target_data = path.path_vertices[-1]
        
        # Calculate semantic score if embedding is available
        semantic_score = 0.0
        if "embedding" in target_data:
            target_embedding = target_data["embedding"]
            semantic_score = calculate_semantic_relevance(
                query_embedding, target_embedding
            )
        
        # Calculate structural score
        structural_score = calculate_structural_relevance(path, config)
        
        # Apply distance decay to semantic score based on path length
        decayed_semantic = apply_distance_decay(
            semantic_score, 
            path.length, 
            config.distance_decay_factor
        )
        
        # Combine scores
        combined_score = combine_scores(
            decayed_semantic, 
            structural_score, 
            config
        )
        
        # Create result
        results.append(RetrievalResult(
            node_id=target_node_id,
            data=target_data,
            score=combined_score,
            path_from_query=[v for v in path.path_vertices],
            semantic_score=semantic_score,
            structural_score=structural_score,
        ))
    
    # Sort by combined score
    results.sort(key=lambda x: x.score, reverse=True)
    
    # Return top results
    return results[:config.max_results]


def retrieve_related_nodes(
    db: Any,  # Database connection
    node_id: NodeIDType,
    max_depth: int = 2,
    min_weight: float = 0.3,
    edge_types: Optional[List[str]] = None,
    limit: int = 10
) -> List[Dict[str, Any]]:
    """
    Retrieve nodes related to a specific node through the graph.
    
    This is a simpler alternative to full path-based retrieval when you
    just want to find nodes connected to a specific node.
    
    Args:
        db: Database connection
        node_id: Starting node ID
        max_depth: Maximum traversal depth
        min_weight: Minimum edge weight
        edge_types: Optional list of edge types to filter by
        limit: Maximum number of results
        
    Returns:
        List of related nodes with their relationship data
    """
    # Build path query
    path_query = PathQuery(
        start_vertex=node_id,
        direction="ANY",  # Both incoming and outgoing connections
        min_depth=1,
        max_depth=max_depth,
        edge_collections=["edges"],
        edge_types=edge_types,
        min_weight=min_weight,
        sort_by="weight",  # Sort by edge weight
        limit=limit,
    )
    
    # Execute query
    paths = execute_path_query(db, path_query)
    
    # Extract related nodes
    related_nodes = []
    for path in paths:
        # Skip empty paths
        if not path.path_vertices or len(path.path_vertices) <= 1:
            continue
        
        # Determine which node is the target (last one that isn't the start node)
        target_vertex = path.path_vertices[-1]
        if target_vertex["_id"] == node_id and len(path.path_vertices) > 1:
            target_vertex = path.path_vertices[-2]
        
        # Get edge connecting to this node
        connecting_edge = path.path_edges[-1] if path.path_edges else None
        
        related_nodes.append({
            "node": target_vertex,
            "edge": connecting_edge,
            "path_weight": path.avg_weight,
            "path_length": path.length,
        })
    
    return related_nodes

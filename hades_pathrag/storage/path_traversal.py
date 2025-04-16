"""
Path traversal utilities for ArangoDB.

This module provides functions for efficient path traversal and retrieval in ArangoDB,
supporting typed edges and XnX path notation for the PathRAG architecture.
"""
from typing import Dict, List, Optional, Tuple, Any, Set, Union, cast, MutableMapping, Iterable
import logging
import time
from dataclasses import dataclass, field

import numpy as np
from arango.database import Database
from arango.graph import Graph as ArangoGraph
from arango.collection import Collection
from arango.exceptions import AQLQueryExecuteError
from arango.cursor import Cursor
from arango.job import AsyncJob, BatchJob

# Define a type alias to make type annotations clearer
ArangoBindVars = Optional[MutableMapping[str, Any]]

from hades_pathrag.typings import (
    NodeIDType, NodeData, EdgeData, PathType, Graph, DiGraph
)

from .edge_types import EDGE_TYPES, EdgeCategory, get_edge_weight

logger = logging.getLogger(__name__)


@dataclass
class PathQuery:
    """
    Configuration for path traversal queries.
    """
    start_vertex: NodeIDType
    direction: str = "OUTBOUND"  # OUTBOUND, INBOUND, ANY
    min_depth: int = 1
    max_depth: int = 3
    edge_collections: List[str] = field(default_factory=lambda: ["edges"])
    edge_types: Optional[List[str]] = None  # Filter by specific edge types
    min_weight: Optional[float] = None  # Minimum edge weight
    sort_by: str = "weight"  # Sort paths by: weight, length, or combined
    limit: int = 10  # Max number of paths to return
    with_vertices: bool = True  # Include vertices in results
    with_edges: bool = True  # Include edges in results


@dataclass
class PathResult:
    """
    Result of a path traversal query.
    """
    path_vertices: List[Dict[str, Any]]  # List of vertices in the path
    path_edges: List[Dict[str, Any]]  # List of edges in the path
    total_weight: float  # Sum of edge weights
    avg_weight: float  # Average edge weight
    length: int  # Path length (number of edges)
    score: float  # Composite path score


def build_xnx_pattern(
    direction: str = "OUTBOUND",
    min_depth: int = 1,
    max_depth: int = 3,
    edge_types: Optional[List[str]] = None
) -> str:
    """
    Build an XnX (traversal) pattern for ArangoDB path queries.
    
    Args:
        direction: Direction of traversal (OUTBOUND, INBOUND, ANY)
        min_depth: Minimum traversal depth
        max_depth: Maximum traversal depth
        edge_types: List of edge types to filter by
        
    Returns:
        XnX pattern string for use in AQL
    """
    # Base XnX pattern
    base_pattern = f"{direction}"
    
    # Add edge type filter if specified
    if edge_types and len(edge_types) > 0:
        edge_filter = ", ".join([f"'{edge_type}'" for edge_type in edge_types])
        base_pattern += f"[relation_type IN [{edge_filter}]]"
    
    # Add depth specifier
    if min_depth == max_depth:
        depth_pattern = f"{min_depth}"
    else:
        depth_pattern = f"{min_depth}..{max_depth}"
    
    return f"{base_pattern} {depth_pattern}"


def generate_path_query(
    db: Database,
    query: PathQuery
) -> str:
    """
    Generate an AQL query for path traversal based on the query parameters.
    
    Args:
        db: ArangoDB database instance
        query: Path query configuration
        
    Returns:
        AQL query string
    """
    # Build XnX pattern
    xnx_pattern = build_xnx_pattern(
        direction=query.direction,
        min_depth=query.min_depth,
        max_depth=query.max_depth,
        edge_types=query.edge_types
    )
    
    # Build collections string
    collections_str = ", ".join([f"@@{col}" for col in query.edge_collections])
    collections_bind = {f"@{col}": col for col in query.edge_collections}
    
    # Weight filter
    weight_filter = ""
    if query.min_weight is not None:
        weight_filter = f"FILTER e.weight >= {query.min_weight}"
    
    # Build the core AQL query
    aql = f"""
    FOR v, e, p IN {xnx_pattern}
        @start_vertex
        {collections_str}
        {weight_filter}
        RETURN {{
            "vertices": p.vertices,
            "edges": p.edges,
            "weights": (
                FOR edge IN p.edges
                    RETURN edge.weight
            )
        }}
    """
    
    # Add sorting and limiting
    if query.sort_by == "weight":
        aql += """
        SORT SUM(RETURN_VALUE.weights) DESC
        """
    elif query.sort_by == "length":
        aql += """
        SORT LENGTH(RETURN_VALUE.edges) ASC
        """
    elif query.sort_by == "combined":
        # Combined score: higher weights and shorter paths are better
        aql += """
        SORT SUM(RETURN_VALUE.weights) / LENGTH(RETURN_VALUE.edges) DESC
        """
    
    # Add limit
    aql += f"""
    LIMIT {query.limit}
    """
    
    return aql


def execute_path_query(
    db: Database,
    query: PathQuery
) -> List[PathResult]:
    """
    Execute a path traversal query and return the results.
    
    Args:
        db: ArangoDB database instance
        query: Path query configuration
        
    Returns:
        List of path results
    """
    # Generate the AQL query
    aql = generate_path_query(db, query)
    
    # Prepare bind variables
    bind_vars = {
        "start_vertex": query.start_vertex
    }
    
    # Add collection bindings
    for col in query.edge_collections:
        bind_vars[f"@{col}"] = col
    
    # Execute the query
    try:
        # Use explicitly typed ArangoBindVars for compatibility with ArangoDB AQL
        cursor = db.aql.execute(aql, bind_vars=cast(ArangoBindVars, bind_vars))
        # Explicitly cast cursor to ensure it's iterable
        paths = list(cast(Iterable[Any], cursor))
        
        # Convert to PathResult objects
        results = []
        for path in paths:
            vertices = path["vertices"] if query.with_vertices else []
            edges = path["edges"] if query.with_edges else []
            weights = path["weights"]
            
            # Calculate metrics
            total_weight = sum(weights) if weights else 0.0
            avg_weight = total_weight / len(weights) if weights else 0.0
            length = len(edges)
            
            # Calculate composite score (70% weight, 30% length)
            # Higher is better for weight, lower is better for length
            # Normalize length to 0-1 scale (1.0 for length=1, 0.1 for length=10)
            length_score = max(0.1, 1.0 / length)
            score = (0.7 * avg_weight) + (0.3 * length_score)
            
            results.append(PathResult(
                path_vertices=vertices,
                path_edges=edges,
                total_weight=total_weight,
                avg_weight=avg_weight,
                length=length,
                score=score
            ))
        
        return results
    
    except AQLQueryExecuteError as e:
        logger.error(f"Error executing path query: {e}")
        return []


def find_paths_between(
    db: Database,
    start_vertex: NodeIDType,
    end_vertex: NodeIDType,
    edge_collections: List[str] = ["edges"],
    max_depth: int = 3,
    edge_types: Optional[List[str]] = None,
    min_weight: Optional[float] = None,
    limit: int = 5
) -> List[PathResult]:
    """
    Find paths between two vertices with a focus on high-weight paths.
    
    Args:
        db: ArangoDB database instance
        start_vertex: Starting vertex ID
        end_vertex: Ending vertex ID
        edge_collections: List of edge collection names
        max_depth: Maximum traversal depth
        edge_types: Optional list of edge types to filter by
        min_weight: Minimum edge weight to consider
        limit: Maximum number of paths to return
        
    Returns:
        List of paths between the vertices
    """
    # This requires a different query than the general path traversal
    # We need to find paths specifically to the end vertex
    
    # Build XnX pattern
    xnx_pattern = build_xnx_pattern(
        direction="OUTBOUND",  # Typically we use outbound for path finding
        min_depth=1,
        max_depth=max_depth,
        edge_types=edge_types
    )
    
    # Build collections string
    collections_str = ", ".join([f"@@{col}" for col in edge_collections])
    
    # Weight filter
    weight_filter = ""
    if min_weight is not None:
        weight_filter = f"FILTER e.weight >= {min_weight}"
    
    # Build the AQL query to find paths to the end vertex
    aql = f"""
    FOR v, e, p IN {xnx_pattern}
        @start_vertex
        {collections_str}
        {weight_filter}
        FILTER v._id == @end_vertex
        RETURN {{
            "vertices": p.vertices,
            "edges": p.edges,
            "weights": (
                FOR edge IN p.edges
                    RETURN edge.weight
            )
        }}
    """
    
    # Sort by combined score (higher weights and shorter paths are better)
    aql += """
    SORT SUM(RETURN_VALUE.weights) / LENGTH(RETURN_VALUE.edges) DESC
    """
    
    # Add limit
    aql += f"""
    LIMIT {limit}
    """
    
    # Prepare bind variables
    bind_vars = {
        "start_vertex": start_vertex,
        "end_vertex": end_vertex
    }
    
    # Add collection bindings
    for col in edge_collections:
        bind_vars[f"@{col}"] = col
    
    # Execute the query
    try:
        # Use explicitly typed ArangoBindVars for compatibility with ArangoDB AQL
        cursor = db.aql.execute(aql, bind_vars=cast(ArangoBindVars, bind_vars))
        # Explicitly cast cursor to ensure it's iterable
        paths = list(cast(Iterable[Any], cursor))
        
        # Convert to PathResult objects
        results = []
        for path in paths:
            vertices = path["vertices"]
            edges = path["edges"]
            weights = path["weights"]
            
            # Calculate metrics
            total_weight = sum(weights) if weights else 0.0
            avg_weight = total_weight / len(weights) if weights else 0.0
            length = len(edges)
            
            # Calculate composite score (70% weight, 30% length)
            length_score = max(0.1, 1.0 / length)
            score = (0.7 * avg_weight) + (0.3 * length_score)
            
            results.append(PathResult(
                path_vertices=vertices,
                path_edges=edges,
                total_weight=total_weight,
                avg_weight=avg_weight,
                length=length,
                score=score
            ))
        
        return results
    
    except AQLQueryExecuteError as e:
        logger.error(f"Error executing path query: {e}")
        return []


def expand_paths_from_nodes(
    db: Database,
    start_nodes: List[NodeIDType],
    edge_collections: List[str] = ["edges"],
    max_depth: int = 2,
    edge_types: Optional[List[str]] = None,
    min_weight: Optional[float] = None,
    limit_per_node: int = 5,
    total_limit: int = 20
) -> List[PathResult]:
    """
    Expand paths from multiple starting nodes and return the highest-scoring paths.
    
    Args:
        db: ArangoDB database instance
        start_nodes: List of starting node IDs
        edge_collections: List of edge collection names
        max_depth: Maximum traversal depth
        edge_types: Optional list of edge types to filter by
        min_weight: Minimum edge weight to consider
        limit_per_node: Maximum number of paths per starting node
        total_limit: Overall maximum number of paths to return
        
    Returns:
        List of expanded paths
    """
    all_paths = []
    
    # Query paths from each starting node
    for start_node in start_nodes:
        query = PathQuery(
            start_vertex=start_node,
            direction="OUTBOUND",
            min_depth=1,
            max_depth=max_depth,
            edge_collections=edge_collections,
            edge_types=edge_types,
            min_weight=min_weight,
            sort_by="combined",
            limit=limit_per_node
        )
        
        paths = execute_path_query(db, query)
        all_paths.extend(paths)
    
    # Sort all paths by score and return the top results
    all_paths.sort(key=lambda p: p.score, reverse=True)
    return all_paths[:total_limit]

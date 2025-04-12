"""
Path pruning algorithms for PathRAG.

This module contains the flow-based path pruning algorithm implementations
used by PathRAG for extracting and scoring relevant paths in knowledge graphs.
"""
from typing import Dict, List, Set, Tuple, Any, Optional, Iterator, Callable, Mapping, cast
from dataclasses import dataclass, field
import logging
import heapq
import math
from collections import defaultdict

import networkx as nx
import numpy as np

from ..graph.base import Path, NodeID, EdgeID

logger = logging.getLogger(__name__)


@dataclass
class PathPruningConfig:
    """Configuration for path pruning algorithms."""
    
    # Maximum path length
    max_path_length: int = 4
    
    # Resource decay rate for each hop in the path
    decay_rate: float = 0.5
    
    # Minimum reliability threshold for path pruning
    pruning_threshold: float = 0.01
    
    # Maximum number of paths between any two nodes
    max_paths_per_node_pair: int = 5
    
    # Minimum initial resource for source nodes
    min_initial_resource: float = 0.1
    
    # Max initial resource for source nodes
    max_initial_resource: float = 1.0
    
    # Whether to consider edge weights in resource propagation
    use_edge_weights: bool = True
    
    # Whether to normalize resources at each step
    normalize_resources: bool = True
    
    # Resource allocation method: 'proportional' or 'equal'
    resource_allocation: str = 'proportional'
    
    # Whether to use diversity-based path selection
    use_diversity: bool = True
    
    # Diversity weight in path selection (0-1)
    diversity_weight: float = 0.3


@dataclass
class PathResource:
    """Resource allocation for a path."""
    
    # Path nodes
    nodes: List[NodeID]
    
    # Path edges (with attributes)
    edges: List[Dict[str, Any]]
    
    # Reliability score (0-1)
    reliability: float
    
    # Resource value at each node
    node_resources: Dict[NodeID, float] = field(default_factory=dict)
    
    # Parent path (for tracking origin)
    parent_path: Optional['PathResource'] = None
    
    # Diversity score (used for diverse path selection)
    diversity: float = 0.0
    
    # Combined score (reliability + diversity)
    combined_score: float = 0.0
    
    def __lt__(self, other: 'PathResource') -> bool:
        """Comparison for priority queue."""
        return self.reliability > other.reliability
    
    def add_node(self, node_id: NodeID, resource: float, edge: Dict[str, Any]) -> 'PathResource':
        """Add a node to this path with the given resource level."""
        new_nodes = self.nodes.copy()
        new_nodes.append(node_id)
        
        new_edges = self.edges.copy()
        new_edges.append(edge)
        
        new_resources = self.node_resources.copy()
        new_resources[node_id] = resource
        
        return PathResource(
            nodes=new_nodes,
            edges=new_edges,
            reliability=resource,  # Reliability of the path is the resource of the last node
            node_resources=new_resources,
            parent_path=self
        )
    
    def to_path(self) -> Path:
        """Convert to Path object."""
        # Extract edge IDs from edge dictionaries
        edge_ids: List[EdgeID] = []
        edge_weights: Dict[Tuple[NodeID, NodeID], float] = {}
        
        for i, edge_dict in enumerate(self.edges):
            # Extract or generate edge ID - use 'id' field if present, otherwise generate one
            edge_id = edge_dict.get('id', f"edge_{i}")
            # Must be a string according to the EdgeID type
            edge_id = str(edge_id)
            edge_ids.append(edge_id)
            
            # If we have source and target nodes, store the edge weight
            if i < len(self.nodes) - 1:
                source = self.nodes[i]
                target = self.nodes[i + 1]
                weight = edge_dict.get('weight', 1.0)
                edge_weights[(source, target)] = float(weight)
        
        return Path(
            nodes=self.nodes,
            edges=edge_ids,
            reliability=self.reliability,
            edge_weights=edge_weights
        )
    
    def nodes_set(self) -> Set[NodeID]:
        """Return set of nodes in this path."""
        return set(self.nodes)
    
    def overlaps_with(self, other: 'PathResource') -> float:
        """Compute the overlap ratio with another path."""
        my_nodes = self.nodes_set()
        other_nodes = other.nodes_set()
        
        if not my_nodes or not other_nodes:
            return 0.0
        
        # Overlap = intersection / union
        intersection = len(my_nodes.intersection(other_nodes))
        union = len(my_nodes.union(other_nodes))
        
        return intersection / union if union > 0 else 0.0


def calculate_propagation_weights(
    graph: nx.Graph,
    node_id: NodeID,
    allocation_method: str = 'proportional'
) -> Dict[NodeID, float]:
    """
    Calculate propagation weights for a node's neighbors.
    
    Args:
        graph: NetworkX graph
        node_id: Source node ID
        allocation_method: How to allocate resources ('proportional' or 'equal')
        
    Returns:
        Dictionary mapping neighbor node IDs to weight values
    """
    neighbors = list(graph.neighbors(node_id))
    weights: Dict[NodeID, float] = {}
    
    if not neighbors:
        return weights
    
    if allocation_method == 'equal':
        # Equal distribution to all neighbors
        weight_per_neighbor = 1.0 / len(neighbors)
        return {neighbor: weight_per_neighbor for neighbor in neighbors}
    
    # Default: proportional to edge weights
    total_weight = 0.0
    
    # First collect raw weights
    for neighbor in neighbors:
        edge_data = graph.get_edge_data(node_id, neighbor) or {}
        weight = edge_data.get('weight', 1.0)
        weights[neighbor] = weight
        total_weight += weight
    
    # Normalize weights
    if total_weight > 0:
        for neighbor in weights:
            weights[neighbor] /= total_weight
    else:
        # Fallback to equal weighting if all weights are zero
        weight_per_neighbor = 1.0 / len(neighbors)
        for neighbor in neighbors:
            weights[neighbor] = weight_per_neighbor
    
    return weights


def propagate_resources(
    graph: nx.Graph,
    source_nodes: Dict[NodeID, float],
    config: PathPruningConfig
) -> Dict[Tuple[NodeID, NodeID], List[PathResource]]:
    """
    Propagate resources through the graph and extract paths.
    
    Implementation of the flow-based path pruning algorithm described
    in the PathRAG paper.
    
    Args:
        graph: NetworkX graph
        source_nodes: Dictionary mapping source node IDs to initial resource levels
        config: Path pruning configuration
        
    Returns:
        Dictionary mapping (source, target) pairs to lists of paths
    """
    paths: Dict[Tuple[NodeID, NodeID], List[PathResource]] = defaultdict(list)
    visited_edges = set()
    
    # Queue of active paths (priority is reliability)
    active_paths: List[PathResource] = []
    
    # Initialize paths from source nodes
    for source_id, initial_resource in source_nodes.items():
        if source_id not in graph:
            logger.warning(f"Source node {source_id} not found in graph")
            continue
        
        # Check resource levels
        if initial_resource < config.min_initial_resource:
            initial_resource = config.min_initial_resource
        elif initial_resource > config.max_initial_resource:
            initial_resource = config.max_initial_resource
        
        # Create initial path
        path = PathResource(
            nodes=[source_id],
            edges=[],
            reliability=initial_resource,
            node_resources={source_id: initial_resource}
        )
        
        # Add to active paths
        active_paths.append(path)
    
    # Process paths by iterative resource propagation
    while active_paths:
        # Get path with highest reliability
        current_path = heapq.heappop(active_paths)
        
        # Stop propagation if path reliability is below threshold
        if current_path.reliability < config.pruning_threshold:
            continue
        
        # Stop propagation if path is too long
        if len(current_path.nodes) > config.max_path_length:
            continue
        
        # Get current node (last in path)
        current_node = current_path.nodes[-1]
        
        # Get neighbors of current node
        neighbors = list(graph.neighbors(current_node))
        
        # Skip if no neighbors
        if not neighbors:
            continue
        
        # Calculate propagation weights
        prop_weights = calculate_propagation_weights(
            graph, current_node, config.resource_allocation
        )
        
        # Propagate resources to neighbors
        current_resource = current_path.reliability
        
        for neighbor in neighbors:
            # Skip if neighbor is already in path (no cycles)
            if neighbor in current_path.nodes:
                continue
            
            # Get edge data
            edge_data = graph.get_edge_data(current_node, neighbor) or {}
            edge_id = edge_data.get('id', f"{current_node}-{neighbor}")
            
            # Check if this exact edge has been visited in this propagation
            edge_key = (current_node, neighbor, edge_id)
            if edge_key in visited_edges:
                continue
            
            visited_edges.add(edge_key)
            
            # Calculate resource propagated to this neighbor
            weight = prop_weights.get(neighbor, 0.0)
            propagated_resource = current_resource * weight * config.decay_rate
            
            # Skip if propagated resource is below threshold
            if propagated_resource < config.pruning_threshold:
                continue
            
            # Create new path with this neighbor
            # Cast edge_data to Dict[str, Any] to ensure type compatibility
            dict_edge_data = cast(Dict[str, Any], edge_data)
            new_path = current_path.add_node(
                neighbor, propagated_resource, dict_edge_data
            )
            
            # Store path from source to this node
            source_id = current_path.nodes[0]
            path_key = (source_id, neighbor)
            
            # Check if we've reached max paths for this node pair
            current_paths = paths[path_key]
            if len(current_paths) >= config.max_paths_per_node_pair:
                # Only keep if better than worst existing path
                if min(p.reliability for p in current_paths) < propagated_resource:
                    # Remove worst path
                    worst_path = min(current_paths, key=lambda p: p.reliability)
                    current_paths.remove(worst_path)
                    current_paths.append(new_path)
            else:
                current_paths.append(new_path)
            
            # Continue propagation from this neighbor
            heapq.heappush(active_paths, new_path)
    
    return paths


def calculate_path_diversity(paths: List[PathResource]) -> List[PathResource]:
    """
    Calculate diversity scores for paths and update combined scores.
    
    Args:
        paths: List of paths to process
        
    Returns:
        Updated paths with diversity and combined scores
    """
    if not paths:
        return paths
    
    # First, sort by reliability
    sorted_paths = sorted(paths, key=lambda p: p.reliability, reverse=True)
    
    # Set highest diversity for first path
    sorted_paths[0].diversity = 1.0
    sorted_paths[0].combined_score = sorted_paths[0].reliability
    
    # Process remaining paths
    for i in range(1, len(sorted_paths)):
        current_path = sorted_paths[i]
        
        # Calculate maximum overlap with any higher-ranked path
        max_overlap = max(
            current_path.overlaps_with(sorted_paths[j])
            for j in range(i)
        )
        
        # Diversity is inverse of overlap
        current_path.diversity = 1.0 - max_overlap
        
        # Combined score: reliability + diversity bonus
        current_path.combined_score = (
            current_path.reliability * (1.0 - 0.3) +
            current_path.diversity * 0.3
        )
    
    # Return paths sorted by combined score
    return sorted(sorted_paths, key=lambda p: p.combined_score, reverse=True)


def extract_paths_with_pruning(
    graph: nx.Graph,
    source_nodes: Dict[NodeID, float],
    config: Optional[PathPruningConfig] = None
) -> List[Path]:
    """
    Extract paths from the graph using flow-based pruning.
    
    Args:
        graph: NetworkX graph
        source_nodes: Dictionary mapping source node IDs to initial resource levels
        config: Optional path pruning configuration
        
    Returns:
        List of extracted paths sorted by reliability
    """
    # Use default config if not provided
    pruning_config = config or PathPruningConfig()
    
    # Propagate resources through the graph
    path_dict = propagate_resources(graph, source_nodes, pruning_config)
    
    # Collect all paths
    all_paths: List[PathResource] = []
    for paths in path_dict.values():
        all_paths.extend(paths)
    
    # Apply diversity-based selection if enabled
    if pruning_config.use_diversity and all_paths:
        all_paths = calculate_path_diversity(all_paths)
    else:
        # Otherwise, sort by reliability
        all_paths = sorted(all_paths, key=lambda p: p.reliability, reverse=True)
    
    # Convert to Path objects
    return [path.to_path() for path in all_paths]

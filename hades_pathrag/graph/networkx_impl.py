"""
NetworkX implementation of the graph interface.

This module provides a NetworkX-based implementation of the BaseGraph
interface for testing and development purposes.
"""
from typing import Dict, List, Optional, Set, Tuple, Any, Type, cast, Union
from collections import defaultdict, deque
import logging

import networkx as nx
import numpy as np

from .base import BaseGraph, Path, NodeID, EdgeID

logger = logging.getLogger(__name__)


class NetworkXGraph(BaseGraph):
    """NetworkX implementation of the graph interface."""
    
    def __init__(self, directed: bool = True) -> None:
        """
        Initialize an empty NetworkX graph.
        
        Args:
            directed: Whether to use a directed graph
        """
        self.directed = directed
        self.graph: Union[nx.DiGraph, nx.Graph] = nx.DiGraph() if directed else nx.Graph()
        logger.info(f"Initialized NetworkX graph (directed={directed})")
    
    def add_node(self, node_id: NodeID, attributes: Dict[str, Any]) -> None:
        """
        Add a node to the graph.
        
        Args:
            node_id: Unique identifier for the node
            attributes: Node attributes including text content
        """
        self.graph.add_node(node_id, **attributes)
    
    def add_edge(
        self, 
        source_id: NodeID, 
        target_id: NodeID, 
        relation_type: str,
        weight: float = 1.0,
        attributes: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add an edge between two nodes.
        
        Args:
            source_id: Source node ID
            target_id: Target node ID
            relation_type: Type of relationship
            weight: Edge weight
            attributes: Optional edge attributes
        """
        edge_attrs = {"relation_type": relation_type, "weight": weight}
        if attributes:
            edge_attrs.update(attributes)
        
        self.graph.add_edge(source_id, target_id, **edge_attrs)
    
    def get_node(self, node_id: NodeID) -> Optional[Dict[str, Any]]:
        """
        Get node by ID.
        
        Args:
            node_id: ID of the node to retrieve
            
        Returns:
            Node attributes if found, None otherwise
        """
        if node_id in self.graph.nodes:
            return dict(self.graph.nodes[node_id])
        return None
    
    def get_neighbors(self, node_id: NodeID) -> List[NodeID]:
        """
        Get neighbors of a node.
        
        Args:
            node_id: ID of the node
            
        Returns:
            List of neighbor node IDs
        """
        if node_id in self.graph.nodes:
            if self.directed:
                # For directed graphs use successors
                digraph = cast(nx.DiGraph, self.graph)
                return list(digraph.successors(node_id))
            else:
                # For undirected graphs use neighbors
                return list(self.graph.neighbors(node_id))
        return []
    
    def get_paths(
        self,
        start_node: NodeID,
        end_node: NodeID,
        max_length: int = 4,
        decay_rate: float = 0.8,
        pruning_threshold: float = 0.01
    ) -> List[Path]:
        """
        Get paths between two nodes using flow-based pruning algorithm.
        
        Args:
            start_node: Starting node ID
            end_node: Target node ID
            max_length: Maximum path length
            decay_rate: Decay rate for information propagation (alpha)
            pruning_threshold: Threshold for early stopping (theta)
            
        Returns:
            List of paths from start_node to end_node
        """
        # Check if nodes exist
        if start_node not in self.graph.nodes or end_node not in self.graph.nodes:
            return []
        
        # Initialize resource distribution
        resources: Dict[NodeID, float] = defaultdict(float)
        resources[start_node] = 1.0
        visited: Set[NodeID] = {start_node}
        
        # Queue for BFS traversal: (node_id, path_so_far, edges_so_far, edge_weights)
        queue: deque = deque([(start_node, [start_node], [], {})])
        valid_paths: List[Path] = []
        
        while queue:
            current_node, current_path, current_edges, current_weights = queue.popleft()
            
            # Skip if path exceeds max length
            if len(current_path) > max_length + 1:
                continue
            
            # If reached end node, add path to valid paths
            if current_node == end_node and len(current_path) > 2:  # Path must have at least one intermediate node
                path_reliability = sum(current_weights.values()) / max(1, len(current_weights))
                valid_paths.append(Path(
                    nodes=current_path,
                    edges=current_edges,
                    reliability=path_reliability,
                    edge_weights=current_weights
                ))
                continue
            
            # Get outgoing edges
            if self.directed:
                # For directed graphs use out_edges
                digraph = cast(nx.DiGraph, self.graph)
                out_edges = list(digraph.out_edges(current_node, data=True))
            else:
                # For undirected graphs use edges
                out_edges = [(current_node, neighbor, data) for u, neighbor, data in 
                             self.graph.edges(current_node, data=True)]
            
            # Early stopping if resource is too small
            if resources[current_node] / max(1, len(out_edges)) < pruning_threshold:
                continue
            
            # Distribute resources to neighbors
            for src, dst, edge_data in out_edges:
                # Skip if node already in path (avoid cycles)
                if dst in current_path:
                    continue
                
                # Calculate resource flow to neighbor
                edge_weight = edge_data.get("weight", 1.0)
                neighbor_resource = decay_rate * resources[current_node] * edge_weight / max(1, len(out_edges))
                
                # If neighbor not visited or new resource is higher, update
                if dst not in visited or neighbor_resource > resources[dst]:
                    resources[dst] = neighbor_resource
                    visited.add(dst)
                    
                    # Add new path to queue
                    new_path = current_path + [dst]
                    relation_type = edge_data.get("relation_type", "related_to")
                    new_edges = current_edges + [relation_type]
                    
                    # Update edge weights
                    new_weights = dict(current_weights)
                    new_weights[(src, dst)] = neighbor_resource
                    
                    queue.append((dst, new_path, new_edges, new_weights))
        
        return valid_paths
    
    def extract_paths(
        self,
        retrieved_nodes: Set[NodeID],
        max_length: int = 4,
        decay_rate: float = 0.8,
        pruning_threshold: float = 0.01
    ) -> List[Path]:
        """
        Extract key relational paths between each pair of retrieved nodes.
        
        Args:
            retrieved_nodes: Set of retrieved node IDs
            max_length: Maximum path length
            decay_rate: Decay rate for information propagation (alpha)
            pruning_threshold: Threshold for early stopping (theta)
            
        Returns:
            List of paths between retrieved nodes
        """
        paths: List[Path] = []
        
        # Filter to existing nodes
        valid_nodes = [node for node in retrieved_nodes if node in self.graph.nodes]
        
        # Generate all possible node pairs from retrieved nodes
        for i, start_node in enumerate(valid_nodes):
            for j in range(i+1, len(valid_nodes)):
                end_node = valid_nodes[j]
                
                # For each pair, extract paths in both directions
                paths_i_to_j = self.get_paths(
                    start_node, 
                    end_node,
                    max_length=max_length,
                    decay_rate=decay_rate,
                    pruning_threshold=pruning_threshold
                )
                
                paths_j_to_i = self.get_paths(
                    end_node, 
                    start_node,
                    max_length=max_length,
                    decay_rate=decay_rate,
                    pruning_threshold=pruning_threshold
                )
                
                paths.extend(paths_i_to_j)
                paths.extend(paths_j_to_i)
        
        # Sort paths by reliability
        sorted_paths = sorted(paths, key=lambda p: p.reliability, reverse=True)
        
        return sorted_paths
    
    def to_networkx(self) -> nx.DiGraph:
        """
        Convert to NetworkX graph.
        
        Returns:
            NetworkX directed graph representation
        """
        # Ensure we return a DiGraph regardless of our internal representation
        if not self.directed:
            # If we're using an undirected graph internally, convert to DiGraph
            digraph: nx.DiGraph = nx.DiGraph()
            # Add all nodes with their attributes
            for node, attrs in self.graph.nodes(data=True):
                digraph.add_node(node, **attrs)
            # Add edges in both directions to maintain the undirected nature
            for u, v, attrs in self.graph.edges(data=True):
                digraph.add_edge(u, v, **attrs)
                digraph.add_edge(v, u, **attrs)  # Add reverse edge
            return digraph
        else:
            # If already a DiGraph, just return a copy
            return cast(nx.DiGraph, self.graph.copy())
    
    @classmethod
    def from_networkx(cls: Type['NetworkXGraph'], graph: nx.DiGraph) -> 'NetworkXGraph':
        """
        Create from NetworkX graph.
        
        Args:
            graph: NetworkX graph to convert
            
        Returns:
            Instance of this graph class
        """
        instance = cls(directed=isinstance(graph, nx.DiGraph))
        instance.graph = graph.copy()
        return instance

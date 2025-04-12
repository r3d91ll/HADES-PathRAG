"""
Base interfaces for graph operations in the PathRAG framework.

This module defines the abstract base classes for graph operations.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, TypeVar, Generic, Set, Tuple, Any, Type, cast

import networkx as nx
import numpy as np

# Type variables for graph classes
T = TypeVar('T', bound='BaseGraph')
NodeID = str
EdgeID = str
Weight = float


@dataclass
class Path:
    """
    Represents a path in the graph with reliability scoring.
    """
    nodes: List[NodeID]
    edges: List[EdgeID]
    reliability: float = 0.0
    edge_weights: Optional[Dict[Tuple[NodeID, NodeID], float]] = None
    
    def __post_init__(self) -> None:
        """Validate path structure after initialization."""
        if len(self.nodes) < 2:
            raise ValueError("Path must have at least 2 nodes")
        if len(self.edges) != len(self.nodes) - 1:
            raise ValueError("Number of edges must be one less than number of nodes")
        
        self.edge_weights = self.edge_weights or {}


class BaseGraph(ABC):
    """Base class for all graph implementations."""
    
    @abstractmethod
    def add_node(self, node_id: NodeID, attributes: Dict[str, Any]) -> None:
        """
        Add a node to the graph.
        
        Args:
            node_id: Unique identifier for the node
            attributes: Node attributes including text content
        """
        pass
    
    @abstractmethod
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
        pass
    
    @abstractmethod
    def get_node(self, node_id: NodeID) -> Optional[Dict[str, Any]]:
        """
        Get node by ID.
        
        Args:
            node_id: ID of the node to retrieve
            
        Returns:
            Node attributes if found, None otherwise
        """
        pass
    
    @abstractmethod
    def get_neighbors(self, node_id: NodeID) -> List[NodeID]:
        """
        Get neighbors of a node.
        
        Args:
            node_id: ID of the node
            
        Returns:
            List of neighbor node IDs
        """
        pass
    
    @abstractmethod
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
        pass
    
    @abstractmethod
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
        pass
    
    @abstractmethod
    def to_networkx(self) -> nx.DiGraph:
        """
        Convert to NetworkX graph.
        
        Returns:
            NetworkX directed graph representation
        """
        pass
    
    @classmethod
    @abstractmethod
    def from_networkx(cls: Type[T], graph: nx.DiGraph) -> T:
        """
        Create from NetworkX graph.
        
        Args:
            graph: NetworkX graph to convert
            
        Returns:
            Instance of this graph class
        """
        pass

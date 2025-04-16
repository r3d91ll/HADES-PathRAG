"""
Extended interfaces and data structures for graph operations in PathRAG.

This module provides additional interfaces and helper classes for 
graph operations in the PathRAG framework.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any, TypeVar, Generic, Protocol, Union

import networkx as nx  # type: ignore[attr-defined]
import numpy as np
from pydantic import BaseModel, Field

# Import common types from our centralized typing module
from hades_pathrag.typings import (
    NodeIDType, NodeData, EdgeData, PathType, Graph, DiGraph
)

from .base import BaseGraph, Path, EdgeID, Weight


class PathScoringConfig(BaseModel):
    """Configuration for path scoring algorithms."""
    
    decay_rate: float = Field(
        default=0.8,
        description="Decay rate for resource propagation (α)",
        ge=0.0,
        le=1.0
    )
    pruning_threshold: float = Field(
        default=0.01,
        description="Threshold for early stopping (θ)",
        ge=0.0
    )
    max_path_length: int = Field(
        default=4,
        description="Maximum path length",
        ge=2
    )
    reliability_scaling: float = Field(
        default=1.0,
        description="Scaling factor for reliability scores",
        gt=0.0
    )
    min_reliability: float = Field(
        default=0.0,
        description="Minimum reliability threshold for paths",
        ge=0.0
    )


class PathStats(BaseModel):
    """Statistics about paths in the graph."""
    
    total_paths: int = Field(
        default=0,
        description="Total number of paths extracted"
    )
    avg_path_length: float = Field(
        default=0.0,
        description="Average path length"
    )
    avg_reliability: float = Field(
        default=0.0,
        description="Average path reliability score"
    )
    reliability_distribution: Dict[str, int] = Field(
        default_factory=dict,
        description="Distribution of reliability scores"
    )
    length_distribution: Dict[int, int] = Field(
        default_factory=dict,
        description="Distribution of path lengths"
    )
    extraction_time_ms: float = Field(
        default=0.0,
        description="Time taken to extract paths in milliseconds"
    )


class GraphStats(BaseModel):
    """Statistics about the graph structure."""
    
    node_count: int = Field(
        default=0,
        description="Number of nodes in the graph"
    )
    edge_count: int = Field(
        default=0,
        description="Number of edges in the graph"
    )
    avg_degree: float = Field(
        default=0.0,
        description="Average node degree"
    )
    density: float = Field(
        default=0.0,
        description="Graph density"
    )
    connected_components: int = Field(
        default=0,
        description="Number of connected components"
    )
    avg_clustering: Optional[float] = Field(
        default=None,
        description="Average clustering coefficient"
    )
    diameter: Optional[int] = Field(
        default=None,
        description="Graph diameter"
    )


class PathFilterProtocol(Protocol):
    """Protocol for path filtering implementations."""
    
    def filter(self, paths: List[Path]) -> List[Path]:
        """
        Filter paths based on implementation-specific criteria.
        
        Args:
            paths: List of paths to filter
            
        Returns:
            Filtered list of paths
        """
        ...


@dataclass
class ReliabilityFilter:
    """Filter paths based on reliability threshold."""
    
    min_reliability: float = 0.1
    
    def filter(self, paths: List[Path]) -> List[Path]:
        """
        Filter paths based on minimum reliability threshold.
        
        Args:
            paths: List of paths to filter
            
        Returns:
            Filtered list of paths
        """
        return [p for p in paths if p.reliability >= self.min_reliability]


@dataclass
class DiversityFilter:
    """Filter paths to maximize diversity."""
    
    max_paths: int = 10
    diversity_weight: float = 0.5
    
    def filter(self, paths: List[Path]) -> List[Path]:
        """
        Filter paths to maximize diversity.
        
        Args:
            paths: List of paths to filter
            
        Returns:
            Filtered list of paths that maximize diversity
        """
        if not paths or len(paths) <= self.max_paths:
            return paths
        
        # Sort by reliability first
        sorted_paths = sorted(paths, key=lambda p: p.reliability, reverse=True)
        
        # Always include the top path
        result = [sorted_paths[0]]
        remaining = sorted_paths[1:]
        
        # Add paths that maximize diversity
        while len(result) < self.max_paths and remaining:
            # For each remaining path, calculate diversity score
            best_score = -1.0
            best_path_idx = 0
            
            for i, path in enumerate(remaining):
                # Calculate diversity as average Jaccard distance to existing paths
                diversity_score = 0.0
                for existing_path in result:
                    existing_nodes = set(existing_path.nodes)
                    current_nodes = set(path.nodes)
                    intersection = len(existing_nodes.intersection(current_nodes))
                    union = len(existing_nodes.union(current_nodes))
                    
                    # Jaccard distance = 1 - Jaccard similarity
                    if union > 0:
                        diversity_score += 1.0 - (intersection / union)
                
                avg_diversity = diversity_score / len(result)
                
                # Combined score: balance between reliability and diversity
                combined_score = (
                    (1.0 - self.diversity_weight) * path.reliability + 
                    self.diversity_weight * avg_diversity
                )
                
                if combined_score > best_score:
                    best_score = combined_score
                    best_path_idx = i
            
            # Add the best path to results and remove from remaining
            result.append(remaining.pop(best_path_idx))
        
        return result


class AnalyzableGraph(Protocol):
    """Protocol for graphs that support advanced analysis."""
    
    @abstractmethod
    def get_stats(self) -> GraphStats:
        """
        Get statistics about the graph structure.
        
        Returns:
            Statistics about the graph
        """
        pass
    
    @abstractmethod
    def get_path_stats(self, paths: List[Path]) -> PathStats:
        """
        Get statistics about a set of paths.
        
        Args:
            paths: List of paths to analyze
            
        Returns:
            Statistics about the paths
        """
        pass
    
    @abstractmethod
    def filter_paths(self, paths: List[Path], filter_method: PathFilterProtocol) -> List[Path]:
        """
        Filter paths using the provided filter.
        
        Args:
            paths: List of paths to filter
            filter_method: Path filtering implementation
            
        Returns:
            Filtered list of paths
        """
        pass
    
    @abstractmethod
    def get_subgraph(self, node_ids: Set[NodeIDType]) -> 'AnalyzableGraph':
        """
        Extract a subgraph containing only the specified nodes.
        
        Args:
            node_ids: Set of node IDs to include
            
        Returns:
            Subgraph containing only the specified nodes
        """
        pass

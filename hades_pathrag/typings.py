"""
Central typing module for HADES-PathRAG.

This module provides common type aliases and protocols used throughout the project.
It simplifies type annotations for complex structures and third-party libraries
like NetworkX and NumPy.
"""

from typing import (
    Any, Dict, List, Optional, Set, Tuple, Union, TypeVar, Generic,
    Protocol, runtime_checkable, Callable, Iterable, Mapping
)
import numpy as np
from numpy.typing import NDArray
import networkx as nx

# Type variables
T = TypeVar('T')
NodeIDType = str
EdgeType = TypeVar('EdgeType')
NodeType = TypeVar('NodeType')

# NumPy array type aliases
EmbeddingArray = NDArray[np.float32]  # Standard embedding array type
FloatArray = NDArray[np.float32]      # General floating point array
IntArray = NDArray[np.int64]          # Integer array type

# NetworkX type aliases with type ignore comments
# We use Any for node/edge data generics to simplify typing
Graph = nx.Graph  # type: ignore[type-arg]
DiGraph = nx.DiGraph  # type: ignore[type-arg] 
MultiGraph = nx.MultiGraph  # type: ignore[type-arg]
MultiDiGraph = nx.MultiDiGraph  # type: ignore[type-arg]

# Common structure type aliases
NodeData = Dict[str, Any]  # Data stored on nodes
EdgeData = Dict[str, Any]  # Data stored on edges
PathType = List[NodeIDType]  # A path as a sequence of node IDs
GraphDict = Dict[NodeIDType, Dict[NodeIDType, EdgeData]]  # Adjacency dictionary

# Specific to PathRAG
EmbeddingDict = Dict[NodeIDType, EmbeddingArray]
SimilarityScoreType = float
NodeScoreMapping = Dict[NodeIDType, SimilarityScoreType]

# Protocol types for interfaces
@runtime_checkable
class Embedding(Protocol):
    """Protocol for embedding objects."""
    
    def __len__(self) -> int:
        """Return embedding dimension."""
        ...
    
    def __getitem__(self, idx: int) -> float:
        """Access embedding values by index."""
        ...

@runtime_checkable
class EmbeddingProvider(Protocol):
    """Protocol for objects that can provide embeddings."""
    
    def get_embedding(self, key: str) -> Optional[EmbeddingArray]:
        """Get embedding for a given key."""
        ...
    
    def get_embedding_dimension(self) -> int:
        """Get dimension of embeddings."""
        ...

@runtime_checkable
class GraphLike(Protocol):
    """Protocol for graph-like objects."""
    
    def nodes(self) -> Iterable[NodeIDType]:
        """Return an iterable of node IDs."""
        ...
    
    def edges(self) -> Iterable[Tuple[NodeIDType, NodeIDType]]:
        """Return an iterable of edge tuples."""
        ...
    
    def get_node_data(self, node_id: NodeIDType) -> Optional[NodeData]:
        """Get data associated with a node."""
        ...
    
    def get_edge_data(self, u: NodeIDType, v: NodeIDType) -> Optional[EdgeData]:
        """Get data associated with an edge."""
        ...

"""
Graph interfaces for PathRAG.

This module provides graph data structure interfaces and implementations
for the PathRAG system.
"""
from typing import List, Type, TypeVar, Dict, Any, Optional, Set, Tuple

# Type variables for graph classes
T = TypeVar('T', bound='BaseGraph')
NodeID = str
EdgeID = str
Weight = float

# Import base classes and interfaces
from .base import BaseGraph, Path
from .interfaces import (
    AnalyzableGraph, GraphStats, PathStats, PathScoringConfig,
    PathFilterProtocol, ReliabilityFilter, DiversityFilter
)

# Import concrete implementations
from .networkx_impl import NetworkXGraph

# __all__ defines the public API
__all__: List[str] = [
    # Type definitions
    'NodeID',
    'EdgeID',
    'Weight',
    
    # Base classes and interfaces
    'BaseGraph',
    'Path',
    'AnalyzableGraph',
    'GraphStats',
    'PathStats',
    'PathScoringConfig',
    'PathFilterProtocol',
    
    # Filter implementations
    'ReliabilityFilter',
    'DiversityFilter',
    
    # Concrete implementations
    'NetworkXGraph'
]

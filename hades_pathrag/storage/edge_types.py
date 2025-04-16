"""
Edge type definitions for the PathRAG graph.

This module defines the edge types used in the PathRAG graph,
along with their weights and properties.
"""
from typing import Dict, List, Optional, Tuple, Any, Set, Type, cast, Union
from enum import Enum
from dataclasses import dataclass, field

from hades_pathrag.typings import EdgeData


class EdgeCategory(Enum):
    """Categories of edge relationship types."""
    PRIMARY = "primary"      # Highest importance relationships (0.8-1.0)
    SECONDARY = "secondary"  # Medium importance relationships (0.5-0.7)
    TERTIARY = "tertiary"    # Lower importance relationships (0.2-0.4)


@dataclass
class EdgeTypeDefinition:
    """Definition of an edge type with its properties."""
    name: str
    category: EdgeCategory
    base_weight: float
    description: str
    bidirectional: bool = False
    default_attributes: Dict[str, Any] = field(default_factory=dict)

    # No need for post_init with field(default_factory=dict)


# Define standard edge types
EDGE_TYPES = {
    # PRIMARY RELATIONSHIPS (0.8-1.0)
    "CALLS": EdgeTypeDefinition(
        name="CALLS",
        category=EdgeCategory.PRIMARY,
        base_weight=0.9,
        description="Direct function calls (function → function)",
        bidirectional=False,
    ),
    "CONTAINS": EdgeTypeDefinition(
        name="CONTAINS",
        category=EdgeCategory.PRIMARY,
        base_weight=0.95,
        description="Containment relationships (class → method, file → function)",
        bidirectional=False,
    ),
    "IMPLEMENTS": EdgeTypeDefinition(
        name="IMPLEMENTS",
        category=EdgeCategory.PRIMARY,
        base_weight=0.85,
        description="Implementation relationships (interface → class)",
        bidirectional=False,
    ),
    
    # SECONDARY RELATIONSHIPS (0.5-0.7)
    "IMPORTS": EdgeTypeDefinition(
        name="IMPORTS",
        category=EdgeCategory.SECONDARY,
        base_weight=0.7,
        description="Module/package imports (file → file)",
        bidirectional=False,
    ),
    "REFERENCES": EdgeTypeDefinition(
        name="REFERENCES",
        category=EdgeCategory.SECONDARY,
        base_weight=0.6,
        description="Code references or uses (variable → function)",
        bidirectional=False,
    ),
    "EXTENDS": EdgeTypeDefinition(
        name="EXTENDS",
        category=EdgeCategory.SECONDARY,
        base_weight=0.65,
        description="Inheritance relationships (class → class)",
        bidirectional=False,
    ),
    
    # TERTIARY RELATIONSHIPS (0.2-0.4)
    "SIMILAR_TO": EdgeTypeDefinition(
        name="SIMILAR_TO",
        category=EdgeCategory.TERTIARY,
        base_weight=0.4,
        description="Semantic similarity (any → any)",
        bidirectional=True,
    ),
    "DOCUMENTED_BY": EdgeTypeDefinition(
        name="DOCUMENTED_BY",
        category=EdgeCategory.TERTIARY,
        base_weight=0.35,
        description="Documentation relationship (code → documentation)",
        bidirectional=False,
    ),
    "RELATED_TO": EdgeTypeDefinition(
        name="RELATED_TO",
        category=EdgeCategory.TERTIARY,
        base_weight=0.3,
        description="General relationship without specific type",
        bidirectional=True,
    ),
}


def get_edge_weight(edge_type: str, custom_weight: Optional[float] = None) -> float:
    """
    Get the weight for an edge type, with optional custom override.
    
    Args:
        edge_type: Name of the edge type
        custom_weight: Optional custom weight to override base weight
        
    Returns:
        Edge weight (0.0-1.0)
    """
    if edge_type not in EDGE_TYPES:
        # Default weight for unknown edge types
        return 0.1
    
    # Use custom weight if provided, otherwise use base weight
    return custom_weight if custom_weight is not None else EDGE_TYPES[edge_type].base_weight


def create_edge_data(
    edge_type: str,
    custom_weight: Optional[float] = None,
    custom_attributes: Optional[Dict[str, Any]] = None,
) -> EdgeData:
    """
    Create edge data with appropriate type, weight, and attributes.
    
    Args:
        edge_type: Type of the edge (e.g., CALLS, IMPORTS)
        custom_weight: Optional custom weight to override base weight
        custom_attributes: Optional additional attributes to include
        
    Returns:
        Edge data dictionary
    """
    # Get the edge type definition
    edge_def = EDGE_TYPES.get(edge_type)
    
    # Base edge data
    edge_data: EdgeData = {
        "relation_type": edge_type,
        "weight": get_edge_weight(edge_type, custom_weight),
        "bidirectional": edge_def.bidirectional if edge_def else False,
        "category": edge_def.category.value if edge_def else EdgeCategory.TERTIARY.value,
    }
    
    # Add default attributes from the edge type definition
    if edge_def and edge_def.default_attributes:
        edge_data.update(edge_def.default_attributes)
    
    # Add custom attributes if provided
    if custom_attributes:
        edge_data.update(custom_attributes)
    
    return edge_data

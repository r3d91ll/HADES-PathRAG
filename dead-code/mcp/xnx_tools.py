"""
MCP tools for XnX-enhanced PathRAG.

This module provides MCP tools that integrate XnX notation with PathRAG
for the HADES system, following the entity-relationship model pattern.
"""

import sys
from typing import Dict, List, Any, Optional, Union, TypedDict
from datetime import datetime, timedelta

# Try to import from our structured location
try:
    from src.xnx import XnXPathRAG, XnXQueryParams, XnXIdentityToken
    from src.db.arango_connection import ArangoConnection
except ImportError:
    # Fall back to old_hades_imports
    sys.path.append('old_hades_imports')
    from src.db.arango_connection import ArangoConnection
    sys.path.append('.')
    from src.xnx import XnXPathRAG, XnXQueryParams, XnXIdentityToken

# Placeholder for MCP tool registration decorator
# In a real implementation, this would come from the MCP server
from typing import TypeVar, Callable, Any, cast, Protocol, runtime_checkable

@runtime_checkable
class MCPTool(Protocol):
    """Protocol for MCP tools."""
    _is_mcp_tool: bool

F = TypeVar('F', bound=Callable[..., Any])

def mcp_tool(func: F) -> F:
    """Decorator to register an MCP tool."""
    setattr(func, '_is_mcp_tool', True)
    return func


# Create a global instance for simplicity
# In production, this would be part of the MCP server's state
_arango_connection: Optional[ArangoConnection] = None
_xnx_pathrag: Optional[XnXPathRAG] = None

def _get_arango_connection() -> ArangoConnection:
    """Get or create ArangoDB connection."""
    global _arango_connection
    if _arango_connection is None:
        _arango_connection = ArangoConnection()
    return _arango_connection

def _get_xnx_pathrag() -> XnXPathRAG:
    """Get or create XnXPathRAG instance."""
    global _xnx_pathrag
    if _xnx_pathrag is None:
        from PathRAG.llm import gpt_4o_mini_complete  # Placeholder, use actual LLM
        _xnx_pathrag = XnXPathRAG(
            working_dir="./path_cache",
            llm_model_func=gpt_4o_mini_complete,
            arango_adapter=None  # Will create default
        )
    return _xnx_pathrag


# TypedDict definitions for MCP tool return types
class XnxParamsDict(TypedDict, total=False):
    """Parameters for XnX queries in response format."""
    min_weight: float
    max_distance: int
    direction: Optional[int]
    as_of_version: Optional[str]

class XnxRetrieveResponse(TypedDict):
    """Response format for mcp0_xnx_pathrag_retrieve."""
    query: str
    xnx_params: XnxParamsDict
    paths: List[Any]
    response: str

class XnxRelationshipResponse(TypedDict):
    edge_id: Any
    from_entity: str
    to_entity: str
    weight: float
    direction: int
    temporal_bounds: Optional[Dict[str, str]]
    xnx_notation: str
    metadata: Dict[str, Any]

class XnxIdentityResponse(TypedDict):
    token_id: str
    user_id: str
    object_id: str
    relationship_weight: float
    created_at: str
    expires_at: str
    is_valid: bool
    effective_weight: Any

class XnxVerifyAccessResponse(TypedDict, total=False):
    access_granted: bool
    reason: str
    user_id: str
    resource_id: str
    path_weight: Optional[float]
    path_length: Optional[int]
    using_assumed_identity: Optional[bool]
    highest_weight: Optional[float]
    required_weight: Optional[float]


@mcp_tool
def mcp0_xnx_pathrag_retrieve(
    query: str,
    domain_filter: Optional[str] = None,
    min_weight: float = 0.5,
    max_distance: int = 3,
    direction: Optional[int] = None,
    as_of_version: Optional[str] = None
) -> XnxRetrieveResponse:
    """Retrieve paths from the knowledge graph using XnX PathRAG.
    
    Args:
        query: The query to retrieve paths for
        domain_filter: Optional domain filter
        min_weight: Minimum weight threshold for relationships (0.0 to 1.0)
        max_distance: Maximum number of hops to traverse
        direction: Flow direction (1=inbound, -1=outbound, None=both)
        as_of_version: Optional version for temporal queries
        
    Returns:
        Dictionary with query results and paths
    """
    # Create XnX query parameters
    query_params = XnXQueryParams(
        min_weight=min_weight,
        max_distance=max_distance,
        direction=direction,
        temporal_constraint=as_of_version
    )
    
    # Get XnXPathRAG instance
    pathrag = _get_xnx_pathrag()
    
    # Execute query with XnX parameters
    results = pathrag.query(query, xnx_params=query_params)
    
    # Format results for MCP response
    response_params: XnxParamsDict = {
        "min_weight": min_weight,
        "max_distance": max_distance
    }
    
    # Add optional parameters only if they are not None
    if direction is not None:
        response_params["direction"] = direction
    if as_of_version is not None:
        response_params["as_of_version"] = as_of_version
        
    response: XnxRetrieveResponse = {
        "query": query,
        "xnx_params": response_params,
        "paths": results.get("xnx_paths", []),
        "response": results.get("response", "")
    }
    
    return response


@mcp_tool
def mcp0_xnx_create_relationship(
    from_entity: str,
    to_entity: str,
    weight: float = 1.0,
    direction: int = -1,
    valid_from: Optional[str] = None,
    valid_until: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> XnxRelationshipResponse:
    """Create a relationship with XnX notation between entities.
    
    Args:
        from_entity: Source entity name
        to_entity: Target entity name
        weight: Relationship weight (0.0 to 1.0)
        direction: Flow direction (-1=outbound, 1=inbound)
        valid_from: Optional start time for temporal constraint
        valid_until: Optional end time for temporal constraint
        metadata: Additional metadata
        
    Returns:
        Dictionary with relationship information
    """
    # Format temporal bounds if provided
    temporal_bounds = None
    if valid_from or valid_until:
        temporal_bounds = {
            "start": valid_from or "",
            "end": valid_until or ""
        }
    
    # Get XnXPathRAG instance
    pathrag = _get_xnx_pathrag()
    
    # Create relationship
    edge_id = pathrag.create_relationship(
        from_id=from_entity,
        to_id=to_entity,
        weight=weight,
        direction=direction,
        temporal_bounds=temporal_bounds,
        metadata=metadata
    )
    
    # Format XnX notation string
    xnx_notation = f"{weight:.2f} {to_entity} {direction}"
    if temporal_bounds:
        time_str = f"[T[{valid_from}→{valid_until}]]"
        xnx_notation = f"{weight:.2f} {to_entity} {direction}{time_str}"
    
    # Return relationship data
    return {
        "edge_id": edge_id,
        "from_entity": from_entity,
        "to_entity": to_entity,
        "weight": weight,
        "direction": direction,
        "temporal_bounds": temporal_bounds,
        "xnx_notation": xnx_notation,
        "metadata": metadata or {}
    }


@mcp_tool
def mcp0_xnx_assume_identity(
    user_id: str,
    object_id: str,
    duration_minutes: int = 60
) -> XnxIdentityResponse:
    """Create an identity assumption token for a user to act as an object.
    
    Args:
        user_id: ID of the user assuming the identity
        object_id: ID of the object whose identity is being assumed
        duration_minutes: How long the token is valid in minutes
        
    Returns:
        Dictionary with token information
    """
    # Get XnXPathRAG instance
    pathrag = _get_xnx_pathrag()
    
    # Create identity token
    token = pathrag.assume_identity(
        user_id=user_id,
        object_id=object_id,
        expiration_minutes=duration_minutes
    )
    
    # Return token information
    return {
        "token_id": token.token_id,
        "user_id": token.user_id,
        "object_id": token.object_id,
        "relationship_weight": token.relationship_weight,
        "created_at": token.created_at.isoformat(),
        "expires_at": (token.created_at + timedelta(minutes=token.expiration_minutes)).isoformat(),
        "is_valid": token.is_valid,
        "effective_weight": token.effective_weight
    }


@mcp_tool
def mcp0_xnx_verify_access(
    user_id: str,
    resource_id: str,
    min_weight: float = 0.7,
    identity_token_id: Optional[str] = None
) -> XnxVerifyAccessResponse:
    """Verify if a user has access to a resource using XnX access control.
    
    Args:
        user_id: ID of the user requesting access
        resource_id: ID of the resource to access
        min_weight: Minimum weight required for access
        identity_token_id: Optional identity token ID for identity assumption
        
    Returns:
        Dictionary with access verification result
    """
    # Get XnXPathRAG instance
    pathrag = _get_xnx_pathrag()
    
    # If identity token provided, validate it
    identity_token = None
    if identity_token_id and identity_token_id in pathrag.identity_tokens:
        identity_token = pathrag.identity_tokens[identity_token_id]
        if not identity_token.is_valid:
            return {
                "access_granted": False,
                "reason": "Identity token has expired",
                "user_id": user_id,
                "resource_id": resource_id
            }
            
        # If token is for a different user, deny access
        if identity_token.user_id != user_id:
            return {
                "access_granted": False,
                "reason": "Identity token belongs to a different user",
                "user_id": user_id,
                "resource_id": resource_id
            }
    
    # Create XnX parameters for access check
    xnx_params = XnXQueryParams(
        min_weight=min_weight,
        identity_token=identity_token
    )
    
    # Use a simple query to check access
    # In a real implementation, we would have a specialized access check method
    query = f"Does {user_id} have access to {resource_id}?"
    results = pathrag.query(query, xnx_params=xnx_params)
    
    # Simple check: if we got paths with enough weight, grant access
    paths = results.get("xnx_paths", [])
    if not paths:
        return {
            "access_granted": False,
            "reason": "No access path found",
            "user_id": user_id,
            "resource_id": resource_id
        }
        
    # Check if any path has sufficient weight
    highest_weight_path = max(paths, key=lambda p: p.get("avg_weight", 0))
    if highest_weight_path.get("avg_weight", 0) >= min_weight:
        return {
            "access_granted": True,
            "path_weight": highest_weight_path.get("avg_weight", 0),
            "path_length": highest_weight_path.get("length", 0),
            "user_id": user_id,
            "resource_id": resource_id,
            "using_assumed_identity": identity_token is not None
        }
    else:
        return {
            "access_granted": False,
            "reason": "Path weight insufficient",
            "highest_weight": highest_weight_path.get("avg_weight", 0),
            "required_weight": min_weight,
            "user_id": user_id,
            "resource_id": resource_id
        }


@mcp_tool
def mcp0_self_analyze(query: str,
                     target_components: Optional[List[str]] = None,
                     min_confidence: float = 0.7) -> Dict[str, Any]:
    """MCP tool for HADES to analyze its own codebase.
    
    This recursive tool enables HADES to examine its own implementation,
    understand its components, and propose improvements.
    
    Args:
        query: Natural language query about HADES itself
        target_components: Optional list of specific components to analyze
        min_confidence: Minimum confidence threshold for results
    
    Returns:
        Analysis of the requested HADES components
    """
    # Get XnXPathRAG instance
    pathrag = _get_xnx_pathrag()
    
    # Build self-referential query
    enhanced_query = query
    if target_components:
        enhanced_query = f"{query} Focus on the following components: {', '.join(target_components)}"
    
    # Execute query with XnX parameters
    xnx_params = XnXQueryParams(
        min_weight=min_confidence,
        max_distance=3,
        direction=-1  # Only consider outbound relationships for code analysis
    )
    
    # Query the knowledge graph
    results = pathrag.query(enhanced_query, xnx_params=xnx_params)
    
    # Extract code from paths
    code_segments = []
    for path in results.get("xnx_paths", []):
        for node in path.get("nodes", []):
            if node.get("type") == "code_function" or node.get("type") == "code_class":
                code_segments.append({
                    "content": node.get("content", ""),
                    "file_path": node.get("metadata", {}).get("file_path", ""),
                    "entity_type": node.get("type", ""),
                    "name": node.get("name", "")
                })
    
    # Analyze code quality (placeholder for now)
    code_quality = {
        "complexity": 0.85,
        "maintainability": 0.78,
        "test_coverage": 0.65,
        "performance": 0.92
    }
    
    # Generate improvement suggestions (placeholder for now)
    suggestions = [
        {
            "metric": "test_coverage",
            "current_value": 0.65,
            "target_value": 0.8,
            "suggestion": "Add tests for the PathRAG query methods",
            "affected_files": ["src/pathrag/PathRAG.py"]
        }
    ]
    
    return {
        "paths": results.get("xnx_paths", []),
        "code_segments": code_segments,
        "code_quality": code_quality,
        "improvement_suggestions": suggestions,
        "query_parameters": {
            "query": query,
            "target_components": target_components,
            "min_confidence": min_confidence
        }
    }

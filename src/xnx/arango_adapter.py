"""
ArangoDB adapter for XnX-enhanced PathRAG.

This module provides an adapter to replace the default PathRAG storage
with ArangoDB, supporting XnX notation for weighted path tuning.
"""

import sys
from typing import Dict, List, Any, Optional, Tuple
import json
from datetime import datetime

# Create dynamic import based on where we import ArangoDB connection from
try:
    # Try to import from our main package structure
    from src.db.arango_connection import ArangoConnection
except ImportError:
    try:
        # Fall back to the old_hades_imports
        sys.path.append('old_hades_imports')
        from src.db.arango_connection import ArangoConnection
    except ImportError:
        raise ImportError("Could not find ArangoConnection in either location")


class ArangoPathRAGAdapter:
    """Adapter that connects PathRAG to ArangoDB with XnX notation support."""
    
    def __init__(self, 
                 arango_connection: Optional[ArangoConnection] = None,
                 db_name: str = "hades",
                 nodes_collection: str = "pathrag_nodes",
                 edges_collection: str = "pathrag_edges",
                 graph_name: str = "pathrag_graph"):
        """Initialize the ArangoDB adapter.
        
        Args:
            arango_connection: Existing ArangoDB connection or None to create new
            db_name: Name of the ArangoDB database
            nodes_collection: Name of the collection storing nodes
            edges_collection: Name of the collection storing edges
            graph_name: Name of the graph in ArangoDB
        """
        # Use provided connection or create a new one
        self.conn = arango_connection or ArangoConnection(db_name=db_name)
        self.db_name = db_name
        self.nodes_collection = nodes_collection
        self.edges_collection = edges_collection
        self.graph_name = graph_name
        
        # Ensure collections and graph exist
        self._ensure_collections()
        
    def _ensure_collections(self):
        """Ensure that required collections and graph exist."""
        # Create nodes collection if it doesn't exist
        if not self.conn.collection_exists(self.nodes_collection):
            self.conn.create_collection(self.nodes_collection)
            
        # Create edges collection if it doesn't exist
        if not self.conn.collection_exists(self.edges_collection):
            self.conn.create_edge_collection(self.edges_collection)
            
        # Create graph if it doesn't exist
        if not self.conn.graph_exists(self.graph_name):
            self.conn.create_graph(
                self.graph_name,
                edge_definitions=[{
                    'collection': self.edges_collection,
                    'from': [self.nodes_collection],
                    'to': [self.nodes_collection]
                }]
            )
    
    def store_node(self, node_id: str, content: str, 
                  embedding: List[float], metadata: Dict[str, Any] = None) -> str:
        """Store a node in ArangoDB.
        
        Args:
            node_id: Unique identifier for the node
            content: Text content of the node
            embedding: Vector embedding of the node content
            metadata: Additional metadata for the node
            
        Returns:
            ArangoDB document key
        """
        metadata = metadata or {}
        
        # Create node document
        node_doc = {
            "_key": node_id,
            "content": content,
            "embedding": embedding,
            "metadata": metadata,
            "created_at": datetime.now().isoformat()
        }
        
        # Insert or replace the node
        result = self.conn.insert_document(
            self.nodes_collection, 
            node_doc, 
            overwrite=True
        )
        
        return result["_key"]
    
    def create_relationship(self, from_id: str, to_id: str, 
                           weight: float = 1.0, 
                           direction: int = -1,
                           temporal_bounds: Optional[Dict[str, str]] = None,
                           metadata: Dict[str, Any] = None) -> str:
        """Create an XnX relationship between nodes.
        
        Args:
            from_id: Source node ID
            to_id: Target node ID
            weight: XnX weight (0.0 to 1.0)
            direction: XnX direction (-1 for outbound, 1 for inbound)
            temporal_bounds: Optional time bounds for relationship
            metadata: Additional metadata
            
        Returns:
            ArangoDB edge document key
        """
        metadata = metadata or {}
        
        # Create edge document with XnX notation properties
        edge_doc = {
            "_from": f"{self.nodes_collection}/{from_id}",
            "_to": f"{self.nodes_collection}/{to_id}",
            "weight": weight,
            "direction": direction,
            "xnx_notation": f"{weight:.2f} {to_id} {direction}",
            "created_at": datetime.now().isoformat(),
            "metadata": metadata
        }
        
        # Add temporal bounds if provided
        if temporal_bounds:
            edge_doc["temporal_bounds"] = temporal_bounds
            # Update XnX notation to include temporal bounds
            time_str = f"[T[{temporal_bounds.get('start')}→{temporal_bounds.get('end')}]]"
            edge_doc["xnx_notation"] = f"{weight:.2f} {to_id} {direction}{time_str}"
        
        result = self.conn.insert_edge(
            self.edges_collection,
            edge_doc
        )
        
        return result["_key"]
    
    def query_paths(self, query_embedding: List[float], 
                   xnx_params, max_results: int = 10) -> List[Dict]:
        """Query paths based on XnX parameters.
        
        Args:
            query_embedding: Vector embedding for similarity search
            xnx_params: XnX query parameters
            max_results: Maximum number of results to return
            
        Returns:
            List of paths matching the query
        """
        # Construct AQL query with XnX filters
        min_weight = xnx_params.min_weight
        max_distance = xnx_params.max_distance
        direction = xnx_params.direction
        
        # Temporal constraint handling
        temporal_constraint = None
        if xnx_params.temporal_constraint:
            if isinstance(xnx_params.temporal_constraint, datetime):
                temporal_constraint = xnx_params.temporal_constraint.isoformat()
            else:
                temporal_constraint = xnx_params.temporal_constraint
        
        # Start with vector similarity search to find entry points
        query = f"""
        LET entry_points = (
            FOR doc IN {self.nodes_collection}
            LET similarity = VECTOR_DISTANCE(doc.embedding, @query_embedding)
            SORT similarity ASC
            LIMIT 5
            RETURN doc
        )
        
        LET paths = (
            FOR entry IN entry_points
            FOR v, e, p IN 1..@max_distance OUTBOUND entry._id 
            GRAPH @graph_name
            FILTER e.weight >= @min_weight
        """
        
        # Add direction filter if specified
        if direction is not None:
            query += f" FILTER e.direction == @direction"
            
        # Add temporal constraint if specified
        if temporal_constraint:
            query += f"""
            FILTER (
                e.temporal_bounds == NULL OR
                (e.temporal_bounds.start <= @temporal_constraint AND 
                 (e.temporal_bounds.end == NULL OR e.temporal_bounds.end >= @temporal_constraint))
            )
            """
            
        # Complete the query with sorting and limiting
        query += f"""
            SORT p.edges[*].weight DESC
            RETURN {{
                "path": p,
                "total_weight": SUM(p.edges[*].weight),
                "avg_weight": AVG(p.edges[*].weight),
                "length": LENGTH(p.edges),
                "content": LAST(p.vertices).content
            }}
        )
        
        FOR path IN paths
        SORT path.avg_weight * (1.0 / path.length) DESC
        LIMIT @max_results
        RETURN path
        """
        
        # Prepare query parameters
        bind_vars = {
            "query_embedding": query_embedding,
            "max_distance": max_distance,
            "min_weight": min_weight,
            "graph_name": self.graph_name,
            "max_results": max_results
        }
        
        if direction is not None:
            bind_vars["direction"] = direction
            
        if temporal_constraint:
            bind_vars["temporal_constraint"] = temporal_constraint
        
        # Execute the query
        results = self.conn.execute_query(query, bind_vars=bind_vars)
        
        return results

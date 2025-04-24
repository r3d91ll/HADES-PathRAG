"""
ArangoDB storage adapter for hierarchical document graphs.

This module provides functionality to store and retrieve hierarchical document 
structures in ArangoDB, maintaining the "turtles all the way down" pattern.
"""
from typing import Dict, List, Any, Optional, Union, TypedDict
import uuid
import logging
from datetime import datetime

try:
    from arango import ArangoClient  # type: ignore[import-not-found]
    from arango.exceptions import ArangoError  # type: ignore[import-not-found]
except ImportError:
    raise ImportError(
        "python-arango is required for ArangoGraphStore. "
        "Install with: pip install python-arango"
    )

from ..models.graph_models import BaseNode, Edge, HierarchicalGraph

logger = logging.getLogger(__name__)


class StoreResults(TypedDict):
    node_count: int
    edge_count: int
    root_id: str
    failed_nodes: List[str]
    failed_edges: List[Dict[str, str]]


class ArangoGraphStore:
    """
    Storage adapter for hierarchical document graphs in ArangoDB.
    
    This class handles the persistence of hierarchical document structures,
    preserving all node types, relationships, and the recursive "nested graph"
    pattern.
    """
    
    # Default collection names
    DEFAULT_NODE_COLLECTION = "Nodes"
    DEFAULT_EDGE_COLLECTION = "Edges"
    DEFAULT_GRAPH_NAME = "DocumentGraph"
    
    def __init__(self, 
                 host: str = "http://localhost:8529",
                 username: str = "root",
                 password: str = "",
                 database: str = "pathrag",
                 node_collection: Optional[str] = None,
                 edge_collection: Optional[str] = None,
                 graph_name: Optional[str] = None):
        """
        Initialize the ArangoDB graph store.
        
        Args:
            host: ArangoDB host URL
            username: ArangoDB username
            password: ArangoDB password
            database: Database name
            node_collection: Collection name for nodes
            edge_collection: Collection name for edges
            graph_name: Graph name
        """
        self.client = ArangoClient(hosts=host)
        self.db = self.client.db(database, username=username, password=password)
        
        self.node_collection_name = node_collection or self.DEFAULT_NODE_COLLECTION
        self.edge_collection_name = edge_collection or self.DEFAULT_EDGE_COLLECTION
        self.graph_name = graph_name or self.DEFAULT_GRAPH_NAME
        
        # Ensure collections exist
        self._setup_collections()
    
    def _setup_collections(self) -> None:
        """Ensure required collections and graph exist."""
        # Create node collection if it doesn't exist
        if not self.db.has_collection(self.node_collection_name):
            self.db.create_collection(self.node_collection_name)
            logger.info(f"Created node collection: {self.node_collection_name}")
        
        # Create edge collection if it doesn't exist
        if not self.db.has_collection(self.edge_collection_name):
            self.db.create_collection(
                self.edge_collection_name, edge=True
            )
            logger.info(f"Created edge collection: {self.edge_collection_name}")
        
        # Create graph if it doesn't exist
        if not self.db.has_graph(self.graph_name):
            graph = self.db.create_graph(self.graph_name)
            # Define edge definition
            graph.create_edge_definition(
                edge_collection=self.edge_collection_name,
                from_vertex_collections=[self.node_collection_name],
                to_vertex_collections=[self.node_collection_name]
            )
            logger.info(f"Created graph: {self.graph_name}")
    
    def store_hierarchical_graph(self, graph: HierarchicalGraph) -> Dict[str, Any]:
        """
        Store a complete hierarchical document graph in ArangoDB.
        
        Args:
            graph: The hierarchical graph to store
            
        Returns:
            Dictionary with storage results
        """
        # Get collections
        nodes_collection = self.db.collection(self.node_collection_name)
        edges_collection = self.db.collection(self.edge_collection_name)
        
        results: StoreResults = {
            "node_count": 0,
            "edge_count": 0,
            "root_id": graph.root_node.id,
            "failed_nodes": [],
            "failed_edges": []
        }
        
        # Store all nodes
        for node in graph.nodes:
            try:
                node_data = node.to_dict()
                # Convert ID to ArangoDB _key format
                node_data["_key"] = node_data.pop("id")
                # Store node
                nodes_collection.insert(node_data, overwrite=True)
                results["node_count"] += 1
            except ArangoError as e:
                logger.error(f"Failed to store node {node.id}: {str(e)}")
                results["failed_nodes"].append(node.id)
        
        # Store all edges
        for edge in graph.edges:
            try:
                edge_data = edge.to_dict()
                # Convert IDs to ArangoDB format
                edge_data["_from"] = f"{self.node_collection_name}/{edge_data.pop('from_id')}"
                edge_data["_to"] = f"{self.node_collection_name}/{edge_data.pop('to_id')}"
                edge_data["_key"] = str(uuid.uuid4())
                # Store edge
                edges_collection.insert(edge_data)
                results["edge_count"] += 1
            except ArangoError as e:
                logger.error(f"Failed to store edge {edge.from_id}->{edge.to_id}: {str(e)}")
                results["failed_edges"].append({
                    "from": edge.from_id,
                    "to": edge.to_id
                })
        
        logger.info(
            f"Stored hierarchical graph: {results['node_count']} nodes, "
            f"{results['edge_count']} edges"
        )
        
        return results
    
    def get_document_by_id(self, document_id: str) -> Dict[str, Any]:
        """
        Retrieve a complete document hierarchy by document ID.
        
        This includes the document node and all its child nodes recursively.
        
        Args:
            document_id: ID of the root document node
            
        Returns:
            Complete hierarchical structure
        """
        # Get the document node
        document_node = self.db.collection(self.node_collection_name).get(document_id)
        if not document_node:
            raise ValueError(f"Document node {document_id} not found")
        
        # Execute a graph traversal to get all nodes in the document hierarchy
        traversal_results = self.db.graph(self.graph_name).traverse(
            start_vertex=f"{self.node_collection_name}/{document_id}",
            direction="outbound",  # Follow outbound edges from document
            strategy="dfs",  # Depth-first search
            edge_uniqueness="global",  # Only include each edge once
            vertex_uniqueness="global",  # Only include each vertex once
        )
        
        # Structure the results
        nodes = []
        edges = []
        
        # Add all vertices (nodes)
        for vertex in traversal_results["vertices"]:
            # Convert _key back to id
            vertex["id"] = vertex.pop("_key")
            nodes.append(vertex)
        
        # Add all edges
        for edge in traversal_results["edges"]:
            # Convert _from and _to to from_id and to_id
            from_id = edge["_from"].split("/")[1]
            to_id = edge["_to"].split("/")[1]
            edges.append({
                "from_id": from_id,
                "to_id": to_id,
                "type": edge.get("type", "generic"),
                "weight": edge.get("weight", 1.0),
                "metadata": edge.get("metadata", {})
            })
        
        # Return reconstructed hierarchy
        return {
            "root_id": document_id,
            "nodes": nodes,
            "edges": edges
        }
    
    def search_nodes(self, query: str, node_types: Optional[List[str]] = None, 
                    limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search for nodes matching the given query.
        
        Args:
            query: Search query
            node_types: Optional list of node types to filter by
            limit: Maximum number of results to return
            
        Returns:
            List of matching nodes
        """
        aql_filters = []
        bind_vars = {"query": f"%{query}%", "limit": limit}
        
        # Add filter for node types if provided
        if node_types:
            aql_filters.append("FILTER doc.type IN @node_types")
            bind_vars["node_types"] = node_types
        
        # Build the AQL query
        aql = f"""
        FOR doc IN {self.node_collection_name}
          FILTER doc.title LIKE @query OR doc.content LIKE @query
          {' '.join(aql_filters)}
          LIMIT @limit
          RETURN {{ 
            id: doc._key, 
            type: doc.type,
            title: doc.title,
            metadata: doc.metadata
          }}
        """
        
        # Execute query
        cursor = self.db.aql.execute(aql, bind_vars=bind_vars)
        return [document for document in cursor]

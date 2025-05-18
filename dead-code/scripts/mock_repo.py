"""
Mock repository for testing the ingestor.

This module provides mock implementations of the database repository classes
to enable testing without an actual database connection.
"""

from typing import Dict, Any, List, Optional, Union
import logging

from src.isne.types.models import DocumentRelation
from src.types.common import EmbeddingVector

logger = logging.getLogger(__name__)


class MockArangoConnection:
    """Mock implementation of ArangoConnection for testing."""
    
    def __init__(self, **kwargs):
        """Initialize mock connection."""
        self.collections = {}
        self.graphs = {}
        self.connected = True
        logger.info("Initialized mock ArangoDB connection")
    
    def disconnect(self):
        """Mock disconnect method."""
        self.connected = False
        logger.info("Mock ArangoDB disconnected")
    
    def create_collection(self, name: str, **kwargs) -> Dict[str, Any]:
        """Mock create_collection method."""
        if name not in self.collections:
            self.collections[name] = []
            logger.info(f"Created mock collection: {name}")
        return {"name": name, "status": "created"}
    
    def create_graph(self, name: str, edge_definitions: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        """Mock create_graph method."""
        if name not in self.graphs:
            self.graphs[name] = {"edges": edge_definitions, "nodes": []}
            logger.info(f"Created mock graph: {name}")
        return {"name": name, "status": "created"}
    
    def delete_collection(self, name: str) -> Dict[str, Any]:
        """Mock delete_collection method."""
        if name in self.collections:
            del self.collections[name]
            logger.info(f"Deleted mock collection: {name}")
        return {"name": name, "status": "deleted"}
    
    def delete_graph(self, name: str) -> Dict[str, Any]:
        """Mock delete_graph method."""
        if name in self.graphs:
            del self.graphs[name]
            logger.info(f"Deleted mock graph: {name}")
        return {"name": name, "status": "deleted"}


class MockArangoRepository:
    """Mock implementation of ArangoRepository for testing."""
    
    def __init__(self, connection):
        """Initialize mock repository."""
        self.connection = connection
        self.nodes = []
        self.edges = []
        self.vectors = []
        logger.info("Initialized mock ArangoDB repository")
    
    def initialize(self):
        """Mock initialize method."""
        logger.info("Initialized mock repository collections and graphs")
        return True
    
    def store_node(self, node_data: Dict[str, Any], collection: str = "nodes") -> str:
        """Mock store_node method."""
        node_id = node_data.get("id", f"node_{len(self.nodes)}")
        node_data["_id"] = f"{collection}/{node_id}"
        self.nodes.append(node_data)
        return node_id
    
    def store_edge(
        self,
        from_id: str,
        to_id: str,
        edge_data: Dict[str, Any],
        edge_collection: str = "edges"
    ) -> str:
        """Mock store_edge method."""
        edge_id = f"edge_{len(self.edges)}"
        edge = {
            "_id": f"{edge_collection}/{edge_id}",
            "_from": from_id,
            "_to": to_id,
            **edge_data
        }
        self.edges.append(edge)
        return edge_id
    
    def get_node(self, node_id: str, collection: str = "nodes") -> Optional[Dict[str, Any]]:
        """Mock get_node method."""
        for node in self.nodes:
            if node.get("id") == node_id or node.get("_id") == f"{collection}/{node_id}":
                return node
        return None
    
    def get_path(self, start_id: str, end_id: str) -> List[Dict[str, Any]]:
        """Mock get_path method."""
        # Just return a direct path for testing
        return [
            {"id": start_id, "type": "node"},
            {"id": end_id, "type": "node"}
        ]
    
    def store_vector(
        self,
        node_id: str,
        vector: EmbeddingVector,
        collection: str = "vectors"
    ) -> str:
        """Mock store_vector method."""
        vector_id = f"vector_{len(self.vectors)}"
        vector_data = {
            "_id": f"{collection}/{vector_id}",
            "node_id": node_id,
            "vector": vector
        }
        self.vectors.append(vector_data)
        return vector_id
    
    def find_similar(
        self,
        vector: EmbeddingVector,
        limit: int = 10,
        collection: str = "vectors"
    ) -> List[Dict[str, Any]]:
        """Mock find_similar method."""
        # Just return some nodes for testing
        return self.nodes[:min(limit, len(self.nodes))]

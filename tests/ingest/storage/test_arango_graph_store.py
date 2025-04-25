"""
Tests for the ArangoDB graph store module.
"""
import unittest
from unittest.mock import patch, MagicMock
import pytest
from typing import Dict, List, Any, Optional, Tuple

from src.ingest.storage.arango_graph_store import (
    ArangoGraphStore,
    ArangoConnectionError
)
from src.ingest.models.graph_models import Node, Edge, GraphRelationship


class TestArangoGraphStore(unittest.TestCase):
    """Test suite for ArangoGraphStore class."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        # Mock the ArangoDB client
        self.connection_patcher = patch('src.ingest.storage.arango_graph_store.ArangoConnection')
        self.mock_connection_class = self.connection_patcher.start()
        self.mock_connection = MagicMock()
        self.mock_connection_class.return_value = self.mock_connection
        
        # Create a db mock that will be returned by the connection
        self.mock_db = MagicMock()
        self.mock_connection.db = self.mock_db
        
        # Mock collections
        self.mock_node_collection = MagicMock()
        self.mock_edge_collection = MagicMock()
        self.mock_db.collection.side_effect = lambda name: (
            self.mock_node_collection if name == "nodes" else self.mock_edge_collection
        )
        
        # Create the store with mock connection
        self.store = ArangoGraphStore(
            host="localhost",
            port=8529,
            username="user",
            password="pass",
            database="test_db"
        )
    
    def tearDown(self) -> None:
        """Tear down test fixtures."""
        self.connection_patcher.stop()

    def test_initialization(self) -> None:
        """Test that the store initializes correctly."""
        self.assertIsNotNone(self.store)
        self.mock_connection_class.assert_called_once_with(
            host="localhost",
            port=8529,
            username="user",
            password="pass",
            database="test_db"
        )
    
    def test_initialization_with_error(self) -> None:
        """Test handling of connection errors during initialization."""
        # Arrange
        self.mock_connection_class.side_effect = Exception("Connection failed")
        
        # Act & Assert
        with self.assertRaises(ArangoConnectionError):
            ArangoGraphStore(
                host="localhost",
                port=8529,
                username="user",
                password="pass",
                database="test_db"
            )
    
    def test_setup_collection_creates_if_not_exists(self) -> None:
        """Test collection setup creates collections if they don't exist."""
        # Arrange
        self.mock_db.has_collection.return_value = False
        
        # Act
        self.store.setup_collections()
        
        # Assert
        self.mock_db.create_collection.assert_any_call("nodes")
        self.mock_db.create_collection.assert_any_call("edges", edge=True)
    
    def test_setup_collection_skips_if_exists(self) -> None:
        """Test collection setup skips existing collections."""
        # Arrange
        self.mock_db.has_collection.return_value = True
        
        # Act
        self.store.setup_collections()
        
        # Assert
        self.mock_db.create_collection.assert_not_called()
    
    def test_create_graph_success(self) -> None:
        """Test successful graph creation."""
        # Arrange
        self.mock_db.has_graph.return_value = False
        
        # Act
        self.store.create_graph("test_graph")
        
        # Assert
        self.mock_db.create_graph.assert_called_once()
    
    def test_create_graph_skips_if_exists(self) -> None:
        """Test graph creation skips if graph already exists."""
        # Arrange
        self.mock_db.has_graph.return_value = True
        
        # Act
        self.store.create_graph("test_graph")
        
        # Assert
        self.mock_db.create_graph.assert_not_called()
    
    def test_store_node_success(self) -> None:
        """Test successful node storage."""
        # Arrange
        node = Node(
            node_id="test_node",
            node_type="test",
            properties={"name": "Test Node"}
        )
        self.mock_node_collection.has.return_value = False
        
        # Act
        result = self.store.store_node(node)
        
        # Assert
        self.assertTrue(result)
        self.mock_node_collection.insert.assert_called_once()
    
    def test_store_node_update_if_exists(self) -> None:
        """Test node update if it already exists."""
        # Arrange
        node = Node(
            node_id="test_node",
            node_type="test",
            properties={"name": "Test Node"}
        )
        self.mock_node_collection.has.return_value = True
        
        # Act
        result = self.store.store_node(node)
        
        # Assert
        self.assertTrue(result)
        self.mock_node_collection.update.assert_called_once()
    
    def test_store_node_handles_error(self) -> None:
        """Test error handling during node storage."""
        # Arrange
        node = Node(
            node_id="test_node",
            node_type="test",
            properties={"name": "Test Node"}
        )
        self.mock_node_collection.insert.side_effect = Exception("Database error")
        
        # Act
        result = self.store.store_node(node)
        
        # Assert
        self.assertFalse(result)
    
    def test_store_edge_success(self) -> None:
        """Test successful edge storage."""
        # Arrange
        edge = Edge(
            edge_id="test_edge",
            from_node="node1",
            to_node="node2",
            edge_type="CONTAINS",
            properties={"weight": 0.8}
        )
        self.mock_edge_collection.has.return_value = False
        
        # Act
        result = self.store.store_edge(edge)
        
        # Assert
        self.assertTrue(result)
        self.mock_edge_collection.insert.assert_called_once()
    
    def test_store_edge_update_if_exists(self) -> None:
        """Test edge update if it already exists."""
        # Arrange
        edge = Edge(
            edge_id="test_edge",
            from_node="node1",
            to_node="node2",
            edge_type="CONTAINS",
            properties={"weight": 0.8}
        )
        self.mock_edge_collection.has.return_value = True
        
        # Act
        result = self.store.store_edge(edge)
        
        # Assert
        self.assertTrue(result)
        self.mock_edge_collection.update.assert_called_once()
    
    def test_store_edge_handles_error(self) -> None:
        """Test error handling during edge storage."""
        # Arrange
        edge = Edge(
            edge_id="test_edge",
            from_node="node1",
            to_node="node2",
            edge_type="CONTAINS",
            properties={"weight": 0.8}
        )
        self.mock_edge_collection.insert.side_effect = Exception("Database error")
        
        # Act
        result = self.store.store_edge(edge)
        
        # Assert
        self.assertFalse(result)
    
    def test_get_node_by_id_success(self) -> None:
        """Test successful node retrieval by ID."""
        # Arrange
        node_data = {
            "_id": "test_node",
            "node_type": "test",
            "name": "Test Node"
        }
        self.mock_node_collection.get.return_value = node_data
        
        # Act
        node = self.store.get_node_by_id("test_node")
        
        # Assert
        self.assertIsNotNone(node)
        self.assertEqual(node.node_id, "test_node")
        self.assertEqual(node.node_type, "test")
        self.assertEqual(node.properties["name"], "Test Node")
    
    def test_get_node_by_id_not_found(self) -> None:
        """Test node retrieval when node doesn't exist."""
        # Arrange
        self.mock_node_collection.get.side_effect = Exception("Document not found")
        
        # Act
        node = self.store.get_node_by_id("nonexistent")
        
        # Assert
        self.assertIsNone(node)
    
    def test_get_edge_by_id_success(self) -> None:
        """Test successful edge retrieval by ID."""
        # Arrange
        edge_data = {
            "_id": "test_edge",
            "_from": "node1",
            "_to": "node2",
            "edge_type": "CONTAINS",
            "weight": 0.8
        }
        self.mock_edge_collection.get.return_value = edge_data
        
        # Act
        edge = self.store.get_edge_by_id("test_edge")
        
        # Assert
        self.assertIsNotNone(edge)
        self.assertEqual(edge.edge_id, "test_edge")
        self.assertEqual(edge.from_node, "node1")
        self.assertEqual(edge.to_node, "node2")
        self.assertEqual(edge.edge_type, "CONTAINS")
        self.assertEqual(edge.properties["weight"], 0.8)
    
    def test_get_edge_by_id_not_found(self) -> None:
        """Test edge retrieval when edge doesn't exist."""
        # Arrange
        self.mock_edge_collection.get.side_effect = Exception("Document not found")
        
        # Act
        edge = self.store.get_edge_by_id("nonexistent")
        
        # Assert
        self.assertIsNone(edge)
    
    def test_query_nodes_success(self) -> None:
        """Test successful node query."""
        # Arrange
        query_result = [
            {
                "_id": "node1",
                "node_type": "test",
                "name": "Node 1"
            },
            {
                "_id": "node2",
                "node_type": "test",
                "name": "Node 2"
            }
        ]
        
        self.mock_db.aql.execute.return_value = query_result
        
        # Act
        nodes = self.store.query_nodes(node_type="test", limit=10)
        
        # Assert
        self.assertEqual(len(nodes), 2)
        self.assertEqual(nodes[0].node_id, "node1")
        self.assertEqual(nodes[1].node_id, "node2")
        self.mock_db.aql.execute.assert_called_once()
    
    def test_query_nodes_empty_result(self) -> None:
        """Test node query with no matching results."""
        # Arrange
        self.mock_db.aql.execute.return_value = []
        
        # Act
        nodes = self.store.query_nodes(node_type="nonexistent", limit=10)
        
        # Assert
        self.assertEqual(len(nodes), 0)
    
    def test_query_nodes_handles_error(self) -> None:
        """Test error handling during node query."""
        # Arrange
        self.mock_db.aql.execute.side_effect = Exception("Query error")
        
        # Act & Assert
        with self.assertRaises(Exception):
            self.store.query_nodes(node_type="test", limit=10)
    
    def test_query_edges_success(self) -> None:
        """Test successful edge query."""
        # Arrange
        query_result = [
            {
                "_id": "edge1",
                "_from": "node1",
                "_to": "node2",
                "edge_type": "CONTAINS",
                "weight": 0.8
            },
            {
                "_id": "edge2",
                "_from": "node2",
                "_to": "node3",
                "edge_type": "CALLS",
                "weight": 0.9
            }
        ]
        
        self.mock_db.aql.execute.return_value = query_result
        
        # Act
        edges = self.store.query_edges(edge_type="CONTAINS", limit=10)
        
        # Assert
        self.assertEqual(len(edges), 2)
        self.assertEqual(edges[0].edge_id, "edge1")
        self.assertEqual(edges[1].edge_id, "edge2")
        self.mock_db.aql.execute.assert_called_once()
    
    def test_find_paths_success(self) -> None:
        """Test successful path finding between nodes."""
        # Arrange
        path_result = [
            {
                "vertices": [
                    {"_id": "node1", "node_type": "test", "name": "Node 1"},
                    {"_id": "node2", "node_type": "test", "name": "Node 2"},
                    {"_id": "node3", "node_type": "test", "name": "Node 3"}
                ],
                "edges": [
                    {"_id": "edge1", "_from": "node1", "_to": "node2", "edge_type": "CONTAINS"},
                    {"_id": "edge2", "_from": "node2", "_to": "node3", "edge_type": "CALLS"}
                ],
                "path_score": 0.85
            }
        ]
        
        self.mock_db.aql.execute.return_value = path_result
        
        # Act
        paths = self.store.find_paths(
            start_node_id="node1",
            end_node_id="node3",
            max_depth=3
        )
        
        # Assert
        self.assertEqual(len(paths), 1)
        self.assertEqual(len(paths[0]["vertices"]), 3)
        self.assertEqual(len(paths[0]["edges"]), 2)
        self.assertEqual(paths[0]["path_score"], 0.85)
    
    def test_find_paths_no_results(self) -> None:
        """Test path finding when no paths exist."""
        # Arrange
        self.mock_db.aql.execute.return_value = []
        
        # Act
        paths = self.store.find_paths(
            start_node_id="node1",
            end_node_id="nonexistent",
            max_depth=3
        )
        
        # Assert
        self.assertEqual(len(paths), 0)
    
    def test_semantic_search_success(self) -> None:
        """Test successful semantic search."""
        # Arrange
        search_result = [
            {
                "_id": "node1",
                "node_type": "document",
                "content": "This is relevant content",
                "score": 0.95
            },
            {
                "_id": "node2",
                "node_type": "document",
                "content": "This is also relevant",
                "score": 0.85
            }
        ]
        
        self.mock_db.aql.execute.return_value = search_result
        
        # Act
        results = self.store.semantic_search(
            query_text="relevant content",
            node_type="document",
            limit=10
        )
        
        # Assert
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["_id"], "node1")
        self.assertEqual(results[0]["score"], 0.95)
        self.assertEqual(results[1]["_id"], "node2")
        self.assertEqual(results[1]["score"], 0.85)
    
    def test_delete_node_success(self) -> None:
        """Test successful node deletion."""
        # Arrange
        self.mock_node_collection.delete.return_value = True
        
        # Act
        result = self.store.delete_node("test_node")
        
        # Assert
        self.assertTrue(result)
        self.mock_node_collection.delete.assert_called_once_with("test_node")
    
    def test_delete_node_handles_error(self) -> None:
        """Test error handling during node deletion."""
        # Arrange
        self.mock_node_collection.delete.side_effect = Exception("Delete error")
        
        # Act
        result = self.store.delete_node("test_node")
        
        # Assert
        self.assertFalse(result)
    
    def test_delete_edge_success(self) -> None:
        """Test successful edge deletion."""
        # Arrange
        self.mock_edge_collection.delete.return_value = True
        
        # Act
        result = self.store.delete_edge("test_edge")
        
        # Assert
        self.assertTrue(result)
        self.mock_edge_collection.delete.assert_called_once_with("test_edge")
    
    def test_delete_edge_handles_error(self) -> None:
        """Test error handling during edge deletion."""
        # Arrange
        self.mock_edge_collection.delete.side_effect = Exception("Delete error")
        
        # Act
        result = self.store.delete_edge("test_edge")
        
        # Assert
        self.assertFalse(result)


# Add pytest marker for categorization
pytestmark = pytest.mark.storage

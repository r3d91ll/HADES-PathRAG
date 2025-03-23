"""
Tests for ArangoDB adapter integration with PathRAG.

These tests verify the functionality of the ArangoDB adapter for XnX PathRAG,
including node storage, edge creation, and path traversal.
"""

import os
import unittest
import logging
from typing import List, Dict, Any, Optional
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set test environment variables if not already set
if "HADES_ARANGO_DATABASE" not in os.environ:
    os.environ["HADES_ARANGO_DATABASE"] = "pathrag_test"

# Import modules after environment setup
from src.db.arango_connection import ArangoConnection
from src.xnx.arango_adapter import ArangoPathRAGAdapter


class TestArangoAdapter(unittest.TestCase):
    """Test suite for ArangoDB adapter for PathRAG."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test database and collections."""
        cls.db_name = os.environ.get("HADES_ARANGO_DATABASE", "pathrag_test")
        
        # Connect to ArangoDB
        cls.conn = ArangoConnection(db_name=cls.db_name)
        
        # Clean up existing collections if any
        for collection in ["test_nodes", "test_edges"]:
            if cls.conn.collection_exists(collection):
                cls.conn.delete_collection(collection)
                
        # Create test adapter
        cls.adapter = ArangoPathRAGAdapter(
            arango_connection=cls.conn,
            db_name=cls.db_name,
            nodes_collection="test_nodes",
            edges_collection="test_edges",
            graph_name="test_graph"
        )
        
    def generate_embedding(self, dim=32):
        """Generate a random embedding for testing."""
        return list(np.random.rand(dim))
    
    def test_01_store_node(self):
        """Test storing a node with embedding."""
        # Create test node
        node_id = self.adapter.store_node(
            node_id="test1",
            content="Test node one",
            embedding=self.generate_embedding(),
            metadata={"domain": "test", "source": "unittest"}
        )
        
        # Verify node was stored
        self.assertEqual(node_id, "test1")
        
        # Create another node
        node_id = self.adapter.store_node(
            node_id="test2",
            content="Test node two",
            embedding=self.generate_embedding(),
            metadata={"domain": "code", "source": "unittest"}
        )
        
        self.assertEqual(node_id, "test2")
    
    def test_02_create_edge(self):
        """Test creating an edge between nodes."""
        # Create edge
        edge_id = self.adapter.create_edge(
            from_node="test1",
            to_node="test2",
            weight=0.75,
            metadata={"relation": "test_relation"}
        )
        
        # Verify edge was created
        self.assertIsNotNone(edge_id)
        
        # Create another node for testing more edges
        self.adapter.store_node(
            node_id="test3",
            content="Test node three",
            embedding=self.generate_embedding(),
            metadata={"domain": "ai", "source": "unittest"}
        )
        
        # Create more edges to form a path
        self.adapter.create_edge(
            from_node="test2",
            to_node="test3",
            weight=0.85,
            metadata={"relation": "next_relation"}
        )
    
    def test_03_get_paths(self):
        """Test retrieving paths from nodes."""
        # Get paths from test1
        paths = self.adapter.get_paths_from_node("test1", max_depth=2)
        
        # Verify paths were found
        self.assertGreaterEqual(len(paths), 1)
        
        # Check first path structure
        self.assertIn("nodes", paths[0])
        self.assertIn("edges", paths[0])
        self.assertIn("total_weight", paths[0])
        
        # Verify path contains correct nodes
        node_keys = [node.get("_key", "") for node in paths[0]["nodes"]]
        self.assertIn("test1", node_keys)
        self.assertIn("test2", node_keys)
    
    def test_04_weighted_paths(self):
        """Test weighted path traversal with XnX query."""
        # Get weighted paths
        weighted_paths = self.adapter.get_weighted_paths(
            node_id="test1",
            xnx_query="X(domain='code')2",
            max_depth=3
        )
        
        # Verify weighted paths were found
        self.assertGreaterEqual(len(weighted_paths), 1)
        
        # Check weighted path structure
        self.assertIn("nodes", weighted_paths[0])
        self.assertIn("edges", weighted_paths[0])
        self.assertIn("base_score", weighted_paths[0])
        self.assertIn("xnx_score", weighted_paths[0])
        
        # Verify XnX score is greater than base score for code domain
        for path in weighted_paths:
            for node in path["nodes"]:
                if node.get("_key") == "test2":  # This is our code domain node
                    self.assertGreaterEqual(path["xnx_score"], path["base_score"])
    
    def test_05_find_similar_nodes(self):
        """Test finding nodes similar to a query embedding."""
        # Create a query embedding similar to test1's embedding
        query_embedding = self.generate_embedding()
        
        # Find similar nodes
        similar_nodes = self.adapter.find_similar_nodes(
            query_embedding=query_embedding,
            top_k=3
        )
        
        # Verify similar nodes were found
        self.assertGreaterEqual(len(similar_nodes), 1)
        
        # Check similar node structure
        self.assertIn("similarity", similar_nodes[0])
        self.assertIn("_key", similar_nodes[0])
        
    @classmethod
    def tearDownClass(cls):
        """Clean up test collections."""
        # Clean up collections
        for collection in ["test_nodes", "test_edges"]:
            if cls.conn.collection_exists(collection):
                cls.conn.delete_collection(collection)
        
        # Clean up graph
        if cls.conn.graph_exists("test_graph"):
            cls.conn.delete_graph("test_graph")


if __name__ == "__main__":
    unittest.main()

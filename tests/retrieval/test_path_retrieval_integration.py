"""
Integration tests for the path-based retrieval module with ArangoDB.

These tests verify that the path retrieval module works correctly with
an actual ArangoDB instance, testing the full retrieval pipeline.
"""
import sys
import os
import unittest
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from hades_pathrag.storage.arango import ArangoDBConnection, ArangoGraphStorage
from hades_pathrag.storage.arango_enhanced import EnhancedArangoGraphStorage
from hades_pathrag.storage.edge_types import create_edge_data, EDGE_TYPES
from hades_pathrag.retrieval.path_retrieval import (
    PathRankingConfig,
    RetrievalResult,
    path_based_retrieval,
    retrieve_related_nodes
)


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestPathRetrievalIntegration(unittest.TestCase):
    """Integration tests for path-based retrieval with ArangoDB."""
    
    @classmethod
    def setUpClass(cls):
        """Set up an ArangoDB connection and graph for testing."""
        # Check if we should skip tests if ArangoDB is not available
        cls.skip_tests = False
        
        try:
            # Configure ArangoDB connection
            cls.connection = ArangoDBConnection(
                host=os.environ.get("ARANGO_HOST", "localhost"),
                port=int(os.environ.get("ARANGO_PORT", "8529")),
                username=os.environ.get("ARANGO_USERNAME", "root"),
                password=os.environ.get("ARANGO_PASSWORD", "root"),
                database=os.environ.get("ARANGO_DATABASE", "pathrag_test")
            )
            
            try:
                # Try to connect to the database
                cls.connection.connect()
                
                # Create a test graph
                cls.graph_storage = EnhancedArangoGraphStorage(
                    connection=cls.connection,
                    graph_name="test_pathrag",
                    node_collection_name="test_nodes",
                    edge_collection_name="test_edges"
                )
                
                # Initialize the graph
                cls.graph_storage.initialize(create_collections=True)
                
                # Create test data
                cls._create_test_graph(cls.graph_storage)
            except Exception as inner_e:
                logger.error(f"Error connecting to ArangoDB: {inner_e}")
                cls.skip_tests = True
            
        except Exception as e:
            logger.error(f"Error setting up ArangoDB for tests: {e}")
            cls.skip_tests = True
    
    @classmethod
    def tearDownClass(cls):
        """Clean up after tests."""
        if not cls.skip_tests:
            try:
                # Delete the test graph and collections
                if cls.graph_storage.graph is not None:
                    cls.graph_storage.graph.delete()
                
                # Delete the test database
                if cls.connection.client is not None:
                    sys_db = cls.connection.client.db("_system")
                    if sys_db.has_database(cls.connection.database):
                        sys_db.delete_database(cls.connection.database)
                
            except Exception as e:
                logger.error(f"Error cleaning up after tests: {e}")
    
    @classmethod
    def _create_test_graph(cls, graph_storage):
        """Create a test graph with nodes and edges for testing."""
        # Create some test nodes with embeddings
        nodes = [
            {
                "_key": "code1",
                "name": "Code Node 1",
                "type": "function",
                "content": "def hello_world(): print('Hello, World!')",
                "embedding": np.random.random(5).tolist()
            },
            {
                "_key": "code2",
                "name": "Code Node 2",
                "type": "function",
                "content": "def calculate_sum(a, b): return a + b",
                "embedding": np.random.random(5).tolist()
            },
            {
                "_key": "code3",
                "name": "Code Node 3",
                "type": "class",
                "content": "class Calculator: pass",
                "embedding": np.random.random(5).tolist()
            },
            {
                "_key": "doc1",
                "name": "Doc Node 1",
                "type": "documentation",
                "content": "API documentation for hello_world",
                "embedding": np.random.random(5).tolist()
            },
            {
                "_key": "doc2",
                "name": "Doc Node 2",
                "type": "documentation",
                "content": "API documentation for Calculator",
                "embedding": np.random.random(5).tolist()
            }
        ]
        
        # Store the nodes
        for node in nodes:
            graph_storage.store_node(node["_key"], node)
        
        # Create edges between nodes with different types
        edges = [
            # Primary relationship (high weight)
            ("code3", "code2", "CALLS"),
            
            # Secondary relationship (medium weight)
            ("code1", "code2", "REFERENCES"),
            
            # Tertiary relationship (lower weight)
            ("code1", "doc1", "DOCUMENTED_BY"),
            ("code3", "doc2", "DOCUMENTED_BY"),
            
            # Additional relationships for path testing
            ("code2", "doc2", "RELATED_TO"),
            ("doc1", "doc2", "SIMILAR_TO")
        ]
        
        # Store the edges
        for source, target, edge_type in edges:
            graph_storage.store_typed_edge(source, target, edge_type)
    
    def setUp(self):
        """Set up for each test."""
        if self.skip_tests:
            self.skipTest("ArangoDB is not available")
    
    def test_retrieve_related_nodes(self):
        """Test retrieving nodes related to a specific node."""
        # Get related nodes for code1
        related = retrieve_related_nodes(
            self.connection.db,
            "code1",
            max_depth=2,
            min_weight=0.2,
            limit=10
        )
        
        # Check that we found at least the direct connections
        self.assertGreaterEqual(len(related), 2)  # code1 -> code2, code1 -> doc1
        
        # Check that we have the expected structure
        for node in related:
            self.assertIn("node", node)
            self.assertIn("edge", node)
            self.assertIn("path_weight", node)
            self.assertIn("path_length", node)
            
            # Check that path weights are in valid range
            self.assertGreaterEqual(node["path_weight"], 0.0)
            self.assertLessEqual(node["path_weight"], 1.0)
    
    def test_path_based_retrieval(self):
        """Test the path-based retrieval with semantic and structural scoring."""
        # Create a query embedding
        query_embedding = np.random.random(5)
        
        # Get embeddings and create initial nodes (simulating semantic search)
        nodes = [
            ("code1", self.graph_storage.get_node("code1"), 0.85),
            ("doc1", self.graph_storage.get_node("doc1"), 0.75)
        ]
        
        # Perform path-based retrieval
        config = PathRankingConfig(
            semantic_weight=0.7,
            structural_weight=0.3,
            max_path_length=3,
            min_edge_weight=0.2,
            max_results=10
        )
        
        results = path_based_retrieval(
            self.connection.db,
            query_embedding,
            nodes,
            config
        )
        
        # Check that we got results (should include both initial nodes + expanded nodes)
        self.assertGreaterEqual(len(results), 2)
        
        # Check that the results have the expected structure
        for result in results:
            self.assertIsInstance(result, RetrievalResult)
            self.assertIsNotNone(result.node_id)
            self.assertIsNotNone(result.data)
            self.assertIsNotNone(result.score)
            
            # Check that scores are in valid range
            self.assertGreaterEqual(result.score, 0.0)
            self.assertLessEqual(result.score, 1.0)
    
    def test_expand_paths_from_nodes(self):
        """Test expanding paths from multiple starting nodes."""
        # Use the enhanced graph storage to expand paths
        paths = self.graph_storage.expand_paths(
            start_nodes=["code1", "code3"],
            max_depth=2,
            min_weight=0.2,
            limit_per_node=5,
            total_limit=10
        )
        
        # Check that we got paths
        self.assertGreater(len(paths), 0)
        
        # Check that paths have the expected structure
        for path in paths:
            self.assertIn("nodes", path)
            self.assertIn("edges", path)
            self.assertIn("total_weight", path)
            self.assertIn("avg_weight", path)
            self.assertIn("length", path)
            self.assertIn("score", path)
            
            # Check that metrics are calculated correctly
            if len(path["edges"]) > 0:
                # Weights should be a valid average
                self.assertGreaterEqual(path["avg_weight"], 0.0)
                self.assertLessEqual(path["avg_weight"], 1.0)
                
                # Length should match number of edges
                self.assertEqual(path["length"], len(path["edges"]))
    
    def test_find_paths_between_nodes(self):
        """Test finding paths between specific nodes."""
        # Use the enhanced graph storage to find paths
        paths = self.graph_storage.find_paths(
            start_node="code3",
            end_node="doc2",
            max_depth=3,
            min_weight=0.2,
            limit=5
        )
        
        # Check that we found at least one path (code3 -> doc2)
        self.assertGreater(len(paths), 0)
        
        # Check the structure of the paths
        for path in paths:
            # Check that the path starts and ends at the correct nodes
            nodes = path["nodes"]
            self.assertEqual(nodes[0]["_id"], "code3")
            self.assertEqual(nodes[-1]["_id"], "doc2")
            
            # Check that scores are calculated
            self.assertIn("score", path)
            self.assertGreaterEqual(path["score"], 0.0)
            self.assertLessEqual(path["score"], 1.0)


if __name__ == '__main__':
    unittest.main()

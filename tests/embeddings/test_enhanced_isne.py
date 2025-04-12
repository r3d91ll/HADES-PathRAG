"""
Tests for the Enhanced ISNE embedder implementation.

This demonstrates the core capabilities of the EnhancedISNEEmbedder,
particularly its inductive learning capabilities for new nodes.
"""
import os
import tempfile
import unittest
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import networkx as nx

from hades_pathrag.embeddings.enhanced_isne import EnhancedISNEEmbedder


class TestEnhancedISNE(unittest.TestCase):
    """Test suite for the Enhanced ISNE embedder."""
    
    def setUp(self) -> None:
        """Set up test fixtures."""
        # Create a simple test graph
        self.graph = nx.Graph()
        
        # Add nodes with text attributes
        node_data = [
            ("A", "This is node A content"),
            ("B", "Node B has different content"),
            ("C", "Node C has some information"),
            ("D", "This is node D with unique text"),
            ("E", "Node E contains special data"),
        ]
        
        for node_id, text in node_data:
            self.graph.add_node(node_id, text=text)
        
        # Add edges to create a connected graph
        edges = [
            ("A", "B"), ("A", "C"), 
            ("B", "D"), ("C", "D"),
            ("D", "E")
        ]
        
        for source, target in edges:
            self.graph.add_edge(source, target)
        
        # Create embedder with small dimension for fast tests
        self.embedder = EnhancedISNEEmbedder(
            embedding_dim=16,
            epochs=2,  # Few epochs for faster tests
            batch_size=2,
            negative_samples=2,
            text_model_name="all-MiniLM-L6-v2"
        )
    
    def test_basic_training(self) -> None:
        """Test that training works correctly."""
        # Train the model
        self.embedder.fit(self.graph)
        
        # Verify all nodes have embeddings
        for node in self.graph.nodes():
            # Get node embedding
            embedding = self.embedder.encode(node)
            
            # Check shape and normalization
            self.assertEqual(embedding.shape, (16,), f"Wrong shape for node {node}")
            self.assertAlmostEqual(
                np.linalg.norm(embedding), 
                1.0, 
                places=5, 
                msg=f"Embedding for node {node} is not normalized"
            )
    
    def test_inductive_embedding(self) -> None:
        """Test that inductive embedding works for new nodes."""
        # Train the model on a subset of nodes
        train_graph = self.graph.subgraph(["A", "B", "C", "D"])
        self.embedder.fit(train_graph)
        
        # Test inductive embedding for node E (not in training set)
        neighbors = list(self.graph.neighbors("E"))
        text = self.graph.nodes["E"]["text"]
        
        # Get inductive embedding
        embedding = self.embedder.inductive_embedding("E", neighbors, text)
        
        # Check shape and normalization
        self.assertEqual(embedding.shape, (16,))
        self.assertAlmostEqual(np.linalg.norm(embedding), 1.0, places=5)
    
    def test_incremental_training(self) -> None:
        """Test that incremental training works correctly."""
        # Train on a subset first
        train_graph = self.graph.subgraph(["A", "B", "C"])
        self.embedder.fit(train_graph)
        
        # Get embeddings for initial nodes
        initial_embeddings = {
            node: self.embedder.encode(node) 
            for node in ["A", "B", "C"]
        }
        
        # Now add new nodes incrementally
        self.embedder.partial_fit(self.graph, ["D", "E"])
        
        # Verify original embeddings haven't changed significantly
        for node in ["A", "B", "C"]:
            original = initial_embeddings[node]
            updated = self.embedder.encode(node)
            # Embeddings should be very similar after partial fit
            similarity = np.dot(original, updated)
            self.assertGreater(similarity, 0.95, 
                              f"Original embedding for {node} changed too much after partial_fit")
        
        # Verify new nodes have valid embeddings
        for node in ["D", "E"]:
            embedding = self.embedder.encode(node)
            self.assertEqual(embedding.shape, (16,))
            self.assertAlmostEqual(np.linalg.norm(embedding), 1.0, places=5)
    
    def test_save_load(self) -> None:
        """Test saving and loading the model."""
        # Train the model
        self.embedder.fit(self.graph)
        
        # Get embeddings before saving
        original_embeddings = {
            node: self.embedder.encode(node) 
            for node in self.graph.nodes()
        }
        
        # Save the model to a temporary file
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp:
            model_path = tmp.name
        
        try:
            self.embedder.save(model_path)
            
            # Load the model
            loaded_embedder = EnhancedISNEEmbedder.load(model_path)
            
            # Compare embeddings
            for node in self.graph.nodes():
                original = original_embeddings[node]
                loaded = loaded_embedder.encode(node)
                
                # Embeddings should be identical after load
                np.testing.assert_array_almost_equal(
                    original, loaded, decimal=5,
                    err_msg=f"Embedding mismatch for node {node} after save/load"
                )
            
            # Test inductive capability of loaded model
            new_node = "F"
            neighbors = ["D", "E"]  # Connect to existing nodes
            text = "This is a completely new node F"
            
            # Get inductive embedding
            embedding = loaded_embedder.inductive_embedding(new_node, neighbors, text)
            
            # Check shape and normalization
            self.assertEqual(embedding.shape, (16,))
            self.assertAlmostEqual(np.linalg.norm(embedding), 1.0, places=5)
            
        finally:
            # Clean up the temporary file
            os.unlink(model_path)
    
    def test_batch_encode(self) -> None:
        """Test batch encoding of multiple nodes."""
        # Train the model
        self.embedder.fit(self.graph)
        
        # Prepare batch input
        batch_input = [
            # Existing node with text
            ("A", list(self.graph.neighbors("A")), self.graph.nodes["A"]["text"]),
            # New node with neighbors but no text
            ("X", ["A", "B"], None),
            # New node with text but no valid neighbors
            ("Y", ["X", "Z"], "This is a new node Y"),
            # New node with neighbors and text
            ("Z", ["A", "D"], "Node Z has special properties")
        ]
        
        # Get batch embeddings
        embeddings = self.embedder.batch_encode(batch_input)
        
        # Verify results
        self.assertEqual(len(embeddings), 4, "Wrong number of embeddings returned")
        
        for i, embedding in enumerate(embeddings):
            # Check shape and normalization
            self.assertEqual(embedding.shape, (16,), f"Wrong shape for embedding {i}")
            self.assertAlmostEqual(
                np.linalg.norm(embedding), 
                1.0, 
                places=5, 
                msg=f"Embedding {i} is not normalized"
            )
    
    def test_embedding_comparison(self) -> None:
        """Test embedding comparison functionality."""
        # Train the model
        self.embedder.fit(self.graph)
        
        # Get embeddings for all nodes
        node_embeddings = {
            node: self.embedder.encode(node) 
            for node in self.graph.nodes()
        }
        
        # Compare node A with all others
        source_embedding = node_embeddings["A"]
        target_embeddings = [node_embeddings[n] for n in ["B", "C", "D", "E"]]
        
        # Get similarities
        similarities = self.embedder.compare(source_embedding, target_embeddings)
        
        # Check results
        self.assertEqual(len(similarities), 4, "Wrong number of similarity scores")
        
        # Neighbors of A should have higher similarity than non-neighbors
        neighbors = set(self.graph.neighbors("A"))
        neighbor_indices = [i for i, n in enumerate(["B", "C", "D", "E"]) if n in neighbors]
        non_neighbor_indices = [i for i, n in enumerate(["B", "C", "D", "E"]) if n not in neighbors]
        
        # Average similarity for neighbors should be higher than for non-neighbors
        avg_neighbor_sim = np.mean([similarities[i] for i in neighbor_indices])
        avg_non_neighbor_sim = np.mean([similarities[i] for i in non_neighbor_indices])
        
        self.assertGreater(avg_neighbor_sim, avg_non_neighbor_sim,
                          "Neighbors should have higher similarity than non-neighbors")
    
    def test_embedding_stats(self) -> None:
        """Test that embedding statistics are being tracked correctly."""
        # Train the model
        self.embedder.fit(self.graph)
        
        # Generate some embeddings and do comparisons to build stats
        embeddings = [self.embedder.encode(node) for node in self.graph.nodes()]
        self.embedder.compare(embeddings[0], embeddings[1:])
        
        # Get stats
        stats = self.embedder.get_stats()
        
        # Verify stats are properly populated
        self.assertEqual(stats.model_type, "Enhanced ISNE")
        self.assertEqual(stats.embedding_dim, 16)
        self.assertEqual(stats.num_nodes, 5)
        self.assertGreater(stats.metrics.training_time_seconds, 0)
        self.assertEqual(stats.metrics.num_comparisons, 4)
        self.assertGreater(stats.cache_stats["size"], 0)


if __name__ == "__main__":
    unittest.main()

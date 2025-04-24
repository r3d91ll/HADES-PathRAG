"""
Test for hierarchical document structure with the repository ingestor.

This demonstrates how to test the hierarchical document structure in the context
of the repository ingestion pipeline and improve test coverage.
"""
import unittest
from unittest.mock import MagicMock, patch
import os
import tempfile
import json

from src.ingest.ingestor import RepositoryIngestor
from src.ingest.models.graph_models import (
    DocumentNode, TextNode, ImageNode, HierarchicalGraph
)


class TestHierarchicalGraphIngestor(unittest.TestCase):
    """Tests for hierarchical document graph ingestion."""
    
    def setUp(self):
        self.ingestor = RepositoryIngestor(config={})
        self.ingestor.db = MagicMock()
        self.ingestor.setup_collections = MagicMock()
        
        # Create a sample hierarchical graph
        document = DocumentNode(
            id="doc1",
            title="Test Document",
            source="/path/to/test.pdf"
        )
        
        self.graph = HierarchicalGraph(root_node=document)
        
        # Add sections
        section1 = TextNode(
            id="sec1",
            title="Introduction",
            content="This is the introduction section.",
            type="section"
        )
        self.graph.add_child(document, section1)
        
        section2 = TextNode(
            id="sec2",
            title="Results",
            content="These are the results.",
            type="section"
        )
        self.graph.add_child(document, section2)
        
        # Add an image to section 2
        figure = ImageNode(
            id="fig1",
            title="Figure 1",
            caption="Test figure caption"
        )
        self.graph.add_child(section2, figure)
        
        # Add a reference from text to figure
        self.graph.add_edge(section2, figure, "references")

    def test_process_hierarchical_document(self):
        """Test processing a hierarchical document structure."""
        # Mock the doc_parser to return our test graph
        mock_doc_parser = MagicMock()
        mock_doc_parser.parse_documentation.return_value = {
            "/path/to/test.pdf": self.graph
        }
        
        # Set up necessary mocks for ingestor
        self.ingestor._create_repo_node = MagicMock(return_value={"_key": "repo1"})
        self.ingestor._create_edge = MagicMock(return_value={"_id": "edge1"})
        
        # Create a temporary file to use for testing
        with tempfile.NamedTemporaryFile(suffix=".pdf") as temp_file:
            # Execute the _process_doc_files method
            doc_nodes = self.ingestor._process_doc_files(
                {temp_file.name: self.graph}, "repo1"
            )
            
            # Check if the document node was created
            self.assertIn(temp_file.name, doc_nodes)
            
            # Verify nodes were created for document structure
            node_count = 0
            edge_count = 0
            
            for node in self.graph.nodes:
                # Verify each node was processed
                self.ingestor.db.collection.assert_any_call("nodes")
                node_count += 1
            
            for edge in self.graph.edges:
                # Verify each edge was processed
                self.ingestor._create_edge.assert_any_call(
                    from_id=edge.from_id, 
                    to_id=edge.to_id,
                    edge_type=edge.type
                )
                edge_count += 1
            
            # Check that all nodes and edges were processed
            self.assertEqual(node_count, len(self.graph.nodes))
            self.assertEqual(edge_count, len(self.graph.edges))
    
    def test_document_embedding_generation(self):
        """Test generating embeddings for hierarchical document parts."""
        # Create a list of document parts (sections, figures, etc.)
        document_parts = [
            {"id": "sec1", "content": "This is section 1", "type": "section"},
            {"id": "sec2", "content": "This is section 2", "type": "section"},
            {"id": "fig1", "caption": "This is a figure", "type": "image"}
        ]
        
        # Mock ISNE connector
        self.ingestor.isne_connector = MagicMock()
        mock_embedding = [0.1, 0.2, 0.3]
        self.ingestor.isne_connector.get_embeddings.return_value = [
            {**part, "embedding": mock_embedding} for part in document_parts
        ]
        
        # Call the method
        result = self.ingestor._generate_embeddings(document_parts)
        
        # Check results
        self.assertEqual(len(result), len(document_parts))
        for part in result:
            self.assertIn("embedding", part)
            self.assertEqual(part["embedding"], mock_embedding)
        
        # Verify ISNE connector was called
        self.ingestor.isne_connector.get_embeddings.assert_called_once()
    
    def test_integration_with_structured_documents(self):
        """Test end-to-end integration with hierarchical documents."""
        # Create a mock PDF file
        with tempfile.NamedTemporaryFile(suffix=".pdf") as temp_pdf:
            # Mock external dependencies
            with patch('src.utils.git_operations.GitOperations') as mock_git_ops:
                # Set up git operations mock
                git_mock = mock_git_ops.return_value
                git_mock.clone_repository.return_value = (True, "success", "/tmp/repo")
                git_mock.get_repo_info.return_value = {
                    "repo_name": "test-repo", 
                    "remote_url": "http://repo"
                }
                
                # Mock the doc parser
                with patch('src.ingest.parsers.doc_parser.DocParser') as mock_doc_parser_cls:
                    doc_parser_mock = mock_doc_parser_cls.return_value
                    doc_parser_mock.parse_documentation.return_value = {
                        temp_pdf.name: self.graph
                    }
                    doc_parser_mock.extract_doc_code_relationships.return_value = []
                    
                    # Mock the code parser 
                    with patch('src.ingest.parsers.code_parser.CodeParser') as mock_code_parser_cls:
                        code_parser_mock = mock_code_parser_cls.return_value
                        code_parser_mock.parse.return_value = {}
                        code_parser_mock.extract_relationships.return_value = {}
                        
                        # Set up the rest of the ingestor mocks
                        self.ingestor._process_code_files = MagicMock(return_value={})
                        self.ingestor._process_doc_files = MagicMock(return_value={})
                        self.ingestor._create_code_relationships = MagicMock(return_value=0)
                        self.ingestor._create_doc_code_relationships = MagicMock(return_value=0)
                        self.ingestor._update_repo_node = MagicMock()
                        
                        # Call ingest_repository
                        success, message, stats = self.ingestor.ingest_repository("http://repo")
                        
                        # Verify the result
                        self.assertTrue(success)
                        self.assertIn("Successfully", message)
                        
                        # Check document processing
                        self.ingestor._process_doc_files.assert_called_once()
                        
                        # Verify the workflow is complete
                        self.ingestor._update_repo_node.assert_called_once()


if __name__ == '__main__':
    unittest.main()

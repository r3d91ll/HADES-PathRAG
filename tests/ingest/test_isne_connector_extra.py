import unittest
from unittest.mock import MagicMock, patch
import pytest
import os
from src.ingest.isne_connector import ISNEIngestorConnector

class TestISNEIngestorConnectorEdgeCases(unittest.TestCase):
    def setUp(self):
        self.mock_loader = MagicMock()
        self.mock_storage = MagicMock()
        self.connector = ISNEIngestorConnector()
        self.connector.isne_pipeline = MagicMock()
        self.connector.arango_adapter = MagicMock()

    def test_process_repository_success(self):
        # Simulate successful processing
        mock_pipeline = MagicMock()
        mock_pipeline.dataset = MagicMock()
        mock_pipeline.process.return_value = {'success': True}
        self.connector.isne_pipeline = mock_pipeline
        self.connector.arango_adapter = MagicMock()
        repo_path = "/tmp"
        os.makedirs(repo_path, exist_ok=True)
        result = self.connector.process_repository(repo_path, repo_name="testrepo", store_in_arango=True)
        self.assertIsNotNone(result)

    def test_process_repository_invalid_path(self):
        # Should fail for non-existent repo
        result = self.connector.process_repository("/no/such/path", repo_name="badrepo")
        self.assertIsNone(result)

    def test_process_repository_pipeline_error(self):
        # Pipeline throws error
        self.connector.isne_pipeline.process.side_effect = Exception("fail")
        repo_path = "/tmp"
        os.makedirs(repo_path, exist_ok=True)
        result = self.connector.process_repository(repo_path, repo_name="testrepo")
        self.assertIsNone(result)

    def test_get_document_embedding_success(self):
        # Should return embedding
        mock_embedding = [0.1, 0.2, 0.3]
        mock_result = MagicMock()
        mock_doc = MagicMock()
        mock_doc.embedding = mock_embedding
        mock_result.documents = [mock_doc]
        self.connector.isne_pipeline.embedding_processor = MagicMock()
        self.connector.isne_pipeline.embedding_processor.process.return_value = mock_result
        result = self.connector.get_document_embedding("some text")
        self.assertEqual(result, mock_embedding)

    def test_get_document_embedding_pipeline_missing(self):
        self.connector.isne_pipeline = None
        result = self.connector.get_document_embedding("text")
        self.assertIsNone(result)

    def test_get_document_embedding_error(self):
        self.connector.isne_pipeline.embedding_processor = MagicMock()
        self.connector.isne_pipeline.embedding_processor.process.side_effect = Exception("fail")
        result = self.connector.get_document_embedding("text")
        self.assertIsNone(result)


if __name__ == '__main__':
    unittest.main()

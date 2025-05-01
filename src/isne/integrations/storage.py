"""
Storage integration for ISNE.

This module provides storage adapters for the ISNE pipeline, allowing
documents, embeddings, and relationships to be persistently stored.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple, Union, cast
from datetime import datetime

from src.storage.arango.connection import ArangoConnection
from src.isne.types.models import (
    IngestDocument,
    IngestDataset,
    DocumentRelation,
    RelationType
)
from src.types.common import EmbeddingVector

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ArangoStorage:
    """
    ArangoDB storage adapter for ISNE documents and relationships.
    
    This class provides methods to store ISNE documents, embeddings,
    and relationships in ArangoDB.
    """
    
    # Default collection names
    DOCUMENTS_COLLECTION = "isne_documents"
    EMBEDDINGS_COLLECTION = "isne_embeddings"
    RELATIONS_COLLECTION = "isne_relations"
    DATASETS_COLLECTION = "isne_datasets"
    
    def __init__(
        self, 
        connection: ArangoConnection,
        documents_collection: str = DOCUMENTS_COLLECTION,
        embeddings_collection: str = EMBEDDINGS_COLLECTION,
        relations_collection: str = RELATIONS_COLLECTION,
        datasets_collection: str = DATASETS_COLLECTION
    ) -> None:
        """
        Initialize the ArangoDB storage adapter.
        
        Args:
            connection: ArangoDB connection
            documents_collection: Name of the documents collection
            embeddings_collection: Name of the embeddings collection
            relations_collection: Name of the relations collection
            datasets_collection: Name of the datasets collection
        """
        self.connection = connection
        self.documents_collection = documents_collection
        self.embeddings_collection = embeddings_collection
        self.relations_collection = relations_collection
        self.datasets_collection = datasets_collection
        
        # Ensure collections exist
        self._ensure_collections()
    
    def _ensure_collections(self) -> None:
        """Ensure all required collections exist."""
        try:
            # Create document collection if it doesn't exist
            if not self.connection.collection_exists(self.documents_collection):
                self.connection.create_collection(self.documents_collection)
                logger.info(f"Created collection {self.documents_collection}")
            
            # Create embeddings collection if it doesn't exist
            if not self.connection.collection_exists(self.embeddings_collection):
                self.connection.create_collection(self.embeddings_collection)
                logger.info(f"Created collection {self.embeddings_collection}")
            
            # Create relations collection if it doesn't exist
            if not self.connection.collection_exists(self.relations_collection):
                self.connection.create_edge_collection(self.relations_collection)
                logger.info(f"Created edge collection {self.relations_collection}")
            
            # Create datasets collection if it doesn't exist
            if not self.connection.collection_exists(self.datasets_collection):
                self.connection.create_collection(self.datasets_collection)
                logger.info(f"Created collection {self.datasets_collection}")
                
        except Exception as e:
            logger.error(f"Error ensuring collections: {e}")
            raise
    
    def store_document(self, document: IngestDocument) -> str:
        """
        Store a document in ArangoDB.
        
        Args:
            document: Document to store
            
        Returns:
            Document key
        """
        try:
            # Create document data
            doc_data = {
                "_key": self._normalize_key(document.id),
                "id": document.id,
                "content": document.content,
                "source": document.source,
                "document_type": document.document_type,
                "created_at": datetime.now().isoformat(),
            }
            
            # Add metadata if available
            if document.metadata:
                for key, value in document.metadata.items():
                    doc_data[key] = value
            
            # Store document
            result = self.connection.insert_document(
                self.documents_collection, 
                doc_data, 
                overwrite=True
            )
            
            # Store embedding if available
            if document.embedding is not None:
                self.store_embedding(document.id, document.embedding, document.embedding_model)
            
            logger.debug(f"Stored document {document.id}")
            return str(result["_key"])
            
        except Exception as e:
            logger.error(f"Error storing document {document.id}: {e}")
            raise
    
    def store_embedding(
        self, 
        document_id: str, 
        embedding: EmbeddingVector,
        model_name: Optional[str] = None
    ) -> str:
        """
        Store an embedding in ArangoDB.
        
        Args:
            document_id: ID of the document
            embedding: Embedding vector
            model_name: Name of the embedding model
            
        Returns:
            Embedding key
        """
        try:
            # Create embedding data
            embedding_data = {
                "_key": self._normalize_key(f"{document_id}_embedding"),
                "document_id": document_id,
                "embedding": embedding.tolist() if hasattr(embedding, "tolist") else embedding,
                "model": model_name,
                "created_at": datetime.now().isoformat(),
            }
            
            # Store embedding
            result = self.connection.insert_document(
                self.embeddings_collection, 
                embedding_data, 
                overwrite=True
            )
            
            logger.debug(f"Stored embedding for document {document_id}")
            return str(result["_key"])
            
        except Exception as e:
            logger.error(f"Error storing embedding for document {document_id}: {e}")
            raise
    
    def store_relation(self, relation: DocumentRelation) -> str:
        """
        Store a relation in ArangoDB.
        
        Args:
            relation: Relation to store
            
        Returns:
            Relation key
        """
        try:
            # Create relation data
            relation_data = {
                "_from": f"{self.documents_collection}/{self._normalize_key(relation.source_id)}",
                "_to": f"{self.documents_collection}/{self._normalize_key(relation.target_id)}",
                "source_id": relation.source_id,
                "target_id": relation.target_id,
                "type": relation.relation_type.value,
                "weight": relation.weight,
                "created_at": datetime.now().isoformat(),
            }
            
            # Add metadata if available
            if relation.metadata:
                for key, value in relation.metadata.items():
                    relation_data[key] = value
            
            # Store relation
            result = self.connection.insert_edge(
                self.relations_collection, 
                relation_data
            )
            
            logger.debug(f"Stored relation from {relation.source_id} to {relation.target_id}")
            return str(result["_key"])
            
        except Exception as e:
            logger.error(f"Error storing relation from {relation.source_id} to {relation.target_id}: {e}")
            raise
    
    def store_dataset(self, dataset: IngestDataset) -> str:
        """
        Store a dataset and all its documents and relations in ArangoDB.
        
        Args:
            dataset: Dataset to store
            
        Returns:
            Dataset key
        """
        try:
            # Create dataset data
            dataset_data = {
                "_key": self._normalize_key(dataset.name),
                "name": dataset.name,
                "created_at": datetime.now().isoformat(),
            }
            
            # Add metadata if available
            if dataset.metadata:
                for key, value in dataset.metadata.items():
                    dataset_data[key] = value
            
            # Store dataset
            result = self.connection.insert_document(
                self.datasets_collection, 
                dataset_data, 
                overwrite=True
            )
            
            # Store documents
            document_keys = []
            for document in dataset.documents:
                document.metadata = document.metadata or {}
                document.metadata["dataset"] = dataset.name
                document_key = self.store_document(document)
                document_keys.append(document_key)
            
            # Store relations
            relation_keys = []
            for relation in dataset.relations:
                relation_key = self.store_relation(relation)
                relation_keys.append(relation_key)
            
            logger.info(f"Stored dataset {dataset.name} with {len(document_keys)} documents and {len(relation_keys)} relations")
            return str(result["_key"])
            
        except Exception as e:
            logger.error(f"Error storing dataset {dataset.name}: {e}")
            raise
    
    @staticmethod
    def _normalize_key(key: str) -> str:
        """
        Normalize a key to be valid for ArangoDB.
        
        Args:
            key: Key to normalize
            
        Returns:
            Normalized key
        """
        # Replace characters that are not allowed in ArangoDB keys
        invalid_chars = ['/', '\\', '.', ' ', ':']
        result = key
        for char in invalid_chars:
            result = result.replace(char, '_')
        
        return result

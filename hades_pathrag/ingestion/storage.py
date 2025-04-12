"""
Storage components for the HADES-PathRAG ingestion pipeline.

This module provides classes for storing ingested data in ArangoDB,
ensuring that documents, embeddings, and relationships are properly
stored in the graph database.
"""
import logging
from typing import Dict, List, Optional, Set, Any, Union
import json

from hades_pathrag.storage.arango import ArangoDBConnection
from hades_pathrag.ingestion.models import IngestDataset, IngestDocument, DocumentRelation

logger = logging.getLogger(__name__)


class ArangoStorage:
    """
    Storage component for ArangoDB.
    
    This class handles storing documents, embeddings, and relationships
    in ArangoDB for use by the PathRAG system.
    """
    
    def __init__(
        self,
        connection: ArangoDBConnection,
        document_collection: str = "documents",
        edge_collection: str = "relationships",
        vector_collection: str = "vectors",
        graph_name: str = "knowledge_graph",
    ):
        """
        Initialize the ArangoDB storage component.
        
        Args:
            connection: ArangoDB connection
            document_collection: Name of the document collection
            edge_collection: Name of the edge collection
            vector_collection: Name of the vector collection for embeddings
            graph_name: Name of the graph
        """
        self.connection = connection
        self.document_collection = document_collection
        self.edge_collection = edge_collection
        self.vector_collection = vector_collection
        self.graph_name = graph_name
        
        # Ensure collections exist
        self._setup_collections()
    
    def _setup_collections(self):
        """
        Ensure that all required collections exist.
        """
        db = self.connection.db
        
        # Create document collection if it doesn't exist
        if not db.has_collection(self.document_collection):
            logger.info(f"Creating document collection {self.document_collection}")
            db.create_collection(self.document_collection)
        
        # Create edge collection if it doesn't exist
        if not db.has_collection(self.edge_collection):
            logger.info(f"Creating edge collection {self.edge_collection}")
            db.create_collection(self.edge_collection, edge=True)
        
        # Create vector collection if it doesn't exist
        if not db.has_collection(self.vector_collection):
            logger.info(f"Creating vector collection {self.vector_collection}")
            db.create_collection(self.vector_collection)
            # Add vector index if needed
            # This depends on the ArangoDB version and configuration
            try:
                db[self.vector_collection].add_persistent_index(
                    fields=["embedding"], 
                    unique=False,
                    sparse=True,
                    name="embedding_index"
                )
            except Exception as e:
                logger.warning(f"Could not create vector index: {e}")
        
        # Create graph if it doesn't exist
        if not db.has_graph(self.graph_name):
            logger.info(f"Creating graph {self.graph_name}")
            edge_definitions = [{
                'from_collections': [self.document_collection],
                'to_collections': [self.document_collection],
                'edge_collection': self.edge_collection
            }]
            db.create_graph(self.graph_name, edge_definitions)
    
    def store_document(self, document: IngestDocument) -> str:
        """
        Store a document in ArangoDB.
        
        Args:
            document: The document to store
            
        Returns:
            The ArangoDB document key
        """
        db = self.connection.db
        doc_collection = db[self.document_collection]
        
        # Prepare document data
        doc_data = {
            "_key": document.id.replace("/", "_").replace(" ", "_"),
            "content": document.content,
            "title": document.title or "",
            "metadata": document.metadata,
        }
        
        # Check if document already exists
        if doc_collection.has(doc_data["_key"]):
            logger.debug(f"Document {doc_data['_key']} already exists, updating")
            doc_collection.update(doc_data)
        else:
            logger.debug(f"Creating new document {doc_data['_key']}")
            doc_collection.insert(doc_data)
        
        # Store embedding if available
        if document.embedding:
            vec_collection = db[self.vector_collection]
            vec_data = {
                "_key": doc_data["_key"],
                "document_id": doc_data["_key"],
                "embedding": document.embedding,
            }
            
            if vec_collection.has(vec_data["_key"]):
                logger.debug(f"Vector {vec_data['_key']} already exists, updating")
                vec_collection.update(vec_data)
            else:
                logger.debug(f"Creating new vector {vec_data['_key']}")
                vec_collection.insert(vec_data)
        
        return doc_data["_key"]
    
    def store_relationship(self, relationship: DocumentRelation) -> str:
        """
        Store a relationship in ArangoDB.
        
        Args:
            relationship: The relationship to store
            
        Returns:
            The ArangoDB edge key
        """
        db = self.connection.db
        edge_collection = db[self.edge_collection]
        doc_collection = db[self.document_collection]
        
        # Clean keys
        source_key = relationship.source_id.replace("/", "_").replace(" ", "_")
        target_key = relationship.target_id.replace("/", "_").replace(" ", "_")
        
        # Ensure both documents exist
        if not doc_collection.has(source_key):
            logger.warning(f"Source document {source_key} does not exist")
            return None
        
        if not doc_collection.has(target_key):
            logger.warning(f"Target document {target_key} does not exist")
            return None
        
        # Generate edge ID
        edge_key = f"{source_key}_to_{target_key}_{relationship.relation_type.value}"
        
        # Prepare edge data
        edge_data = {
            "_key": edge_key,
            "_from": f"{self.document_collection}/{source_key}",
            "_to": f"{self.document_collection}/{target_key}",
            "relation_type": relationship.relation_type.value,
            "weight": relationship.weight,
            "metadata": relationship.metadata,
        }
        
        # Check if edge already exists
        try:
            if edge_collection.has(edge_key):
                logger.debug(f"Edge {edge_key} already exists, updating")
                edge_collection.update(edge_data)
            else:
                logger.debug(f"Creating new edge {edge_key}")
                edge_collection.insert(edge_data)
        except Exception as e:
            logger.error(f"Error storing relationship {edge_key}: {e}")
            return None
        
        return edge_key
    
    def store_dataset(self, dataset: IngestDataset) -> dict:
        """
        Store an entire dataset in ArangoDB.
        
        Args:
            dataset: The dataset to store
            
        Returns:
            Statistics about the stored data
        """
        logger.info(f"Storing dataset {dataset.name} in ArangoDB")
        
        # Track statistics
        stats = {
            "documents_stored": 0,
            "documents_failed": 0,
            "relationships_stored": 0,
            "relationships_failed": 0,
        }
        
        # Store documents
        for doc in dataset.documents:
            try:
                self.store_document(doc)
                stats["documents_stored"] += 1
            except Exception as e:
                logger.error(f"Error storing document {doc.id}: {e}")
                stats["documents_failed"] += 1
        
        # Store relationships
        for rel in dataset.relationships:
            try:
                result = self.store_relationship(rel)
                if result:
                    stats["relationships_stored"] += 1
                else:
                    stats["relationships_failed"] += 1
            except Exception as e:
                logger.error(f"Error storing relationship {rel.source_id} -> {rel.target_id}: {e}")
                stats["relationships_failed"] += 1
        
        logger.info(f"Stored {stats['documents_stored']} documents and {stats['relationships_stored']} relationships")
        return stats

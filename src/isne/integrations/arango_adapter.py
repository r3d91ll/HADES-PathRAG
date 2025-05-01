"""
ArangoDB adapter for the ISNE pipeline.

This module provides an adapter for integrating the ISNE pipeline
with ArangoDB storage used by HADES-PathRAG.
"""

import os
import uuid
import json
import logging
from typing import Dict, List, Any, Optional, Union, Tuple, cast
from datetime import datetime

from src.storage.arango.connection import ArangoConnection
from src.types.common import NodeID, EdgeID, NodeData, EdgeData, EmbeddingVector
from src.isne.types.models import (
    IngestDocument, 
    IngestDataset, 
    DocumentRelation, 
    RelationType
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ArangoISNEAdapter:
    """
    Adapter for integrating ISNE with ArangoDB.
    
    This adapter converts ISNE data models to ArangoDB document formats
    and handles storage and retrieval operations.
    """
    
    def __init__(
        self,
        arango_connection: Optional[ArangoConnection] = None,
        db_name: str = "pathrag",
        node_collection: str = "isne_nodes",
        edge_collection: str = "isne_edges",
        graph_name: str = "isne_graph",
        metadata_collection: str = "isne_metadata"
    ) -> None:
        """
        Initialize the ArangoDB adapter for ISNE.
        
        Args:
            arango_connection: Existing ArangoDB connection or None to create new
            db_name: Name of the ArangoDB database
            node_collection: Name of the collection storing nodes
            edge_collection: Name of the collection storing edges
            graph_name: Name of the graph in ArangoDB
            metadata_collection: Name of the collection storing metadata
        """
        # Use provided connection or create a new one
        self.conn = arango_connection or ArangoConnection(db_name=db_name)
        self.db_name = db_name
        self.node_collection = node_collection
        self.edge_collection = edge_collection
        self.graph_name = graph_name
        self.metadata_collection = metadata_collection
        
        # Ensure collections and graph exist
        self._ensure_collections()
    
    def _ensure_collections(self) -> None:
        """Ensure that required collections and graph exist."""
        # Create nodes collection if it doesn't exist
        if not self.conn.collection_exists(self.node_collection):
            self.conn.create_collection(self.node_collection)
            logger.info(f"Created node collection: {self.node_collection}")
            
        # Create edges collection if it doesn't exist
        if not self.conn.collection_exists(self.edge_collection):
            self.conn.create_edge_collection(self.edge_collection)
            logger.info(f"Created edge collection: {self.edge_collection}")
            
        # Create metadata collection if it doesn't exist
        if not self.conn.collection_exists(self.metadata_collection):
            self.conn.create_collection(self.metadata_collection)
            logger.info(f"Created metadata collection: {self.metadata_collection}")
            
        # Create graph if it doesn't exist
        if not self.conn.graph_exists(self.graph_name):
            self.conn.create_graph(
                self.graph_name,
                edge_definitions=[{
                    'edge_collection': self.edge_collection,
                    'from_vertex_collections': [self.node_collection],
                    'to_vertex_collections': [self.node_collection]
                }]
            )
            logger.info(f"Created graph: {self.graph_name}")
    
    def store_document(self, document: IngestDocument) -> str:
        """
        Store an ISNE document in ArangoDB.
        
        Args:
            document: ISNE document to store
            
        Returns:
            ArangoDB document key
        """
        # Convert ISNE document to ArangoDB document
        arango_doc = self._convert_document_to_arango(document)
        
        # Check if document already exists
        existing_key = self._get_document_key(document.id)
        
        if existing_key:
            # Update existing document
            result = self.conn.update_document(
                self.node_collection,
                arango_doc,
                key=existing_key
            )
            logger.debug(f"Updated document: {existing_key}")
            return existing_key
        else:
            # Create new document
            result = self.conn.insert_document(
                self.node_collection,
                arango_doc
            )
            doc_key = result["_key"]
            logger.debug(f"Created document: {doc_key}")
            return doc_key
    
    def store_relation(self, relation: DocumentRelation) -> str:
        """
        Store an ISNE relation in ArangoDB.
        
        Args:
            relation: ISNE relation to store
            
        Returns:
            ArangoDB edge key
        """
        # Convert ISNE relation to ArangoDB edge
        arango_edge = self._convert_relation_to_arango(relation)
        
        # Check if edge already exists
        existing_key = self._get_edge_key(relation.id)
        
        if existing_key:
            # Update existing edge
            result = self.conn.update_document(
                self.edge_collection,
                arango_edge,
                key=existing_key
            )
            logger.debug(f"Updated edge: {existing_key}")
            return existing_key
        else:
            # Create new edge
            result = self.conn.insert_edge(
                self.edge_collection,
                arango_edge
            )
            edge_key = result["_key"]
            logger.debug(f"Created edge: {edge_key}")
            return edge_key
    
    def store_dataset(self, dataset: IngestDataset) -> str:
        """
        Store an entire ISNE dataset in ArangoDB.
        
        Args:
            dataset: ISNE dataset to store
            
        Returns:
            ArangoDB metadata document key
        """
        # Store dataset metadata
        metadata_doc = {
            "_key": dataset.id,
            "type": "isne_dataset",
            "name": dataset.name,
            "description": dataset.description,
            "created_at": dataset.created_at.isoformat() if dataset.created_at else None,
            "updated_at": datetime.now().isoformat(),
            "document_count": len(dataset.documents),
            "relation_count": len(dataset.relations),
            "metadata": dataset.metadata
        }
        
        # Store metadata document
        result = self.conn.insert_document(
            self.metadata_collection,
            metadata_doc,
            overwrite=True
        )
        
        # Store all documents
        for doc_id, document in dataset.documents.items():
            self.store_document(document)
        
        # Store all relations
        for relation in dataset.relations:
            self.store_relation(relation)
        
        logger.info(f"Stored dataset {dataset.name} with {len(dataset.documents)} documents and {len(dataset.relations)} relations")
        return result["_key"]
    
    def load_document(self, document_id: str) -> Optional[IngestDocument]:
        """
        Load an ISNE document from ArangoDB.
        
        Args:
            document_id: ID of the document to load
            
        Returns:
            Loaded IngestDocument or None if not found
        """
        # Get document key
        doc_key = self._get_document_key(document_id)
        
        if not doc_key:
            logger.warning(f"Document not found: {document_id}")
            return None
        
        # Retrieve document
        arango_doc = self.conn.get_document(self.node_collection, doc_key)
        
        if not arango_doc:
            logger.warning(f"Document not found with key: {doc_key}")
            return None
        
        # Convert ArangoDB document to ISNE document
        return self._convert_arango_to_document(arango_doc)
    
    def load_relation(self, relation_id: str) -> Optional[DocumentRelation]:
        """
        Load an ISNE relation from ArangoDB.
        
        Args:
            relation_id: ID of the relation to load
            
        Returns:
            Loaded DocumentRelation or None if not found
        """
        # Get edge key
        edge_key = self._get_edge_key(relation_id)
        
        if not edge_key:
            logger.warning(f"Relation not found: {relation_id}")
            return None
        
        # Retrieve edge
        arango_edge = self.conn.get_document(self.edge_collection, edge_key)
        
        if not arango_edge:
            logger.warning(f"Edge not found with key: {edge_key}")
            return None
        
        # Convert ArangoDB edge to ISNE relation
        return self._convert_arango_to_relation(arango_edge)
    
    def load_dataset(self, dataset_id: str) -> Optional[IngestDataset]:
        """
        Load an entire ISNE dataset from ArangoDB.
        
        Args:
            dataset_id: ID of the dataset to load
            
        Returns:
            Loaded IngestDataset or None if not found
        """
        # Retrieve dataset metadata
        metadata_doc = self.conn.get_document(self.metadata_collection, dataset_id)
        
        if not metadata_doc:
            logger.warning(f"Dataset not found: {dataset_id}")
            return None
        
        # Create dataset object
        dataset = IngestDataset(
            id=metadata_doc["_key"],
            name=metadata_doc["name"],
            description=metadata_doc.get("description"),
            metadata=metadata_doc.get("metadata", {}),
            created_at=datetime.fromisoformat(metadata_doc["created_at"]) if metadata_doc.get("created_at") else None,
            updated_at=datetime.fromisoformat(metadata_doc["updated_at"]) if metadata_doc.get("updated_at") else None
        )
        
        # Query for documents in this dataset
        aql_query = f"""
        FOR doc IN {self.node_collection}
        FILTER doc.metadata.dataset_id == @dataset_id
        RETURN doc
        """
        
        documents = self.conn.aql_query(aql_query, bind_vars={"dataset_id": dataset_id})
        
        # Add documents to dataset
        for arango_doc in documents:
            document = self._convert_arango_to_document(arango_doc)
            if document:
                dataset.add_document(document)
        
        # Query for relations in this dataset
        aql_query = f"""
        FOR edge IN {self.edge_collection}
        FILTER edge.metadata.dataset_id == @dataset_id
        RETURN edge
        """
        
        relations = self.conn.aql_query(aql_query, bind_vars={"dataset_id": dataset_id})
        
        # Add relations to dataset
        for arango_edge in relations:
            relation = self._convert_arango_to_relation(arango_edge)
            if relation:
                dataset.add_relation(relation)
        
        logger.info(f"Loaded dataset {dataset.name} with {len(dataset.documents)} documents and {len(dataset.relations)} relations")
        return dataset
    
    def search_similar_documents(
        self, 
        embedding: EmbeddingVector, 
        limit: int = 10,
        min_score: float = 0.7
    ) -> List[Tuple[IngestDocument, float]]:
        """
        Search for documents with similar embeddings.
        
        Args:
            embedding: Embedding vector to search for
            limit: Maximum number of results to return
            min_score: Minimum similarity score (0-1)
            
        Returns:
            List of (document, score) tuples
        """
        # Convert embedding to list if needed
        if isinstance(embedding, (np.ndarray, list)):
            if isinstance(embedding, np.ndarray):
                embedding_list = embedding.tolist()
            else:
                embedding_list = embedding
        else:
            raise TypeError("Embedding must be a list or numpy array")
        
        # Execute vector search if available
        if hasattr(self.conn, 'vector_search'):
            search_results = self.conn.vector_search(
                self.node_collection,
                embedding_list,
                field="embedding",
                limit=limit,
                min_score=min_score
            )
        else:
            # Fallback to AQL-based vector search
            aql_query = f"""
            FOR doc IN {self.node_collection}
            FILTER doc.embedding != null
            LET score = LENGTH(doc.embedding) == LENGTH(@embedding) ? 
                        1 / (1 + SQRT(SUM(
                            FOR i IN RANGE(0, LENGTH(doc.embedding) - 1)
                            RETURN POW(doc.embedding[i] - @embedding[i], 2)
                        )))
                        : 0
            FILTER score >= @min_score
            SORT score DESC
            LIMIT @limit
            RETURN {{document: doc, score: score}}
            """
            
            search_results = self.conn.aql_query(
                aql_query,
                bind_vars={
                    "embedding": embedding_list,
                    "min_score": min_score,
                    "limit": limit
                }
            )
        
        # Convert results to ISNE documents
        results = []
        for result in search_results:
            document = self._convert_arango_to_document(result["document"])
            if document:
                results.append((document, result["score"]))
        
        return results
    
    def _get_document_key(self, document_id: str) -> Optional[str]:
        """
        Get the ArangoDB key for a document ID.
        
        Args:
            document_id: ISNE document ID
            
        Returns:
            ArangoDB document key or None if not found
        """
        aql_query = f"""
        FOR doc IN {self.node_collection}
        FILTER doc.id == @id
        RETURN doc._key
        """
        
        result = self.conn.aql_query(aql_query, bind_vars={"id": document_id})
        
        if result and len(result) > 0:
            return result[0]
        
        return None
    
    def _get_edge_key(self, relation_id: str) -> Optional[str]:
        """
        Get the ArangoDB key for a relation ID.
        
        Args:
            relation_id: ISNE relation ID
            
        Returns:
            ArangoDB edge key or None if not found
        """
        aql_query = f"""
        FOR edge IN {self.edge_collection}
        FILTER edge.id == @id
        RETURN edge._key
        """
        
        result = self.conn.aql_query(aql_query, bind_vars={"id": relation_id})
        
        if result and len(result) > 0:
            return result[0]
        
        return None
    
    def _convert_document_to_arango(self, document: IngestDocument) -> Dict[str, Any]:
        """
        Convert an ISNE document to ArangoDB format.
        
        Args:
            document: ISNE document to convert
            
        Returns:
            ArangoDB document representation
        """
        # Convert embedding to list if it's numpy array
        embedding = None
        if document.embedding is not None:
            if isinstance(document.embedding, np.ndarray):
                embedding = document.embedding.tolist()
            else:
                embedding = document.embedding
        
        # Format timestamps
        created_at = document.created_at.isoformat() if document.created_at else None
        updated_at = document.updated_at.isoformat() if document.updated_at else None
        
        # Create ArangoDB document
        return {
            "id": document.id,
            "content": document.content,
            "source": document.source,
            "document_type": document.document_type,
            "title": document.title,
            "author": document.author,
            "created_at": created_at,
            "updated_at": updated_at,
            "embedding": embedding,
            "embedding_model": document.embedding_model,
            "metadata": document.metadata,
            "chunks": document.chunks,
            "tags": document.tags
        }
    
    def _convert_arango_to_document(self, arango_doc: Dict[str, Any]) -> Optional[IngestDocument]:
        """
        Convert an ArangoDB document to ISNE document.
        
        Args:
            arango_doc: ArangoDB document to convert
            
        Returns:
            ISNE document representation
        """
        try:
            # Parse timestamps
            created_at = None
            if arango_doc.get("created_at"):
                try:
                    created_at = datetime.fromisoformat(arango_doc["created_at"])
                except (ValueError, TypeError):
                    pass
            
            updated_at = None
            if arango_doc.get("updated_at"):
                try:
                    updated_at = datetime.fromisoformat(arango_doc["updated_at"])
                except (ValueError, TypeError):
                    pass
            
            # Create ISNE document
            return IngestDocument(
                id=arango_doc.get("id") or arango_doc["_key"],
                content=arango_doc["content"],
                source=arango_doc["source"],
                document_type=arango_doc["document_type"],
                title=arango_doc.get("title"),
                author=arango_doc.get("author"),
                created_at=created_at,
                updated_at=updated_at,
                metadata=arango_doc.get("metadata", {}),
                embedding=arango_doc.get("embedding"),
                embedding_model=arango_doc.get("embedding_model"),
                chunks=arango_doc.get("chunks", []),
                tags=arango_doc.get("tags", [])
            )
        except KeyError as e:
            logger.error(f"Error converting ArangoDB document to ISNE document: {e}")
            return None
    
    def _convert_relation_to_arango(self, relation: DocumentRelation) -> Dict[str, Any]:
        """
        Convert an ISNE relation to ArangoDB edge.
        
        Args:
            relation: ISNE relation to convert
            
        Returns:
            ArangoDB edge representation
        """
        # Get document keys for from and to
        from_key = self._get_document_key(relation.source_id)
        to_key = self._get_document_key(relation.target_id)
        
        if not from_key or not to_key:
            logger.warning(f"Cannot create edge: source or target not found: {relation.source_id} -> {relation.target_id}")
            # Create placeholder keys that match the ID format
            # This allows storing the edge even if nodes don't exist yet
            from_key = from_key or relation.source_id
            to_key = to_key or relation.target_id
        
        # Format timestamp
        created_at = relation.created_at.isoformat() if relation.created_at else None
        
        # Create ArangoDB edge
        return {
            "id": relation.id,
            "_from": f"{self.node_collection}/{from_key}",
            "_to": f"{self.node_collection}/{to_key}",
            "source_id": relation.source_id,
            "target_id": relation.target_id,
            "relation_type": relation.relation_type.value,
            "weight": relation.weight,
            "bidirectional": relation.bidirectional,
            "created_at": created_at,
            "metadata": relation.metadata
        }
    
    def _convert_arango_to_relation(self, arango_edge: Dict[str, Any]) -> Optional[DocumentRelation]:
        """
        Convert an ArangoDB edge to ISNE relation.
        
        Args:
            arango_edge: ArangoDB edge to convert
            
        Returns:
            ISNE relation representation
        """
        try:
            # Parse timestamp
            created_at = None
            if arango_edge.get("created_at"):
                try:
                    created_at = datetime.fromisoformat(arango_edge["created_at"])
                except (ValueError, TypeError):
                    pass
            
            # Convert relation type string to enum
            try:
                relation_type = RelationType(arango_edge["relation_type"])
            except (ValueError, KeyError):
                relation_type = RelationType.RELATED_TO
            
            # Create ISNE relation
            return DocumentRelation(
                id=arango_edge.get("id") or arango_edge["_key"],
                source_id=arango_edge["source_id"],
                target_id=arango_edge["target_id"],
                relation_type=relation_type,
                weight=arango_edge.get("weight", 1.0),
                bidirectional=arango_edge.get("bidirectional", False),
                metadata=arango_edge.get("metadata", {}),
                created_at=created_at
            )
        except KeyError as e:
            logger.error(f"Error converting ArangoDB edge to ISNE relation: {e}")
            return None

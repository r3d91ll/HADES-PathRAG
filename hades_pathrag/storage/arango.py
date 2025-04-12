"""
ArangoDB storage implementation for PathRAG.

This module provides implementations of the storage interfaces
using ArangoDB as the backend database.
"""
from typing import Dict, List, Optional, Tuple, Any, Set, Type, cast
import logging
import time
from dataclasses import dataclass, field

import numpy as np
from arango import ArangoClient
from arango.database import Database
from arango.collection import Collection
from arango.graph import Graph as ArangoGraph
from arango.exceptions import (
    CollectionCreateError,
    GraphCreateError,
    DocumentInsertError,
    DocumentGetError,
    AQLQueryExecuteError,
)

from .base import BaseStorage, BaseVectorStorage, BaseDocumentStorage, BaseGraphStorage, NodeID, Embedding

logger = logging.getLogger(__name__)


class ArangoDBConnection:
    """Manages the connection to ArangoDB."""
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 8529,
        username: str = "root",
        password: str = "root",
        database: str = "pathrag",
        timeout: int = 30,
    ) -> None:
        """
        Initialize the ArangoDB connection.
        
        Args:
            host: ArangoDB host
            port: ArangoDB port
            username: ArangoDB username
            password: ArangoDB password
            database: Database name
            timeout: Connection timeout in seconds
        """
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.database_name = database
        self.timeout = timeout
        
        self.client: Optional[ArangoClient] = None
        self.db: Optional[Database] = None
        
        logger.info(f"Initialized ArangoDB connection to {host}:{port}/{database}")
    
    def connect(self) -> Database:
        """
        Connect to ArangoDB and return the database instance.
        
        Returns:
            ArangoDB database instance
        """
        if self.db is not None:
            return self.db
        
        # Connect to ArangoDB
        self.client = ArangoClient(
            hosts=f"http://{self.host}:{self.port}",
            http_client_params={"timeout": self.timeout}
        )
        
        # Connect to system db first to ensure our database exists
        sys_db = self.client.db(
            "_system",
            username=self.username,
            password=self.password,
            verify=True
        )
        
        # Create database if it doesn't exist
        if not sys_db.has_database(self.database_name):
            logger.info(f"Creating database {self.database_name}")
            sys_db.create_database(
                name=self.database_name,
                users=[{"username": self.username, "password": self.password, "active": True}]
            )
        
        # Connect to the application database
        self.db = self.client.db(
            self.database_name,
            username=self.username,
            password=self.password,
            verify=True
        )
        
        logger.info(f"Connected to ArangoDB database {self.database_name}")
        return self.db
    
    def close(self) -> None:
        """Close the ArangoDB connection."""
        # ArangoClient doesn't have a close method
        # Just dereference the objects to allow garbage collection
        self.db = None
        self.client = None
        logger.info("ArangoDB connection closed")


class ArangoVectorStorage(BaseVectorStorage):
    """ArangoDB implementation of vector storage with enhanced features.
    
    This class provides vector storage and retrieval using ArangoDB,
    including vector similarity search and hybrid search capabilities.
    It implements the EnhancedVectorStorage protocol for advanced operations.
    """
    
    def __init__(
        self,
        connection: ArangoDBConnection,
        collection_name: str = "embeddings",
        dimension: int = 128,
    ) -> None:
        """
        Initialize ArangoDB vector storage.
        
        Args:
            connection: ArangoDB connection
            collection_name: Name of the collection to store embeddings
            dimension: Dimension of the embedding vectors
        """
        self.connection = connection
        self.collection_name = collection_name
        self.dimension = dimension
        self.collection: Optional[Collection] = None
        
        logger.info(f"Initialized ArangoDB vector storage with collection {collection_name}")
    
    def initialize(self) -> None:
        """Initialize the vector storage, creating necessary collections."""
        db = self.connection.connect()
        
        # Create collection if it doesn't exist
        if not db.has_collection(self.collection_name):
            logger.info(f"Creating collection {self.collection_name}")
            self.collection = db.create_collection(
                name=self.collection_name,
                edge=False
            )
            
            # Create index for vector search
            # Note: ArangoDB 3.10+ supports vector indexing with ArangoSearch
            self.collection.add_hash_index(fields=["_key"], unique=True)
        else:
            self.collection = db.collection(self.collection_name)
        
        # Create ArangoSearch view for vector search if it doesn't exist
        view_name = f"{self.collection_name}_view"
        if not db.has_view(view_name):
            logger.info(f"Creating ArangoSearch view {view_name}")
            db.create_arangosearch_view(
                name=view_name,
                properties={
                    "links": {
                        self.collection_name: {
                            "includeAllFields": True,
                            "fields": {
                                "embedding": {
                                    "analyzers": ["identity"],
                                }
                            }
                        }
                    }
                }
            )
    
    def store_embedding(self, node_id: NodeID, embedding: Embedding, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Store a node embedding with optional metadata.
        
        Args:
            node_id: Unique identifier for the node
            embedding: Vector embedding of the node
            metadata: Optional metadata to store with the embedding
        """
        if self.collection is None:
            raise ValueError("Storage not initialized")
        
        # Convert numpy array to list for JSON serialization
        embedding_list = embedding.tolist()
        
        document = {
            "_key": node_id,
            "embedding": embedding_list
        }
        
        if metadata:
            # Ensure we don't overwrite _key
            metadata_clean = {k: v for k, v in metadata.items() if k != "_key"}
            document.update(metadata_clean)
        
        try:
            self.collection.insert(document, overwrite=True)
        except DocumentInsertError as e:
            logger.error(f"Error storing embedding for {node_id}: {e}")
            # Try updating if insert failed
            try:
                self.collection.update({"_key": node_id}, document)
            except Exception as e2:
                logger.error(f"Error updating embedding for {node_id}: {e2}")
                raise
    
    def get_embedding(self, node_id: NodeID) -> Optional[Embedding]:
        """
        Get embedding by node ID.
        
        Args:
            node_id: ID of the node to retrieve
            
        Returns:
            Node embedding if found, None otherwise
        """
        if self.collection is None:
            raise ValueError("Storage not initialized")
        
        try:
            document = self.collection.get({"_key": node_id})
            if document and "embedding" in document:
                return cast(np.ndarray, np.array(document["embedding"], dtype=np.float32))
            return None
        except DocumentGetError:
            return None
    
    def retrieve_similar(
        self, 
        query_embedding: Embedding, 
        k: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[NodeID, float, Dict[str, Any]]]:
        """
        Find nodes with embeddings similar to the query embedding.
        
        Args:
            query_embedding: Query vector
            k: Number of results to return
            filter_metadata: Optional filter to apply to metadata
            
        Returns:
            List of (node_id, similarity_score, metadata) tuples
        """
        if self.collection is None:
            raise ValueError("Storage not initialized")
        
        db = self.connection.connect()
        
        # Convert numpy array to list for AQL
        query_vector = query_embedding.tolist()
        
        # Build AQL query
        aql = """
        FOR doc IN @@collection
        """
        
        bind_vars = {"@collection": self.collection_name}
        
        # Add filters if specified
        if filter_metadata:
            filter_conditions = []
            for key, value in filter_metadata.items():
                filter_var = f"filter_{key}"
                filter_conditions.append(f"doc.{key} == @{filter_var}")
                bind_vars[filter_var] = value
            
            if filter_conditions:
                aql += " FILTER " + " AND ".join(filter_conditions)
        
        # Add vector search
        # Note: Using COSINE_DISTANCE since ArangoDB 3.10+ has native vector search
        aql += """
        LET distance = LENGTH(doc.embedding) == LENGTH(@queryVector) ? 
            1.0 - SUM(
                FOR i IN 0..LENGTH(@queryVector)-1
                RETURN doc.embedding[i] * @queryVector[i]
            ) / (
                SQRT(SUM(FOR i IN doc.embedding RETURN i*i)) * 
                SQRT(SUM(FOR i IN @queryVector RETURN i*i))
            ) : 1.0
        SORT distance ASC
        LIMIT @k
        RETURN {
            id: doc._key,
            similarity: 1.0 - distance,
            metadata: doc
        }
        """
        
        bind_vars["queryVector"] = query_vector
        bind_vars["k"] = k
        
        try:
            # Execute AQL query
            cursor = db.aql.execute(aql, bind_vars=bind_vars)
            
            # Process results
            results: List[Tuple[NodeID, float, Dict[str, Any]]] = []
            for doc in cursor:
                node_id = doc["id"]
                similarity = doc["similarity"]
                metadata = doc["metadata"]
                
                # Remove embedding from metadata to save memory
                if "embedding" in metadata:
                    del metadata["embedding"]
                
                results.append((node_id, similarity, metadata))
            
            return results
        except AQLQueryExecuteError as e:
            logger.error(f"Error executing vector similarity query: {e}")
            return []
    
    def delete_embedding(self, node_id: NodeID) -> bool:
        """
        Delete a node embedding.
        
        Args:
            node_id: ID of the node to delete
            
        Returns:
            True if deleted, False if not found
        """
        if self.collection is None:
            raise ValueError("Storage not initialized")
        
        try:
            self.collection.delete({"_key": node_id})
            return True
        except DocumentGetError:
            return False
    
    def update_metadata(self, node_id: NodeID, metadata: Dict[str, Any]) -> bool:
        """
        Update metadata for a node embedding.
        
        Args:
            node_id: ID of the node to update
            metadata: New metadata to store
            
        Returns:
            True if updated, False if not found
        """
        if self.collection is None:
            raise ValueError("Storage not initialized")
        
        try:
            # Ensure we don't overwrite _key or embedding
            metadata_clean = {k: v for k, v in metadata.items() 
                             if k != "_key" and k != "embedding"}
            
            self.collection.update({"_key": node_id}, metadata_clean)
            return True
        except DocumentGetError:
            return False
    
    def close(self) -> None:
        """Close the vector storage."""
        self.collection = None


class ArangoDocumentStorage(BaseDocumentStorage):
    """ArangoDB implementation of document storage."""
    
    def __init__(
        self,
        connection: ArangoDBConnection,
        collection_name: str = "documents",
    ) -> None:
        """
        Initialize ArangoDB document storage.
        
        Args:
            connection: ArangoDB connection
            collection_name: Name of the collection to store documents
        """
        self.connection = connection
        self.collection_name = collection_name
        self.collection: Optional[Collection] = None
    
    def initialize(self) -> None:
        """Initialize the document storage, creating necessary collections."""
        db = self.connection.connect()
        
        # Create collection if it doesn't exist
        if not db.has_collection(self.collection_name):
            logger.info(f"Creating collection {self.collection_name}")
            self.collection = db.create_collection(
                name=self.collection_name,
                edge=False
            )
            
            # Create index for efficient retrieval
            self.collection.add_hash_index(fields=["_key"], unique=True)
        else:
            self.collection = db.collection(self.collection_name)
        
        # Create fulltext index for content search
        try:
            self.collection.add_fulltext_index(fields=["content"])
        except Exception as e:
            logger.warning(f"Could not create fulltext index: {e}")
    
    def store_document(self, doc_id: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Store a document with optional metadata.
        
        Args:
            doc_id: Unique identifier for the document
            content: Text content of the document
            metadata: Optional metadata to store with the document
        """
        if self.collection is None:
            raise ValueError("Storage not initialized")
        
        document = {
            "_key": doc_id,
            "content": content
        }
        
        if metadata:
            # Ensure we don't overwrite _key or content
            metadata_clean = {k: v for k, v in metadata.items() 
                             if k != "_key" and k != "content"}
            document.update(metadata_clean)
        
        try:
            self.collection.insert(document, overwrite=True)
        except DocumentInsertError as e:
            logger.error(f"Error storing document {doc_id}: {e}")
            # Try updating if insert failed
            try:
                self.collection.update({"_key": doc_id}, document)
            except Exception as e2:
                logger.error(f"Error updating document {doc_id}: {e2}")
                raise
    
    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """
        Get document by ID.
        
        Args:
            doc_id: ID of the document to retrieve
            
        Returns:
            Document with content and metadata if found, None otherwise
        """
        if self.collection is None:
            raise ValueError("Storage not initialized")
        
        try:
            document = self.collection.get({"_key": doc_id})
            return document if document else None
        except DocumentGetError:
            return None
    
    def list_documents(self, filter_metadata: Optional[Dict[str, Any]] = None) -> List[str]:
        """
        List document IDs, optionally filtered by metadata.
        
        Args:
            filter_metadata: Optional filter to apply to metadata
            
        Returns:
            List of document IDs matching the filter
        """
        if self.collection is None:
            raise ValueError("Storage not initialized")
        
        db = self.connection.connect()
        
        # Build AQL query
        aql = f"FOR doc IN {self.collection_name}"
        bind_vars = {}
        
        # Add filters if specified
        if filter_metadata:
            filter_conditions = []
            for i, (key, value) in enumerate(filter_metadata.items()):
                filter_var = f"filter_{i}"
                filter_conditions.append(f"doc.{key} == @{filter_var}")
                bind_vars[filter_var] = value
            
            if filter_conditions:
                aql += " FILTER " + " AND ".join(filter_conditions)
        
        aql += " RETURN doc._key"
        
        try:
            # Execute AQL query
            cursor = db.aql.execute(aql, bind_vars=bind_vars)
            return list(cursor)
        except AQLQueryExecuteError as e:
            logger.error(f"Error listing documents: {e}")
            return []
    
    def delete_document(self, doc_id: str) -> bool:
        """
        Delete a document.
        
        Args:
            doc_id: ID of the document to delete
            
        Returns:
            True if deleted, False if not found
        """
        if self.collection is None:
            raise ValueError("Storage not initialized")
        
        try:
            self.collection.delete({"_key": doc_id})
            return True
        except DocumentGetError:
            return False
    
    def close(self) -> None:
        """Close the document storage."""
        self.collection = None


class ArangoGraphStorage(BaseGraphStorage):
    """ArangoDB implementation of graph storage."""
    
    def __init__(
        self,
        connection: ArangoDBConnection,
        graph_name: str = "pathrag",
        node_collection_name: str = "nodes",
        edge_collection_name: str = "edges",
    ) -> None:
        """
        Initialize ArangoDB graph storage.
        
        Args:
            connection: ArangoDB connection
            graph_name: Name of the graph
            node_collection_name: Name of the node collection
            edge_collection_name: Name of the edge collection
        """
        self.connection = connection
        self.graph_name = graph_name
        self.node_collection_name = node_collection_name
        self.edge_collection_name = edge_collection_name
        
        self.graph: Optional[ArangoGraph] = None
        self.node_collection: Optional[Collection] = None
        self.edge_collection: Optional[Collection] = None
    
    def initialize(self) -> None:
        """Initialize the graph storage, creating necessary collections and graph."""
        db = self.connection.connect()
        
        # Create node collection if it doesn't exist
        if not db.has_collection(self.node_collection_name):
            logger.info(f"Creating collection {self.node_collection_name}")
            self.node_collection = db.create_collection(
                name=self.node_collection_name,
                edge=False
            )
        else:
            self.node_collection = db.collection(self.node_collection_name)
        
        # Create edge collection if it doesn't exist
        if not db.has_collection(self.edge_collection_name):
            logger.info(f"Creating edge collection {self.edge_collection_name}")
            self.edge_collection = db.create_collection(
                name=self.edge_collection_name,
                edge=True
            )
        else:
            self.edge_collection = db.collection(self.edge_collection_name)
        
        # Create graph if it doesn't exist
        if not db.has_graph(self.graph_name):
            logger.info(f"Creating graph {self.graph_name}")
            self.graph = db.create_graph(
                name=self.graph_name,
                edge_definitions=[
                    {
                        'collection': self.edge_collection_name,
                        'from_collections': [self.node_collection_name],
                        'to_collections': [self.node_collection_name]
                    }
                ]
            )
        else:
            self.graph = db.graph(self.graph_name)
    
    def store_node(self, node_id: NodeID, attributes: Dict[str, Any]) -> None:
        """
        Store a node with attributes.
        
        Args:
            node_id: Unique identifier for the node
            attributes: Node attributes
        """
        if self.node_collection is None:
            raise ValueError("Storage not initialized")
        
        document = {
            "_key": node_id
        }
        document.update(attributes)
        
        try:
            self.node_collection.insert(document, overwrite=True)
        except DocumentInsertError as e:
            logger.error(f"Error storing node {node_id}: {e}")
            # Try updating if insert failed
            try:
                self.node_collection.update({"_key": node_id}, attributes)
            except Exception as e2:
                logger.error(f"Error updating node {node_id}: {e2}")
                raise
    
    def store_edge(
        self,
        source_id: NodeID,
        target_id: NodeID,
        relation_type: str,
        weight: float = 1.0,
        attributes: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Store an edge between nodes.
        
        Args:
            source_id: Source node ID
            target_id: Target node ID
            relation_type: Type of relationship
            weight: Edge weight
            attributes: Optional edge attributes
        """
        if self.edge_collection is None:
            raise ValueError("Storage not initialized")
        
        # Compute edge key based on source, target, and relation type
        edge_key = f"{source_id}_{relation_type}_{target_id}"
        
        edge = {
            "_key": edge_key,
            "_from": f"{self.node_collection_name}/{source_id}",
            "_to": f"{self.node_collection_name}/{target_id}",
            "relation_type": relation_type,
            "weight": weight
        }
        
        if attributes:
            edge.update(attributes)
        
        try:
            self.edge_collection.insert(edge, overwrite=True)
        except DocumentInsertError as e:
            logger.error(f"Error storing edge {edge_key}: {e}")
            # Try updating if insert failed
            try:
                update_data = {"weight": weight, "relation_type": relation_type}
                if attributes:
                    update_data.update(attributes)
                self.edge_collection.update({"_key": edge_key}, update_data)
            except Exception as e2:
                logger.error(f"Error updating edge {edge_key}: {e2}")
                raise
    
    def get_node(self, node_id: NodeID) -> Optional[Dict[str, Any]]:
        """
        Get node by ID.
        
        Args:
            node_id: ID of the node to retrieve
            
        Returns:
            Node attributes if found, None otherwise
        """
        if self.node_collection is None:
            raise ValueError("Storage not initialized")
        
        try:
            node = self.node_collection.get({"_key": node_id})
            return node if node else None
        except DocumentGetError:
            return None
    
    def get_neighbors(
        self, 
        node_id: NodeID, 
        direction: str = "outbound", 
        relation_types: Optional[List[str]] = None
    ) -> List[Tuple[NodeID, str, float]]:
        """
        Get neighbors of a node.
        
        Args:
            node_id: ID of the node
            direction: Direction of edges ("outbound", "inbound", or "any")
            relation_types: Optional filter by relation types
            
        Returns:
            List of (neighbor_id, relation_type, weight) tuples
        """
        if self.graph is None:
            raise ValueError("Storage not initialized")
        
        db = self.connection.connect()
        
        # Validate direction
        if direction not in ["outbound", "inbound", "any"]:
            raise ValueError(f"Invalid direction: {direction}")
        
        # Build AQL query
        aql = f"""
        FOR v, e IN 1..1 {direction} 
        '{self.node_collection_name}/{node_id}' 
        {self.edge_collection_name}
        """
        
        bind_vars = {}
        
        # Add relation type filter if specified
        if relation_types:
            aql += " FILTER e.relation_type IN @relation_types"
            bind_vars["relation_types"] = relation_types
        
        aql += """
        RETURN {
            neighbor_id: v._key,
            relation_type: e.relation_type,
            weight: e.weight
        }
        """
        
        try:
            # Execute AQL query
            cursor = db.aql.execute(aql, bind_vars=bind_vars)
            
            # Process results
            results: List[Tuple[NodeID, str, float]] = []
            for doc in cursor:
                neighbor_id = doc["neighbor_id"]
                relation_type = doc["relation_type"]
                weight = doc.get("weight", 1.0)
                
                results.append((neighbor_id, relation_type, weight))
            
            return results
        except AQLQueryExecuteError as e:
            logger.error(f"Error getting neighbors for {node_id}: {e}")
            return []
    
    def query_subgraph(
        self,
        start_nodes: List[NodeID],
        max_depth: int = 2,
        relation_types: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Query a subgraph starting from given nodes.
        
        Args:
            start_nodes: List of starting node IDs
            max_depth: Maximum traversal depth
            relation_types: Optional filter by relation types
            
        Returns:
            Dictionary with "nodes" and "edges" representing the subgraph
        """
        if self.graph is None:
            raise ValueError("Storage not initialized")
        
        db = self.connection.connect()
        
        # Validate max_depth
        if max_depth < 1:
            raise ValueError("max_depth must be at least 1")
        
        # For each start node, build query
        all_nodes: Dict[str, Dict[str, Any]] = {}
        all_edges: Dict[str, Dict[str, Any]] = {}
        
        for start_node in start_nodes:
            # Build AQL query
            aql = f"""
            FOR v, e, p IN 1..{max_depth} OUTBOUND 
            '{self.node_collection_name}/{start_node}' 
            {self.edge_collection_name}
            """
            
            bind_vars = {}
            
            # Add relation type filter if specified
            if relation_types:
                aql += " FILTER e.relation_type IN @relation_types"
                bind_vars["relation_types"] = relation_types
            
            aql += """
            RETURN {
                nodes: p.vertices,
                edges: p.edges
            }
            """
            
            try:
                # Execute AQL query
                cursor = db.aql.execute(aql, bind_vars=bind_vars)
                
                # Process results
                for doc in cursor:
                    # Add nodes
                    for node in doc["nodes"]:
                        node_id = node["_key"]
                        if node_id not in all_nodes:
                            all_nodes[node_id] = node
                    
                    # Add edges
                    for edge in doc["edges"]:
                        edge_id = edge["_key"]
                        if edge_id not in all_edges:
                            all_edges[edge_id] = edge
            except AQLQueryExecuteError as e:
                logger.error(f"Error querying subgraph from {start_node}: {e}")
        
        # Add start nodes if they're not already in results
        for node_id in start_nodes:
            if node_id not in all_nodes:
                node = self.get_node(node_id)
                if node:
                    all_nodes[node_id] = node
        
        return {
            "nodes": list(all_nodes.values()),
            "edges": list(all_edges.values())
        }
    
    def close(self) -> None:
        """Close the graph storage."""
        self.graph = None
        self.node_collection = None
        self.edge_collection = None

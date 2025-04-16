"""
Enhanced ArangoDB storage implementation for PathRAG.

This module extends the base ArangoDB storage implementation with 
additional features required by the EnhancedStorage protocols.
"""
from typing import Dict, List, Optional, Tuple, Any, Set, Type, cast, Union
import logging
import time
from datetime import datetime
import json

import numpy as np
from arango import ArangoClient  # type: ignore[attr-defined]
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

# Import common types from our centralized typing module
from hades_pathrag.typings import (
    NodeIDType, NodeData, EdgeData, EmbeddingArray
)

from .base import BaseVectorStorage, BaseDocumentStorage, BaseGraphStorage
from .arango import ArangoDBConnection, ArangoVectorStorage, ArangoDocumentStorage, ArangoGraphStorage
from .interfaces import (
    EnhancedVectorStorage, EnhancedDocumentStorage, EnhancedGraphStorage,
    StorageStats, DocumentChunk, BulkOperationResult,
    QueryOperator, MetadataCondition, MetadataQuery, StorageTransaction
)
from .edge_types import EDGE_TYPES, EdgeCategory, get_edge_weight, create_edge_data
from .path_traversal import PathQuery, PathResult, execute_path_query, find_paths_between, expand_paths_from_nodes
from ..utils.text import chunk_text

logger = logging.getLogger(__name__)


class EnhancedArangoVectorStorage(ArangoVectorStorage):
    """
    Enhanced ArangoDB vector storage implementation.
    
    This class extends the base ArangoVectorStorage with additional features
    required by the EnhancedVectorStorage protocol, including bulk operations,
    hybrid search, and metadata filtering.
    """
    
    def bulk_store_embeddings(
        self, 
        items: List[Tuple[NodeIDType, EmbeddingArray, Optional[Dict[str, Any]]]]
    ) -> BulkOperationResult:
        """
        Store multiple embeddings in a single batch operation.
        
        Args:
            items: List of (node_id, embedding, metadata) tuples
            
        Returns:
            Result of the bulk operation
        """
        if self.collection is None:
            raise ValueError("Storage not initialized")
        
        documents = []
        for node_id, embedding, metadata in items:
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
            
            documents.append(document)
        
        success_count = 0
        error_count = 0
        errors: Dict[str, str] = {}
        
        try:
            # Use import_bulk for better performance
            result = self.collection.import_bulk(
                documents,
                on_duplicate="update",
                halt_on_error=False
            )
            
            # Process result
            success_count = result.get("created", 0) + result.get("updated", 0)
            error_count = result.get("errors", 0)
            
            # Check for detailed errors if available
            if "details" in result and result["details"]:
                for detail in result["details"]:
                    if "errorMessage" in detail:
                        errors[detail.get("_key", "unknown")] = detail["errorMessage"]
            
        except Exception as e:
            logger.error(f"Error in bulk store operation: {e}")
            # If no items were successfully stored, add a generic error
            if success_count == 0:
                errors["batch"] = str(e)
                error_count = len(items)
        
        return BulkOperationResult(
            success_count=success_count,
            error_count=error_count,
            errors=errors
        )
    
    def query_by_metadata(
        self, 
        query: MetadataQuery,
        limit: int = 100
    ) -> List[Tuple[NodeIDType, Dict[str, Any]]]:
        """
        Find nodes by metadata query.
        
        Args:
            query: Metadata query to filter by
            limit: Maximum number of results
            
        Returns:
            List of (node_id, metadata) tuples
        """
        if self.collection is None:
            raise ValueError("Storage not initialized")
        
        db = self.connection.connect()
        
        # Build AQL query
        aql = f"FOR doc IN {self.collection_name} "
        
        # Add filter conditions
        if query.conditions:
            conditions = []
            bind_vars = {}
            
            for i, condition in enumerate(query.conditions):
                field_var = f"field_{i}"
                value_var = f"value_{i}"
                
                # Map operator to AQL
                aql_op = {
                    QueryOperator.EQUALS: "==",
                    QueryOperator.NOT_EQUALS: "!=",
                    QueryOperator.GREATER_THAN: ">",
                    QueryOperator.GREATER_EQUAL: ">=",
                    QueryOperator.LESS_THAN: "<",
                    QueryOperator.LESS_EQUAL: "<=",
                    QueryOperator.CONTAINS: "IN",
                    QueryOperator.NOT_CONTAINS: "NOT IN",
                    QueryOperator.IN: "IN",
                    QueryOperator.NOT_IN: "NOT IN",
                    QueryOperator.MATCHES: "=~",  # Regex match
                }.get(condition.operator, "==")
                
                # Special handling for CONTAINS/NOT_CONTAINS
                if condition.operator in [QueryOperator.CONTAINS, QueryOperator.NOT_CONTAINS]:
                    cond = f"@{value_var} {aql_op} doc.{condition.field}"
                else:
                    cond = f"doc.{condition.field} {aql_op} @{value_var}"
                
                conditions.append(cond)
                bind_vars[value_var] = condition.value
            
            # Combine conditions
            combine_op = " && " if query.combine_operator.upper() == "AND" else " || "
            aql += f"FILTER {combine_op.join(conditions)} "
            
        else:
            bind_vars = {}
        
        # Add limit and return
        aql += f"""
        LIMIT @limit
        RETURN {{
            id: doc._key,
            metadata: doc
        }}
        """
        
        bind_vars["limit"] = limit
        
        try:
            # Execute AQL query
            cursor = db.aql.execute(aql, bind_vars=bind_vars)
            
            # Process results
            results: List[Tuple[NodeIDType, Dict[str, Any]]] = []
            for doc in cursor:
                node_id = doc["id"]
                metadata = doc["metadata"]
                
                # Remove embedding from metadata to save memory
                if "embedding" in metadata:
                    del metadata["embedding"]
                
                results.append((node_id, metadata))
            
            return results
        except AQLQueryExecuteError as e:
            logger.error(f"Error executing metadata query: {e}")
            return []
    
    def hybrid_search(
        self,
        query_embedding: EmbeddingArray,
        metadata_query: MetadataQuery,
        k: int = 10,
        vector_weight: float = 0.5
    ) -> List[Tuple[NodeIDType, float, Dict[str, Any]]]:
        """
        Perform hybrid search combining vector similarity and metadata filtering.
        
        Args:
            query_embedding: Query vector
            metadata_query: Metadata query for filtering
            k: Number of results to return
            vector_weight: Weight of vector similarity vs metadata in final score
            
        Returns:
            List of (node_id, score, metadata) tuples
        """
        if self.collection is None:
            raise ValueError("Storage not initialized")
        
        db = self.connection.connect()
        
        # Convert numpy array to list for AQL
        query_vector = query_embedding.tolist()
        
        # Build AQL query
        aql = f"FOR doc IN {self.collection_name} "
        bind_vars = {}
        
        # Add metadata filter conditions
        if metadata_query.conditions:
            conditions = []
            
            for i, condition in enumerate(metadata_query.conditions):
                field_var = f"field_{i}"
                value_var = f"value_{i}"
                
                # Map operator to AQL
                aql_op = {
                    QueryOperator.EQUALS: "==",
                    QueryOperator.NOT_EQUALS: "!=",
                    QueryOperator.GREATER_THAN: ">",
                    QueryOperator.GREATER_EQUAL: ">=",
                    QueryOperator.LESS_THAN: "<",
                    QueryOperator.LESS_EQUAL: "<=",
                    QueryOperator.CONTAINS: "IN",
                    QueryOperator.NOT_CONTAINS: "NOT IN",
                    QueryOperator.IN: "IN",
                    QueryOperator.NOT_IN: "NOT IN",
                    QueryOperator.MATCHES: "=~",  # Regex match
                }.get(condition.operator, "==")
                
                # Special handling for CONTAINS/NOT_CONTAINS
                if condition.operator in [QueryOperator.CONTAINS, QueryOperator.NOT_CONTAINS]:
                    cond = f"@{value_var} {aql_op} doc.{condition.field}"
                else:
                    cond = f"doc.{condition.field} {aql_op} @{value_var}"
                
                conditions.append(cond)
                bind_vars[value_var] = condition.value
            
            # Combine conditions
            combine_op = " && " if metadata_query.combine_operator.upper() == "AND" else " || "
            aql += f"FILTER {combine_op.join(conditions)} "
        
        # Add vector similarity calculation
        aql += """
        LET distance = LENGTH(doc.embedding) == LENGTH(@queryVector) ? 
            1.0 - SUM(
                FOR i IN 0..LENGTH(@queryVector)-1
                RETURN doc.embedding[i] * @queryVector[i]
            ) / (
                SQRT(SUM(FOR i IN doc.embedding RETURN i*i)) * 
                SQRT(SUM(FOR i IN @queryVector RETURN i*i))
            ) : 1.0
            
        LET vectorScore = 1.0 - distance
        
        // Hybrid scoring - combine vector and metadata scores
        LET finalScore = @vectorWeight * vectorScore + (1.0 - @vectorWeight) * 1.0 
        
        SORT finalScore DESC
        LIMIT @k
        
        RETURN {
            id: doc._key,
            score: finalScore,
            metadata: doc
        }
        """
        
        bind_vars["queryVector"] = query_vector
        bind_vars["k"] = k
        bind_vars["vectorWeight"] = vector_weight
        
        try:
            # Execute AQL query
            cursor = db.aql.execute(aql, bind_vars=bind_vars)
            
            # Process results
            results: List[Tuple[NodeIDType, float, Dict[str, Any]]] = []
            for doc in cursor:
                node_id = doc["id"]
                score = doc["score"]
                metadata = doc["metadata"]
                
                # Remove embedding from metadata to save memory
                if "embedding" in metadata:
                    del metadata["embedding"]
                
                results.append((node_id, score, metadata))
            
            return results
        except AQLQueryExecuteError as e:
            logger.error(f"Error executing hybrid search query: {e}")
            return []
    
    def get_stats(self) -> StorageStats:
        """
        Get statistics about the vector storage.
        
        Returns:
            Storage statistics
        """
        if self.collection is None:
            raise ValueError("Storage not initialized")
        
        db = self.connection.connect()
        
        try:
            # Get collection statistics
            collection_stats = self.collection.statistics()
            
            # Get collection count
            count = collection_stats.get("count", 0)
            
            # Get collection figures
            figures = collection_stats.get("figures", {})
            
            # Size information
            size_bytes = figures.get("datafiles", {}).get("fileSize", 0)
            
            # Get index information
            indexes = self.collection.indexes()
            index_stats = {
                idx["name"]: {
                    "type": idx["type"],
                    "fields": idx.get("fields", []),
                    "size": idx.get("size", 0)
                }
                for idx in indexes
            }
            
            # Get query performance stats (if available)
            query_stats: Dict[str, float] = {}
            
            # Check last modified timestamp if available
            last_updated = None
            if "lastModified" in figures:
                try:
                    timestamp = figures["lastModified"]
                    last_updated = datetime.fromtimestamp(timestamp).isoformat()
                except (ValueError, TypeError):
                    pass
            
            return StorageStats(
                storage_type="vector",
                item_count=count,
                storage_size_bytes=size_bytes,
                index_stats=index_stats,
                query_stats=query_stats,
                last_updated=last_updated
            )
            
        except Exception as e:
            logger.error(f"Error getting vector storage stats: {e}")
            return StorageStats(storage_type="vector")


class EnhancedArangoDocumentStorage(ArangoDocumentStorage):
    """
    Enhanced ArangoDB document storage implementation.
    
    This class extends the base ArangoDocumentStorage with additional features
    required by the EnhancedDocumentStorage protocol, including document chunking,
    metadata querying, and bulk operations.
    """
    
    def __init__(
        self,
        connection: ArangoDBConnection,
        collection_name: str = "documents",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        """
        Initialize enhanced document storage.
        
        Args:
            connection: ArangoDB connection
            collection_name: Name of document collection
            chunk_size: Size of text chunks in characters
            chunk_overlap: Overlap between chunks in characters
        """
        super().__init__(connection, collection_name)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def store_document_with_chunks(
        self,
        doc_id: str,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[DocumentChunk]:
        """
        Store a document by splitting it into chunks.
        
        Args:
            doc_id: Document ID
            text: Document text
            metadata: Optional metadata
            
        Returns:
            List of created document chunks
        """
        if self.collection is None:
            raise ValueError("Storage not initialized")
        
        # Generate chunks
        chunks = chunk_text(
            text, 
            chunk_size=self.chunk_size, 
            overlap=self.chunk_overlap
        )
        
        # Store document metadata
        doc_metadata = metadata or {}
        doc_metadata.update({
            "content_length": len(text),
            "chunk_count": len(chunks),
            "created_at": datetime.now().isoformat(),
        })
        
        # Create parent document
        parent_doc = {
            "_key": doc_id,
            "metadata": doc_metadata,
            "is_parent": True,
        }
        
        try:
            self.collection.insert(parent_doc)
        except DocumentInsertError as e:
            if "unique constraint violated" in str(e):
                # Update existing document
                self.collection.update({"_key": doc_id}, parent_doc)
        
        # Store chunks
        result = []
        for i, chunk_text in enumerate(chunks):
            chunk_id = f"{doc_id}_chunk_{i}"
            chunk = DocumentChunk(
                id=chunk_id,
                parent_id=doc_id,
                text=chunk_text,
                chunk_index=i,
                metadata={
                    "parent_id": doc_id,
                    "chunk_index": i,
                    **doc_metadata
                }
            )
            
            # Store the chunk
            chunk_doc = {
                "_key": chunk_id,
                "text": chunk_text,
                "parent_id": doc_id,
                "chunk_index": i,
                "metadata": doc_metadata,
                "is_parent": False,
            }
            
            try:
                self.collection.insert(chunk_doc)
            except DocumentInsertError as e:
                if "unique constraint violated" in str(e):
                    # Update existing chunk
                    self.collection.update({"_key": chunk_id}, chunk_doc)
            
            result.append(chunk)
        
        return result
    
    def bulk_store_documents(
        self,
        documents: List[Tuple[str, str, Optional[Dict[str, Any]]]],
    ) -> BulkOperationResult:
        """
        Store multiple documents in bulk.
        
        Args:
            documents: List of (doc_id, text, metadata) tuples
            
        Returns:
            Result of bulk operation
        """
        if self.collection is None:
            raise ValueError("Storage not initialized")
        
        success_count = 0
        error_count = 0
        errors: Dict[str, str] = {}
        
        for doc_id, text, metadata in documents:
            try:
                self.store_document_with_chunks(doc_id, text, metadata)
                success_count += 1
            except Exception as e:
                error_count += 1
                errors[doc_id] = str(e)
        
        return BulkOperationResult(
            success_count=success_count,
            error_count=error_count,
            errors=errors
        )
    
    def get_document_chunks(
        self,
        doc_id: str,
    ) -> List[DocumentChunk]:
        """
        Get all chunks for a document.
        
        Args:
            doc_id: Document ID
            
        Returns:
            List of document chunks
        """
        if self.collection is None:
            raise ValueError("Storage not initialized")
        
        db = self.connection.connect()
        
        # Build AQL query
        aql = f"""
        FOR doc IN {self.collection_name}
        FILTER doc.parent_id == @doc_id AND doc.is_parent == false
        SORT doc.chunk_index
        RETURN doc
        """
        
        bind_vars = {"doc_id": doc_id}
        
        try:
            # Execute AQL query
            cursor = db.aql.execute(aql, bind_vars=bind_vars)
            
            # Process results
            chunks = []
            for doc in cursor:
                chunk = DocumentChunk(
                    id=doc["_key"],
                    parent_id=doc["parent_id"],
                    text=doc["text"],
                    chunk_index=doc["chunk_index"],
                    metadata=doc.get("metadata", {})
                )
                chunks.append(chunk)
            
            return chunks
        except AQLQueryExecuteError as e:
            logger.error(f"Error getting document chunks: {e}")
            return []
    
    def search_documents(
        self,
        query: str,
        metadata_filter: Optional[MetadataQuery] = None,
        limit: int = 10,
    ) -> List[Tuple[str, str, Dict[str, Any], float]]:
        """
        Search documents by content and metadata.
        
        Args:
            query: Text query
            metadata_filter: Optional metadata filter
            limit: Maximum number of results
            
        Returns:
            List of (doc_id, text, metadata, score) tuples
        """
        if self.collection is None:
            raise ValueError("Storage not initialized")
        
        db = self.connection.connect()
        
        # Build AQL query
        aql = f"FOR doc IN {self.collection_name} "
        bind_vars = {}
        
        # Add full-text search if query is provided
        if query:
            # First check if there's a full-text index
            has_fulltext = False
            for idx in self.collection.indexes():
                if idx["type"] == "fulltext":
                    has_fulltext = True
                    break
            
            if has_fulltext:
                # Use FULLTEXT if index exists
                aql += "FILTER FULLTEXT(doc.text, @query) "
            else:
                # Fallback to LIKE
                aql += "FILTER LIKE(doc.text, @query_pattern, true) "
                bind_vars["query_pattern"] = f"%{query}%"
            
            bind_vars["query"] = query
        
        # Add metadata filter if provided
        if metadata_filter and metadata_filter.conditions:
            conditions = []
            
            for i, condition in enumerate(metadata_filter.conditions):
                field_var = f"field_{i}"
                value_var = f"value_{i}"
                
                # Map operator to AQL
                aql_op = {
                    QueryOperator.EQUALS: "==",
                    QueryOperator.NOT_EQUALS: "!=",
                    QueryOperator.GREATER_THAN: ">",
                    QueryOperator.GREATER_EQUAL: ">=",
                    QueryOperator.LESS_THAN: "<",
                    QueryOperator.LESS_EQUAL: "<=",
                    QueryOperator.CONTAINS: "IN",
                    QueryOperator.NOT_CONTAINS: "NOT IN",
                    QueryOperator.IN: "IN",
                    QueryOperator.NOT_IN: "NOT IN",
                    QueryOperator.MATCHES: "=~",  # Regex match
                }.get(condition.operator, "==")
                
                # Special handling for CONTAINS/NOT_CONTAINS
                if condition.operator in [QueryOperator.CONTAINS, QueryOperator.NOT_CONTAINS]:
                    cond = f"@{value_var} {aql_op} doc.metadata.{condition.field}"
                else:
                    cond = f"doc.metadata.{condition.field} {aql_op} @{value_var}"
                
                conditions.append(cond)
                bind_vars[value_var] = condition.value
            
            # Combine conditions
            combine_op = " && " if metadata_filter.combine_operator.upper() == "AND" else " || "
            filter_clause = combine_op.join(conditions)
            
            # Add to query
            if "FILTER" in aql:
                aql += f"AND {filter_clause} "
            else:
                aql += f"FILTER {filter_clause} "
        
        # Add sorting and limit
        aql += f"""
        SORT doc.is_parent DESC, LENGTH(doc.text) DESC
        LIMIT @limit
        RETURN {{
            id: doc._key,
            text: doc.text,
            metadata: doc.metadata,
            score: 1.0
        }}
        """
        
        bind_vars["limit"] = limit
        
        try:
            # Execute AQL query
            cursor = db.aql.execute(aql, bind_vars=bind_vars)
            
            # Process results
            results = []
            for doc in cursor:
                results.append((
                    doc["id"],
                    doc["text"],
                    doc["metadata"],
                    doc["score"]
                ))
            
            return results
        except AQLQueryExecuteError as e:
            logger.error(f"Error searching documents: {e}")
            return []
    
    def get_stats(self) -> StorageStats:
        """
        Get statistics about the document storage.
        
        Returns:
            Storage statistics
        """
        if self.collection is None:
            raise ValueError("Storage not initialized")
        
        db = self.connection.connect()
        
        try:
            # Get collection statistics
            collection_stats = self.collection.statistics()
            
            # Get collection count
            count = collection_stats.get("count", 0)
            
            # Get collection figures
            figures = collection_stats.get("figures", {})
            
            # Size information
            size_bytes = figures.get("datafiles", {}).get("fileSize", 0)
            
            # Get index information
            indexes = self.collection.indexes()
            index_stats = {
                idx["name"]: {
                    "type": idx["type"],
                    "fields": idx.get("fields", []),
                    "size": idx.get("size", 0)
                }
                for idx in indexes
            }
            
            # Count parent documents vs chunks
            try:
                parent_count_query = f"""
                RETURN LENGTH(
                    FOR doc IN {self.collection_name}
                    FILTER doc.is_parent == true
                    RETURN doc
                )
                """
                parent_count_cursor = db.aql.execute(parent_count_query)
                parent_count = next(parent_count_cursor)
            except Exception:
                parent_count = 0
            
            # Get query performance stats (if available)
            query_stats: Dict[str, float] = {
                "parent_document_count": float(parent_count),
                "chunk_count": float(count - parent_count)
            }
            
            # Check last modified timestamp if available
            last_updated = None
            if "lastModified" in figures:
                try:
                    timestamp = figures["lastModified"]
                    last_updated = datetime.fromtimestamp(timestamp).isoformat()
                except (ValueError, TypeError):
                    pass
            
            return StorageStats(
                storage_type="document",
                item_count=count,
                storage_size_bytes=size_bytes,
                index_stats=index_stats,
                query_stats=query_stats,
                last_updated=last_updated
            )
            
        except Exception as e:
            logger.error(f"Error getting document storage stats: {e}")
            return StorageStats(storage_type="document")


class EnhancedArangoGraphStorage(ArangoGraphStorage):
    """
    Enhanced ArangoDB graph storage implementation.
    
    This class extends the base ArangoGraphStorage with additional features
    required by the EnhancedGraphStorage protocol, including advanced path
    traversal, node and edge operations with metadata, and graph analysis methods.
    """
    
    def create_index_on_node_collection(self, fields: List[str], index_type: str = "persistent") -> Optional[Dict[str, Any]]:
        """
        Create an index on the node collection.
        
        Args:
            fields: List of fields to index
            index_type: Type of index (persistent, hash, etc.)
            
        Returns:
            Index details if successful, None otherwise
        """
        if self.node_collection is None:
            raise ValueError("Storage not initialized")
        
        try:
            if index_type == "fulltext":
                # Full text search index (for text fields)
                return cast(Optional[Dict[str, Any]], self.node_collection.add_fulltext_index(fields=fields))
            elif index_type == "persistent":
                # Persistent index (good for range queries and sorting)
                return cast(Optional[Dict[str, Any]], self.node_collection.add_persistent_index(fields=fields))
            elif index_type == "hash":
                # Hash index (good for equality lookups)
                return cast(Optional[Dict[str, Any]], self.node_collection.add_hash_index(fields=fields))
            else:
                logger.warning(f"Unsupported index type: {index_type}")
                return None
        except Exception as e:
            logger.error(f"Error creating index on node collection: {e}")
            return None
    
    def create_index_on_edge_collection(self, fields: List[str], index_type: str = "persistent") -> Optional[Dict[str, Any]]:
        """
        Create an index on the edge collection.
        
        Args:
            fields: List of fields to index
            index_type: Type of index (persistent, hash, etc.)
            
        Returns:
            Index details if successful, None otherwise
        """
        if self.edge_collection is None:
            raise ValueError("Storage not initialized")
        
        try:
            if index_type == "fulltext":
                # Full text search index (for text fields)
                return cast(Optional[Dict[str, Any]], self.edge_collection.add_fulltext_index(fields=fields))
            elif index_type == "persistent":
                # Persistent index (good for range queries and sorting)
                return cast(Optional[Dict[str, Any]], self.edge_collection.add_persistent_index(fields=fields))
            elif index_type == "hash":
                # Hash index (good for equality lookups)
                return cast(Optional[Dict[str, Any]], self.edge_collection.add_hash_index(fields=fields))
            else:
                logger.warning(f"Unsupported index type: {index_type}")
                return None
        except Exception as e:
            logger.error(f"Error creating index on edge collection: {e}")
            return None
    
    def get_node_with_neighbors(
        self, 
        node_id: str,
        include_edges: bool = True,
        max_depth: int = 1,
        edge_filter: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Get a node with its neighbors up to max_depth away.
        
        Args:
            node_id: ID of the node
            include_edges: Whether to include edge details
            max_depth: Maximum traversal depth
            edge_filter: Optional filter for edge types
            
        Returns:
            Dict with node and its neighborhood
        """
        if self.graph is None or self.node_collection is None or self.edge_collection is None:
            raise ValueError("Storage not initialized")
        
        db = self.connection.connect()
        
        # Get the node itself
        try:
            node = self.get_node(node_id)
            if not node:
                return {}
        except Exception as e:
            logger.error(f"Error getting node {node_id}: {e}")
            return {}
        
        # Initialize result with proper type-safe collections
        result = {
            "node": node,
            "neighbors": [],  # Always a list
            "edges": [] if include_edges else None  # May be None if edges not requested
        }
        
        # Build AQL traversal query
        direction = "ANY"  # Could be OUTBOUND, INBOUND, or ANY
        
        # Edge filter condition
        edge_filter_condition = ""
        bind_vars = {"start_id": node_id, "max_depth": max_depth}
        
        if edge_filter:
            conditions = []
            for i, (key, value) in enumerate(edge_filter.items()):
                value_var = f"value_{i}"
                conditions.append(f"e.{key} == @{value_var}")
                bind_vars[value_var] = value
            
            if conditions:
                edge_filter_condition = f"FILTER {' && '.join(conditions)}"
        
        aql = f"""
        FOR v, e, p IN 1..@max_depth {direction} @start_id
            GRAPH '{self.graph_name}'
            {edge_filter_condition}
            RETURN {{
                "vertex": v,
                "edge": e,
                "path": p.vertices[*]._key
            }}
        """
        
        try:
            # Execute traversal query
            cursor = db.aql.execute(aql, bind_vars=bind_vars)
            neighbors = set()
            edges = set()
            
            for item in cursor:
                vertex = item["vertex"]
                edge = item["edge"]
                
                # Add neighbor if not the start node
                if vertex["_key"] != node_id:
                    neighbor_id = vertex["_key"]
                    if neighbor_id not in neighbors:
                        neighbors.add(neighbor_id)
                        # Safe append to neighbors list
                        if result["neighbors"] is not None:
                            result["neighbors"].append(vertex)
                
                # Add edge if requested
                if include_edges and edge:
                    edge_id = edge["_key"]
                    if edge_id not in edges:
                        edges.add(edge_id)
                        # Safe append to edges list with null check
                        if result["edges"] is not None:
                            result["edges"].append(edge)
            
            return result
                
        except AQLQueryExecuteError as e:
            logger.error(f"Error executing traversal query: {e}")
            return {"node": node, "neighbors": [], "edges": [] if include_edges else None}
    
    def get_shortest_paths(
        self,
        start_id: str,
        end_id: str,
        max_length: int = 10,
        edge_filter: Optional[Dict[str, Any]] = None,
        max_paths: int = 5
    ) -> List[List[Dict[str, Any]]]:
        """
        Find shortest paths between two nodes.
        
        Args:
            start_id: Starting node ID
            end_id: Ending node ID
            max_length: Maximum path length
            edge_filter: Optional filter for edge types
            max_paths: Maximum number of paths to return
            
        Returns:
            List of paths, where each path is a list of nodes
        """
        if self.graph is None:
            raise ValueError("Storage not initialized")
        
        db = self.connection.connect()
        
        # Build AQL shortest path query
        direction = "ANY"  # Could be OUTBOUND, INBOUND, or ANY
        
        # Edge filter condition
        edge_filter_condition = ""
        bind_vars = {"start_id": start_id, "end_id": end_id}
        
        if edge_filter:
            conditions = []
            for i, (key, value) in enumerate(edge_filter.items()):
                value_var = f"value_{i}"
                conditions.append(f"e.{key} == @{value_var}")
                bind_vars[value_var] = value
            
            if conditions:
                edge_filter_condition = f"OPTIONS {{filter: CURRENT => {' && '.join(conditions)}}}" 
        
        # First try k-shortest paths algorithm
        aql = f"""
        FOR p IN K_SHORTEST_PATHS @start_id TO @end_id
            {direction}
            GRAPH '{self.graph_name}'
            {edge_filter_condition}
            OPTIONS {{weightAttribute: 'weight', defaultWeight: 1}}
            LIMIT @max_paths
            RETURN {{
                "vertices": p.vertices,
                "edges": p.edges,
                "weight": p.weight
            }}
        """
        
        bind_vars["max_paths"] = max_paths
        
        try:
            # Execute shortest path query
            cursor = db.aql.execute(aql, bind_vars=bind_vars)
            
            paths = []
            for item in cursor:
                vertices = item["vertices"]
                edges = item["edges"]
                
                if len(vertices) > max_length + 1:  # +1 because path includes start and end nodes
                    continue
                
                # Create path with both nodes and edges
                path = []
                for i, vertex in enumerate(vertices):
                    path.append({"type": "node", "data": vertex})
                    if i < len(edges):
                        path.append({"type": "edge", "data": edges[i]})
                
                paths.append(path)
            
            return paths
                
        except AQLQueryExecuteError as e:
            logger.error(f"Error executing shortest path query: {e}")
            
            # Fallback to DFS traversal if K_SHORTEST_PATHS fails
            try:
                fallback_aql = f"""
                FOR v, e, p IN 1..{max_length} {direction} @start_id
                    GRAPH '{self.graph_name}'
                    {edge_filter_condition}
                    FILTER v._key == @end_id
                    LIMIT @max_paths
                    RETURN {{
                        "vertices": p.vertices,
                        "edges": p.edges
                    }}
                """
                
                cursor = db.aql.execute(fallback_aql, bind_vars=bind_vars)
                
                paths = []
                for item in cursor:
                    vertices = item["vertices"]
                    edges = item["edges"]
                    
                    # Create path with both nodes and edges
                    path = []
                    for i, vertex in enumerate(vertices):
                        path.append({"type": "node", "data": vertex})
                        if i < len(edges):
                            path.append({"type": "edge", "data": edges[i]})
                    
                    paths.append(path)
                
                return paths
            except AQLQueryExecuteError as e2:
                logger.error(f"Error executing fallback traversal query: {e2}")
                return []
    
    def query_nodes_by_metadata(
        self,
        metadata_query: MetadataQuery,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Find nodes by metadata query.
        
        Args:
            metadata_query: Metadata query to filter by
            limit: Maximum number of results
            
        Returns:
            List of matching nodes
        """
        if self.node_collection is None:
            raise ValueError("Storage not initialized")
        
        db = self.connection.connect()
        
        # Build AQL query
        aql = f"FOR doc IN {self.node_collection.name} "
        
        # Add filter conditions
        if metadata_query.conditions:
            conditions = []
            bind_vars = {}
            
            for i, condition in enumerate(metadata_query.conditions):
                field_var = f"field_{i}"
                value_var = f"value_{i}"
                
                # Map operator to AQL
                aql_op = {
                    QueryOperator.EQUALS: "==",
                    QueryOperator.NOT_EQUALS: "!=",
                    QueryOperator.GREATER_THAN: ">",
                    QueryOperator.GREATER_EQUAL: ">=",
                    QueryOperator.LESS_THAN: "<",
                    QueryOperator.LESS_EQUAL: "<=",
                    QueryOperator.CONTAINS: "IN",
                    QueryOperator.NOT_CONTAINS: "NOT IN",
                    QueryOperator.IN: "IN",
                    QueryOperator.NOT_IN: "NOT IN",
                    QueryOperator.MATCHES: "=~",  # Regex match
                }.get(condition.operator, "==")
                
                # Special handling for CONTAINS/NOT_CONTAINS
                if condition.operator in [QueryOperator.CONTAINS, QueryOperator.NOT_CONTAINS]:
                    cond = f"@{value_var} {aql_op} doc.{condition.field}"
                else:
                    cond = f"doc.{condition.field} {aql_op} @{value_var}"
                
                conditions.append(cond)
                bind_vars[value_var] = condition.value
            
            # Combine conditions
            combine_op = " && " if metadata_query.combine_operator.upper() == "AND" else " || "
            aql += f"FILTER {combine_op.join(conditions)} "
            
        else:
            bind_vars = {}
        
        # Add limit and return
        aql += f"""
        LIMIT @limit
        RETURN doc
        """
        
        bind_vars["limit"] = limit
        
        try:
            # Execute AQL query
            cursor = db.aql.execute(aql, bind_vars=bind_vars)
            
            # Process results
            return [doc for doc in cursor]
        except AQLQueryExecuteError as e:
            logger.error(f"Error executing node metadata query: {e}")
            return []
    
    def get_stats(self) -> StorageStats:
        """
        Get statistics about the graph storage.
        
        Returns:
            Storage statistics
        """
        if self.graph is None or self.node_collection is None or self.edge_collection is None:
            raise ValueError("Storage not initialized")
        
        db = self.connection.connect()
        
        try:
            # Get node collection statistics
            node_stats = self.node_collection.statistics()
            node_count = node_stats.get("count", 0)
            
            # Get edge collection statistics
            edge_stats = self.edge_collection.statistics()
            edge_count = edge_stats.get("count", 0)
            
            # Get size information
            node_size = node_stats.get("figures", {}).get("datafiles", {}).get("fileSize", 0)
            edge_size = edge_stats.get("figures", {}).get("datafiles", {}).get("fileSize", 0)
            total_size = node_size + edge_size
            
            # Get index information
            node_indexes = self.node_collection.indexes()
            edge_indexes = self.edge_collection.indexes()
            
            index_stats = {}
            for idx in node_indexes:
                index_stats[f"node_{idx['name']}"] = {
                    "type": idx["type"],
                    "fields": idx.get("fields", []),
                    "size": idx.get("size", 0)
                }
            
            for idx in edge_indexes:
                index_stats[f"edge_{idx['name']}"] = {
                    "type": idx["type"],
                    "fields": idx.get("fields", []),
                    "size": idx.get("size", 0)
                }
            
            # Get graph information
            query_stats: Dict[str, float] = {
                "node_count": float(node_count),
                "edge_count": float(edge_count),
                "density": float(edge_count / max(node_count * (node_count - 1), 1) if node_count > 1 else 0)
            }
            
            # Check last modified timestamp
            node_last_modified = node_stats.get("figures", {}).get("lastModified")
            edge_last_modified = edge_stats.get("figures", {}).get("lastModified")
            
            last_updated = None
            if node_last_modified and edge_last_modified:
                last_timestamp = max(node_last_modified, edge_last_modified)
                try:
                    last_updated = datetime.fromtimestamp(last_timestamp).isoformat()
                except (ValueError, TypeError):
                    pass
            
            return StorageStats(
                storage_type="graph",
                item_count=node_count + edge_count,
                storage_size_bytes=total_size,
                index_stats=index_stats,
                query_stats=query_stats,
                last_updated=last_updated
            )
            
        except Exception as e:
            logger.error(f"Error getting graph storage stats: {e}")
            return StorageStats(storage_type="graph")

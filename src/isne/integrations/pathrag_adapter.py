"""
PathRAG adapter for the ISNE pipeline.

This module provides an adapter for integrating the ISNE pipeline with
the PathRAG storage and query components.
"""

import os
import uuid
import logging
from typing import Dict, List, Any, Optional, Union, Tuple, cast
from datetime import datetime
import numpy as np

from src.types.common import NodeID, EdgeID, NodeData, EdgeData, EmbeddingVector
from src.isne.types.models import (
    IngestDocument, 
    IngestDataset, 
    DocumentRelation, 
    RelationType
)
from src.pathrag.base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PathRAGISNEAdapter:
    """
    Adapter for integrating ISNE with PathRAG storage.
    
    This adapter provides conversion between ISNE data models and PathRAG
    storage components, enabling the use of ISNE embeddings with PathRAG queries.
    """
    
    def __init__(
        self,
        graph_storage: Optional[BaseGraphStorage] = None,
        kv_storage: Optional[BaseKVStorage] = None,
        vector_storage: Optional[BaseVectorStorage] = None,
        namespace: str = "isne"
    ) -> None:
        """
        Initialize the PathRAG adapter for ISNE.
        
        Args:
            graph_storage: PathRAG graph storage component
            kv_storage: PathRAG key-value storage component
            vector_storage: PathRAG vector storage component
            namespace: Namespace for storage
        """
        self.graph_storage = graph_storage
        self.kv_storage = kv_storage
        self.vector_storage = vector_storage
        self.namespace = namespace
    
    async def store_document(self, document: IngestDocument) -> str:
        """
        Store an ISNE document in PathRAG storage.
        
        Args:
            document: ISNE document to store
            
        Returns:
            Document ID
        """
        # Convert document to PathRAG format
        node_data = self._convert_document_to_node(document)
        
        # Store in graph storage if available
        if self.graph_storage:
            await self.graph_storage.upsert_node(document.id, node_data)
        
        # Store in KV storage if available
        if self.kv_storage:
            await self.kv_storage.upsert({document.id: node_data})
        
        # Store in vector storage if available and document has embedding
        if self.vector_storage and document.embedding is not None:
            # Create document with embedding and metadata for vector storage
            vector_data = {
                document.id: {
                    "content": document.content,
                    "embedding": document.embedding,
                    "metadata": document.metadata,
                    "title": document.title,
                    "id": document.id
                }
            }
            await self.vector_storage.upsert(vector_data)
        
        logger.debug(f"Stored document: {document.id}")
        return document.id
    
    async def store_relation(self, relation: DocumentRelation) -> str:
        """
        Store an ISNE relation in PathRAG storage.
        
        Args:
            relation: ISNE relation to store
            
        Returns:
            Relation ID
        """
        # Convert relation to PathRAG format
        edge_data = self._convert_relation_to_edge(relation)
        
        # Store in graph storage if available
        if self.graph_storage:
            edge_id = await self.graph_storage.upsert_edge(
                relation.source_id, 
                relation.target_id, 
                edge_data
            )
            
            # If relation is bidirectional, create reverse edge
            if relation.bidirectional:
                # Create reverse edge data (only change source/target)
                reverse_edge_data = edge_data.copy()
                
                # Add bidirectional marker to metadata
                if "metadata" not in reverse_edge_data:
                    reverse_edge_data["metadata"] = {}
                reverse_edge_data["metadata"]["is_reverse_edge"] = True
                
                await self.graph_storage.upsert_edge(
                    relation.target_id,
                    relation.source_id,
                    reverse_edge_data
                )
        
        logger.debug(f"Stored relation: {relation.id}")
        return relation.id
    
    async def store_dataset(self, dataset: IngestDataset) -> str:
        """
        Store an entire ISNE dataset in PathRAG storage.
        
        Args:
            dataset: ISNE dataset to store
            
        Returns:
            Dataset ID
        """
        # Store dataset metadata in KV storage
        if self.kv_storage:
            metadata = {
                "id": dataset.id,
                "name": dataset.name,
                "description": dataset.description,
                "created_at": dataset.created_at.isoformat() if dataset.created_at else None,
                "updated_at": datetime.now().isoformat(),
                "document_count": len(dataset.documents),
                "relation_count": len(dataset.relations),
                "type": "isne_dataset",
                "metadata": dataset.metadata
            }
            
            dataset_key = f"{self.namespace}_dataset_{dataset.id}"
            await self.kv_storage.upsert({dataset_key: metadata})
        
        # Store all documents
        for doc_id, document in dataset.documents.items():
            await self.store_document(document)
        
        # Store all relations
        for relation in dataset.relations:
            await self.store_relation(relation)
        
        logger.info(f"Stored dataset {dataset.name} with {len(dataset.documents)} documents and {len(dataset.relations)} relations")
        return dataset.id
    
    async def load_document(self, document_id: str) -> Optional[IngestDocument]:
        """
        Load an ISNE document from PathRAG storage.
        
        Args:
            document_id: ID of the document to load
            
        Returns:
            Loaded IngestDocument or None if not found
        """
        node_data = None
        
        # Try to get from graph storage first
        if self.graph_storage:
            node_data = await self.graph_storage.get_node(document_id)
        
        # If not found, try KV storage
        if not node_data and self.kv_storage:
            node_data = await self.kv_storage.get_by_id(document_id)
        
        if not node_data:
            logger.warning(f"Document not found: {document_id}")
            return None
        
        # Convert to ISNE document
        return self._convert_node_to_document(node_data)
    
    async def load_relation(self, source_id: str, target_id: str) -> Optional[DocumentRelation]:
        """
        Load an ISNE relation from PathRAG storage.
        
        Args:
            source_id: Source document ID
            target_id: Target document ID
            
        Returns:
            Loaded DocumentRelation or None if not found
        """
        # Try to get from graph storage
        if not self.graph_storage:
            logger.warning("No graph storage available")
            return None
        
        edge_data = await self.graph_storage.get_edge(source_id, target_id)
        
        if not edge_data:
            logger.warning(f"Relation not found: {source_id} -> {target_id}")
            return None
        
        # Convert to ISNE relation
        return self._convert_edge_to_relation(edge_data, source_id, target_id)
    
    async def get_related_documents(
        self, 
        document_id: str, 
        relation_type: Optional[RelationType] = None,
        max_distance: int = 1
    ) -> List[Tuple[IngestDocument, List[DocumentRelation]]]:
        """
        Get documents related to a source document.
        
        Args:
            document_id: Source document ID
            relation_type: Optional relation type to filter by
            max_distance: Maximum path distance
            
        Returns:
            List of (document, path_relations) tuples
        """
        if not self.graph_storage:
            logger.warning("No graph storage available")
            return []
        
        # Filter by relation type if specified
        edge_filter = None
        if relation_type:
            edge_filter = lambda edge: edge.get("relation_type") == relation_type.value
        
        # Get related nodes with paths
        related_nodes = []
        
        # For simplicity, we'll just get direct neighbors here
        # A more sophisticated traversal would be implemented for multi-hop paths
        edges = await self.graph_storage.get_node_out_edges(document_id)
        
        for edge in edges:
            # Apply filter if needed
            if edge_filter and not edge_filter(edge):
                continue
                
            target_id = edge.get("target") or edge.get("_to").split("/")[-1]
            
            # Get target node
            target_node = await self.graph_storage.get_node(target_id)
            
            if target_node:
                # Convert to ISNE document
                document = self._convert_node_to_document(target_node)
                
                if document:
                    # Convert edge to relation
                    relation = self._convert_edge_to_relation(edge, document_id, target_id)
                    
                    if relation:
                        related_nodes.append((document, [relation]))
        
        return related_nodes
    
    async def search_similar_documents(
        self, 
        query_embedding: EmbeddingVector,
        limit: int = 10,
        min_score: float = 0.7,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[IngestDocument, float]]:
        """
        Search for documents with similar embeddings.
        
        Args:
            query_embedding: Query embedding vector
            limit: Maximum number of results
            min_score: Minimum similarity score (0-1)
            filter_metadata: Optional metadata filter
            
        Returns:
            List of (document, score) tuples
        """
        if not self.vector_storage:
            logger.warning("No vector storage available")
            return []
        
        # Convert numpy array to list if needed
        if isinstance(query_embedding, np.ndarray):
            query_embedding = query_embedding.tolist()
        
        # Perform vector search
        results = await self.vector_storage.search(
            query_embedding,
            k=limit,
            min_score=min_score,
            filter_func=filter_metadata
        )
        
        # Convert results to ISNE documents
        document_results = []
        
        for result in results:
            doc_id = result["id"]
            score = result["score"]
            
            # Load full document
            document = await self.load_document(doc_id)
            
            if document:
                document_results.append((document, score))
        
        return document_results
    
    def _convert_document_to_node(self, document: IngestDocument) -> Dict[str, Any]:
        """
        Convert an ISNE document to PathRAG node format.
        
        Args:
            document: ISNE document to convert
            
        Returns:
            PathRAG node data
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
        
        # Create node data
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
            "tags": document.tags,
            "type": "isne_document"
        }
    
    def _convert_node_to_document(self, node_data: Dict[str, Any]) -> Optional[IngestDocument]:
        """
        Convert PathRAG node data to ISNE document.
        
        Args:
            node_data: PathRAG node data
            
        Returns:
            ISNE document
        """
        try:
            # Parse timestamps
            created_at = None
            if node_data.get("created_at"):
                try:
                    created_at = datetime.fromisoformat(node_data["created_at"])
                except (ValueError, TypeError):
                    pass
            
            updated_at = None
            if node_data.get("updated_at"):
                try:
                    updated_at = datetime.fromisoformat(node_data["updated_at"])
                except (ValueError, TypeError):
                    pass
            
            # Create ISNE document
            return IngestDocument(
                id=node_data.get("id") or node_data.get("_key", str(uuid.uuid4())),
                content=node_data["content"],
                source=node_data.get("source", "unknown"),
                document_type=node_data.get("document_type", "unknown"),
                title=node_data.get("title"),
                author=node_data.get("author"),
                created_at=created_at,
                updated_at=updated_at,
                metadata=node_data.get("metadata", {}),
                embedding=node_data.get("embedding"),
                embedding_model=node_data.get("embedding_model"),
                chunks=node_data.get("chunks", []),
                tags=node_data.get("tags", [])
            )
        except KeyError as e:
            logger.error(f"Error converting node data to ISNE document: {e}")
            return None
    
    def _convert_relation_to_edge(self, relation: DocumentRelation) -> Dict[str, Any]:
        """
        Convert an ISNE relation to PathRAG edge format.
        
        Args:
            relation: ISNE relation to convert
            
        Returns:
            PathRAG edge data
        """
        # Format timestamp
        created_at = relation.created_at.isoformat() if relation.created_at else None
        
        # Create edge data
        return {
            "id": relation.id,
            "source": relation.source_id,
            "target": relation.target_id,
            "relation_type": relation.relation_type.value,
            "weight": relation.weight,
            "bidirectional": relation.bidirectional,
            "created_at": created_at,
            "metadata": relation.metadata,
            "type": "isne_relation"
        }
    
    def _convert_edge_to_relation(
        self, 
        edge_data: Dict[str, Any],
        source_id: str,
        target_id: str
    ) -> Optional[DocumentRelation]:
        """
        Convert PathRAG edge data to ISNE relation.
        
        Args:
            edge_data: PathRAG edge data
            source_id: Source document ID
            target_id: Target document ID
            
        Returns:
            ISNE relation
        """
        try:
            # Get source and target IDs from edge data if available
            source_id = edge_data.get("source", edge_data.get("_from", source_id))
            target_id = edge_data.get("target", edge_data.get("_to", target_id))
            
            # If IDs include collection name, extract just the key
            if isinstance(source_id, str) and "/" in source_id:
                source_id = source_id.split("/")[-1]
            if isinstance(target_id, str) and "/" in target_id:
                target_id = target_id.split("/")[-1]
            
            # Parse timestamp
            created_at = None
            if edge_data.get("created_at"):
                try:
                    created_at = datetime.fromisoformat(edge_data["created_at"])
                except (ValueError, TypeError):
                    pass
            
            # Convert relation type string to enum
            relation_type_str = edge_data.get("relation_type", "related_to")
            try:
                relation_type = RelationType(relation_type_str)
            except ValueError:
                relation_type = RelationType.RELATED_TO
            
            # Create ISNE relation
            return DocumentRelation(
                id=edge_data.get("id") or edge_data.get("_key", str(uuid.uuid4())),
                source_id=source_id,
                target_id=target_id,
                relation_type=relation_type,
                weight=edge_data.get("weight", 1.0),
                bidirectional=edge_data.get("bidirectional", False),
                metadata=edge_data.get("metadata", {}),
                created_at=created_at
            )
        except Exception as e:
            logger.error(f"Error converting edge data to ISNE relation: {e}")
            return None

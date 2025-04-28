"""
ISNE integration with PathRAG algorithm.

This module connects the ISNE pipeline with the PathRAG algorithm for ranking paths
in code and document repositories based on semantic relevance, path length, and edge strength.
"""

import logging
import time
from typing import Dict, List, Any, Optional, Union, cast
from pathlib import Path

import numpy as np

from src.isne.path_ranking import PathRanker
from src.isne.types.models import IngestDocument, IngestDataset, DocumentRelation, EmbeddingConfig
from src.isne.pipeline import ISNEPipeline, PipelineConfig
from src.ingest.repository.arango_repository import ArangoRepository
from src.types.common import PathRankingConfig, NodeData, EdgeData

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PathRAGConnector:
    """
    Connector between ISNE pipeline and PathRAG algorithm.
    
    This class provides methods to retrieve documents and relations from the repository,
    process them with the ISNE pipeline, and rank paths using the PathRAG algorithm.
    """
    
    def __init__(
        self,
        repository: Optional[ArangoRepository] = None,
        isne_pipeline: Optional[ISNEPipeline] = None,
        path_ranking_config: Optional[PathRankingConfig] = None
    ) -> None:
        """
        Initialize the PathRAG connector.
        
        Args:
            repository: ArangoDB repository
            isne_pipeline: ISNE pipeline
            path_ranking_config: Path ranking configuration
        """
        self.repository = repository
        self.isne_pipeline = isne_pipeline
        self.path_ranker = PathRanker.from_config(path_ranking_config)
        
        # Initialize components if not provided
        if self.isne_pipeline is None:
            logger.info("Initializing default ISNE pipeline")
            self.isne_pipeline = ISNEPipeline()
        
        # Track metrics
        self.metrics: Dict[str, Any] = {
            "queries_processed": 0,
            "paths_found": 0,
            "avg_ranking_time": 0.0,
            "total_ranking_time": 0.0
        }
    
    def retrieve_documents(
        self, 
        query: str,
        limit: int = 100,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[IngestDocument]:
        """
        Retrieve relevant documents from the repository.
        
        Args:
            query: Search query
            limit: Maximum number of documents to retrieve
            filters: Filters to apply to the search
            
        Returns:
            List of retrieved documents
        """
        if self.repository is None:
            logger.error("Repository not initialized")
            return []
        
        try:
            # First get embeddings for the query using the ISNE pipeline
            if not self.isne_pipeline or not self.isne_pipeline.embedding_processor:
                logger.error("ISNE embedding processor not initialized")
                return []
            
            # Initialize the embedding processor if not already done
            if not self.isne_pipeline.embedding_processor.initialized:
                self.isne_pipeline.embedding_processor._initialize_model()
            
            # Generate embedding for the query
            query_embedding = None
            if self.isne_pipeline.embedding_processor.embedding_fn:
                try:
                    embedding_list = self.isne_pipeline.embedding_processor.embedding_fn([query])
                    query_embedding = embedding_list[0] if embedding_list else None
                except Exception as e:
                    logger.error(f"Error generating query embedding: {e}")
            
            if query_embedding is None:
                logger.error("Failed to generate query embedding")
                return []
            
            # Search for similar documents using vector search
            similar_nodes = self.repository.search_vectors(
                query_embedding,
                limit=limit,
                filters=filters
            )
            
            # Convert to IngestDocument objects
            documents: List[IngestDocument] = []
            for node in similar_nodes:
                try:
                    doc = IngestDocument(
                        id=node.get('id', str(len(documents))),
                        content=node.get('content', ''),
                        source=node.get('source', ''),
                        document_type=node.get('type', 'unknown'),
                        title=node.get('title'),
                        embedding=node.get('embedding'),
                        metadata=node.get('metadata', {})
                    )
                    documents.append(doc)
                except Exception as e:
                    logger.error(f"Error converting node to document: {e}")
            
            return documents
            
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            return []
    
    def retrieve_relations(
        self,
        document_ids: List[str]
    ) -> List[DocumentRelation]:
        """
        Retrieve relations for the given document IDs.
        
        Args:
            document_ids: List of document IDs
            
        Returns:
            List of relations between the documents
        """
        if self.repository is None:
            logger.error("Repository not initialized")
            return []
        
        relations: List[DocumentRelation] = []
        
        try:
            # Retrieve edges for each document
            for doc_id in document_ids:
                # Get outgoing edges
                out_edges = self.repository.get_edges(doc_id, direction="outbound")
                
                # Convert to DocumentRelation objects
                for edge in out_edges:
                    try:
                        source_id = edge.get('source_id', '')
                        target_id = edge.get('target_id', '')
                        
                        # Skip if source or target is not in the document list
                        if source_id not in document_ids or target_id not in document_ids:
                            continue
                        
                        # Create relation
                        relation = DocumentRelation(
                            source_id=source_id,
                            target_id=target_id,
                            relation_type=edge.get('type', 'relates_to'),
                            weight=float(edge.get('weight', 1.0)),
                            bidirectional=bool(edge.get('bidirectional', False)),
                            metadata=edge.get('metadata', {})
                        )
                        relations.append(relation)
                    except Exception as e:
                        logger.error(f"Error converting edge to relation: {e}")
            
            return relations
            
        except Exception as e:
            logger.error(f"Error retrieving relations: {e}")
            return []
    
    def rank_paths(
        self,
        query: str,
        source_id: Optional[str] = None,
        target_ids: Optional[List[str]] = None,
        document_limit: int = 100,
        path_limit: int = 20,
        filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Find and rank paths between documents based on the query.
        
        Args:
            query: Search query
            source_id: Optional source document ID
            target_ids: Optional list of target document IDs
            document_limit: Maximum number of documents to retrieve
            path_limit: Maximum number of paths to return
            filters: Filters to apply to the document search
            
        Returns:
            Dictionary containing ranked paths and metrics
        """
        start_time = time.time()
        
        try:
            # Track query
            self.metrics["queries_processed"] += 1
            
            # 1. Generate query embedding
            query_embedding = None
            if self.isne_pipeline and self.isne_pipeline.embedding_processor and self.isne_pipeline.embedding_processor.embedding_fn:
                try:
                    # Initialize the embedding processor if not already done
                    if not self.isne_pipeline.embedding_processor.initialized:
                        self.isne_pipeline.embedding_processor._initialize_model()
                    
                    embedding_list = self.isne_pipeline.embedding_processor.embedding_fn([query])
                    query_embedding = embedding_list[0] if embedding_list else None
                except Exception as e:
                    logger.error(f"Error generating query embedding: {e}")
            
            if query_embedding is None:
                logger.error("Failed to generate query embedding")
                return {
                    "error": "Failed to generate query embedding",
                    "paths": [],
                    "metrics": {
                        "query_time": time.time() - start_time,
                        "documents_retrieved": 0,
                        "relations_retrieved": 0,
                        "paths_found": 0
                    }
                }
            
            # 2. Retrieve relevant documents
            documents = self.retrieve_documents(query, limit=document_limit, filters=filters)
            if not documents:
                logger.warning("No documents retrieved")
                return {
                    "paths": [],
                    "metrics": {
                        "query_time": time.time() - start_time,
                        "documents_retrieved": 0,
                        "relations_retrieved": 0,
                        "paths_found": 0
                    }
                }
            
            # 3. Retrieve relations
            document_ids = [doc.id for doc in documents]
            relations = self.retrieve_relations(document_ids)
            
            # 4. Rank paths
            ranked_paths = self.path_ranker.find_and_rank_paths(
                query_embedding=query_embedding,
                documents=documents,
                relations=relations,
                source_id=source_id,
                target_ids=target_ids,
                top_k=path_limit
            )
            
            # 5. Track metrics
            end_time = time.time()
            query_time = end_time - start_time
            self.metrics["paths_found"] += len(ranked_paths)
            self.metrics["total_ranking_time"] += query_time
            self.metrics["avg_ranking_time"] = (
                self.metrics["total_ranking_time"] / self.metrics["queries_processed"]
            )
            
            # 6. Return results
            return {
                "paths": ranked_paths,
                "metrics": {
                    "query_time": query_time,
                    "documents_retrieved": len(documents),
                    "relations_retrieved": len(relations),
                    "paths_found": len(ranked_paths)
                }
            }
            
        except Exception as e:
            logger.error(f"Error ranking paths: {e}")
            return {
                "error": str(e),
                "paths": [],
                "metrics": {
                    "query_time": time.time() - start_time,
                    "documents_retrieved": 0,
                    "relations_retrieved": 0,
                    "paths_found": 0
                }
            }
    
    def optimize_code_embeddings(
        self,
        documents: List[IngestDocument],
        code_model_name: Optional[str] = None
    ) -> List[IngestDocument]:
        """
        Optimize embeddings for code documents using specialized code models.
        
        Args:
            documents: List of documents
            code_model_name: Name of specialized code model to use
            
        Returns:
            List of documents with optimized embeddings
        """
        if not self.isne_pipeline or not self.isne_pipeline.embedding_processor:
            logger.error("ISNE embedding processor not initialized")
            return documents
        
        # Set default code model if not provided
        if code_model_name is None:
            code_model_name = "microsoft/codebert-base"
        
        try:
            # Initialize specialized embedding processor for code
            code_embedding_config = EmbeddingConfig(
                model_name=code_model_name,
                model_dimension=768,  # Typical for CodeBERT
                batch_size=16,
                use_gpu=True,
                normalize_embeddings=True
            )
            
            # Use current processor for non-code documents
            optimized_docs = []
            code_docs = []
            non_code_docs = []
            
            # Separate code and non-code documents
            for doc in documents:
                if doc.document_type in ["code", "python", "javascript", "java", "cpp", "c", "go", "rust"]:
                    code_docs.append(doc)
                else:
                    non_code_docs.append(doc)
            
            # Process specialized code embeddings
            if code_docs:
                from src.isne.processors.embedding_processor import EmbeddingProcessor
                code_processor = EmbeddingProcessor(embedding_config=code_embedding_config)
                code_processor._initialize_model()
                
                for doc in code_docs:
                    if doc.embedding is None and doc.content:
                        try:
                            embeddings = code_processor.embedding_fn([doc.content])
                            if embeddings and len(embeddings) > 0:
                                doc.embedding = embeddings[0]
                                doc.embedding_model = code_model_name
                        except Exception as e:
                            logger.error(f"Error generating code embedding: {e}")
                    
                    optimized_docs.append(doc)
            
            # Add non-code documents
            optimized_docs.extend(non_code_docs)
            
            return optimized_docs
            
        except Exception as e:
            logger.error(f"Error optimizing code embeddings: {e}")
            return documents
    
    def specialize_relationship_embeddings(
        self,
        relations: List[DocumentRelation],
        documents: List[IngestDocument]
    ) -> List[DocumentRelation]:
        """
        Create specialized embeddings for relationships that consider both endpoints.
        
        Args:
            relations: List of document relations
            documents: List of documents
            
        Returns:
            List of relations with specialized embeddings
        """
        if not self.isne_pipeline or not self.isne_pipeline.embedding_processor:
            logger.error("ISNE embedding processor not initialized")
            return relations
        
        try:
            # Create document lookup by ID
            doc_lookup = {doc.id: doc for doc in documents}
            
            # Process each relation
            for rel in relations:
                source_doc = doc_lookup.get(rel.source_id)
                target_doc = doc_lookup.get(rel.target_id)
                
                if source_doc is None or target_doc is None:
                    continue
                
                # Skip if either document doesn't have an embedding
                if source_doc.embedding is None or target_doc.embedding is None:
                    continue
                
                # Create specialized relationship embedding by combining source and target
                source_emb = np.array(source_doc.embedding)
                target_emb = np.array(target_doc.embedding)
                
                # Combine embeddings (weighted average based on relationship type)
                # Different relationship types get different weightings
                if rel.relation_type == "contains":
                    # Container has more influence (0.7) than contained (0.3)
                    combined_emb = 0.7 * source_emb + 0.3 * target_emb
                elif rel.relation_type == "references":
                    # Referenced has more influence (0.6) than referencer (0.4)
                    combined_emb = 0.4 * source_emb + 0.6 * target_emb
                elif rel.relation_type == "implements":
                    # Implementation has more influence (0.7) than interface (0.3)
                    combined_emb = 0.7 * source_emb + 0.3 * target_emb
                else:
                    # Equal influence for other relationships
                    combined_emb = 0.5 * source_emb + 0.5 * target_emb
                
                # Normalize
                norm = np.linalg.norm(combined_emb)
                if norm > 0:
                    combined_emb = combined_emb / norm
                
                # Store in relation metadata
                rel.metadata["specialized_embedding"] = combined_emb.tolist()
                rel.metadata["specialized_embedding_source"] = "combined"
            
            return relations
            
        except Exception as e:
            logger.error(f"Error specializing relationship embeddings: {e}")
            return relations
    
    def process_repository(
        self,
        repo_path: Union[str, Path],
        ranking_config: Optional[PathRankingConfig] = None
    ) -> Dict[str, Any]:
        """
        Process a repository and prepare it for path ranking.
        
        Args:
            repo_path: Path to the repository
            ranking_config: Path ranking configuration
            
        Returns:
            Processing results and statistics
        """
        if not self.isne_pipeline:
            logger.error("ISNE pipeline not initialized")
            return {"error": "ISNE pipeline not initialized"}
        
        try:
            # 1. Process repository with ISNE pipeline
            logger.info(f"Processing repository: {repo_path}")
            dataset = self.isne_pipeline.process_repository(repo_path)
            
            if not dataset:
                logger.error("Failed to process repository")
                return {"error": "Failed to process repository"}
            
            # 2. Optimize code embeddings
            if dataset.documents:
                # Convert dict values to list
                documents = list(dataset.documents.values())
                optimized_docs = self.optimize_code_embeddings(documents)
                
                # Update dataset with optimized documents
                for doc in optimized_docs:
                    dataset.documents[doc.id] = doc
            
            # 3. Specialize relationship embeddings
            if dataset.relations:
                documents = list(dataset.documents.values())
                specialized_relations = self.specialize_relationship_embeddings(
                    dataset.relations, documents
                )
                
                # Update dataset with specialized relations
                dataset.relations = specialized_relations
            
            # 4. Return processing results
            return {
                "status": "success",
                "dataset": {
                    "id": dataset.id,
                    "name": dataset.name,
                    "document_count": len(dataset.documents),
                    "relation_count": len(dataset.relations)
                },
                "metrics": {
                    "processing_time": dataset.metadata.get("processing_time", 0.0),
                    "chunking_applied": dataset.metadata.get("chunking_applied", False),
                    "semantic_chunking_applied": dataset.metadata.get("semantic_chunking_applied", False)
                }
            }
            
        except Exception as e:
            logger.error(f"Error processing repository: {e}")
            return {"error": str(e)}

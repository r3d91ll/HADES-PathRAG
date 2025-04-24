"""
ISNE connector for the repository ingestor.

This module provides integration between the repository ingestor
and the ISNE (Inductive Shallow Node Embedding) pipeline.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union, cast
import logging
from datetime import datetime
import uuid

from src.ingest.ingestor import RepositoryIngestor
from src.db.arango_connection import ArangoConnection
from src.types.common import EmbeddingVector

from src.isne.types.models import (
    IngestDocument,
    IngestDataset,
    DocumentRelation,
    RelationType,
    EmbeddingConfig,
    ISNEConfig
)
from src.isne.pipeline import ISNEPipeline, PipelineConfig
from src.isne.loaders.text_directory_loader import TextDirectoryLoader
from src.isne.processors.embedding_processor import EmbeddingProcessor
from src.isne.processors.graph_processor import GraphProcessor
from src.isne.integrations.arango_adapter import ArangoISNEAdapter

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ISNEIngestorConnector:
    """
    Connector between the repository ingestor and ISNE pipeline.
    
    This class provides methods to integrate the ISNE pipeline with
    the existing repository ingestor, allowing for seamless processing
    of code repositories with ISNE embeddings.
    """
    
    def __init__(
        self,
        ingestor: Optional[RepositoryIngestor] = None,
        arango_connection: Optional[ArangoConnection] = None,
        isne_pipeline: Optional[ISNEPipeline] = None,
        output_dir: str = "./isne_output",
        embedding_model: str = "sentence-transformers/all-mpnet-base-v2"
    ) -> None:
        """
        Initialize the ISNE ingestor connector.
        
        Args:
            ingestor: Repository ingestor instance (optional)
            arango_connection: ArangoDB connection (optional)
            isne_pipeline: ISNE pipeline instance (optional)
            output_dir: Directory for ISNE pipeline outputs
            embedding_model: Name of the embedding model to use
        """
        self.ingestor = ingestor
        self.arango_connection = arango_connection
        self.isne_pipeline = isne_pipeline
        self.output_dir = output_dir
        self.embedding_model = embedding_model
        
        # Initialize components if not provided
        if not self.isne_pipeline:
            self._init_isne_pipeline()
        
        # Initialize ArangoDB adapter
        from typing import Optional
        self.arango_adapter: Optional[ArangoISNEAdapter]
        if self.arango_connection:
            self.arango_adapter = ArangoISNEAdapter(self.arango_connection)
        else:
            self.arango_adapter = None
    
    def _init_isne_pipeline(self) -> None:
        """Initialize the ISNE pipeline with default configuration."""
        # Create embedding configuration
        embedding_config = EmbeddingConfig(
            model_name=self.embedding_model,
            model_dimension=768,
            batch_size=16,
            use_gpu=True,
            normalize_embeddings=True
        )
        
        # Create pipeline configuration
        pipeline_config = PipelineConfig(
            pipeline_name="repository_isne",
            output_dir=self.output_dir,
            loader_type=TextDirectoryLoader,
            embedding_config=embedding_config,
            enable_chunking=True,
            enable_embedding=True,
            enable_graph_processing=True,
            enable_isne_model=True,
            use_gpu=True,
            save_intermediate_results=True
        )
        
        # Create ISNE pipeline
        self.isne_pipeline = ISNEPipeline(pipeline_config)
        
        # Set loader
        self.isne_pipeline.set_loader(
            TextDirectoryLoader(
                document_type="code",
                create_relationships=True,
                recursive=True,
                file_extensions=[".py", ".js", ".java", ".cpp", ".go", ".ts", ".rb", ".c", ".h", ".cs", ".php", ".html", ".css", ".md", ".rst", ".txt"]
            )
        )
    
    def process_repository(
        self,
        repo_path: Union[str, Path],
        repo_name: Optional[str] = None,
        store_in_arango: bool = True
    ) -> Optional[IngestDataset]:
        """
        Process a code repository with the ISNE pipeline.
        
        Args:
            repo_path: Path to the repository
            repo_name: Name of the repository (optional)
            store_in_arango: Whether to store results in ArangoDB
            
        Returns:
            Processed IngestDataset or None if failed
        """
        repo_path = Path(repo_path)
        if not repo_path.exists() or not repo_path.is_dir():
            logger.error(f"Repository path does not exist or is not a directory: {repo_path}")
            return None
        
        # Use directory name as repo name if not provided
        if not repo_name:
            repo_name = repo_path.name
        
        logger.info(f"Processing repository {repo_name} with ISNE pipeline")
        
        try:
            # Ensure ISNE pipeline is initialized
            if self.isne_pipeline is None:
                logger.error("ISNE pipeline is not initialized.")
                return None
            # Load documents from repository
            load_result = self.isne_pipeline.load(repo_path)
            
            # Update dataset name and metadata
            if self.isne_pipeline.dataset:
                self.isne_pipeline.dataset.name = f"repo_{repo_name}"
                self.isne_pipeline.dataset.metadata.update({
                    "repository_name": repo_name,
                    "repository_path": str(repo_path),
                    "processed_at": datetime.now().isoformat()
                })
            
            # Process documents
            process_stats = self.isne_pipeline.process()
            
            # Store in ArangoDB if requested and adapter available
            if store_in_arango and self.arango_adapter:
                if self.isne_pipeline.dataset:
                    self.arango_adapter.store_dataset(self.isne_pipeline.dataset)
            
            # Return the processed dataset
            return self.isne_pipeline.dataset
            
        except Exception as e:
            logger.error(f"Error processing repository with ISNE pipeline: {e}")
            return None
    
    def get_document_embedding(
        self,
        content: str,
        document_type: str = "code"
    ) -> Optional[EmbeddingVector]:
        """
        Get embedding for a document using the ISNE pipeline's embedding processor.
        
        Args:
            content: Document content
            document_type: Type of document
            
        Returns:
            Embedding vector or None if failed
        """
        if not self.isne_pipeline or not self.isne_pipeline.embedding_processor:
            logger.error("ISNE pipeline or embedding processor not initialized")
            return None
        
        try:
            # Create temporary document
            doc_id = str(uuid.uuid4())
            document = IngestDocument(
                id=doc_id,
                content=content,
                source="inline",
                document_type=document_type
            )
            
            # Process with embedding processor
            result = self.isne_pipeline.embedding_processor.process([document])
            
            # Return embedding if available
            if result and result.documents and len(result.documents) > 0:
                return result.documents[0].embedding
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting document embedding: {e}")
            return None
    
    def find_similar_documents(
        self,
        query_content: str,
        limit: int = 10,
        min_score: float = 0.7
    ) -> List[Tuple[IngestDocument, float]]:
        """
        Find documents similar to the query content.
        
        Args:
            query_content: Query document content
            limit: Maximum number of results
            min_score: Minimum similarity score (0-1)
            
        Returns:
            List of (document, score) tuples
        """
        # Get embedding for query
        query_embedding = self.get_document_embedding(query_content)
        
        if not query_embedding:
            logger.error("Failed to get embedding for query content")
            return []
        
        # Search for similar documents in ArangoDB
        if self.arango_adapter:
            return self.arango_adapter.search_similar_documents(
                query_embedding,
                limit=limit,
                min_score=min_score
            )
        
        # If no adapter, return empty list
        return []
    
    def connect_to_ingestor(self, ingestor: RepositoryIngestor) -> None:
        """
        Connect to an existing repository ingestor.
        
        Args:
            ingestor: Repository ingestor instance
        """
        self.ingestor = ingestor
        
        # Get ArangoDB connection from ingestor if available
        if hasattr(ingestor, 'db_connection'):
            self.arango_connection = ingestor.db_connection
            
            # Initialize ArangoDB adapter
            self.arango_adapter = ArangoISNEAdapter(self.arango_connection)
            
            logger.info(f"Connected to repository ingestor with ArangoDB connection")
        else:
            logger.warning("Repository ingestor does not have an ArangoDB connection")
    
    def enhance_code_node(
        self,
        node_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Enhance a code node with ISNE embeddings.
        
        Args:
            node_id: ID of the code node
            content: Source code content
            metadata: Additional metadata
            
        Returns:
            True if successful, False otherwise
        """
        # Get embedding for the content
        embedding = self.get_document_embedding(content, "code")
        
        if not embedding:
            logger.error(f"Failed to generate embedding for code node {node_id}")
            return False
        
        # Create metadata if not provided
        if metadata is None:
            metadata = {}
        
        # Add embedding model information
        metadata["embedding_model"] = self.embedding_model
        metadata["embedding_dimension"] = len(embedding) if isinstance(embedding, list) else embedding.shape[0]
        metadata["enhanced_with_isne"] = True
        metadata["enhancement_time"] = datetime.now().isoformat()
        
        # Update node in ArangoDB if adapter available
        if self.arango_adapter:
            # Create document
            document = IngestDocument(
                id=node_id,
                content=content,
                source="code_node",
                document_type="code",
                metadata=metadata,
                embedding=embedding,
                embedding_model=self.embedding_model
            )
            
            # Store document
            self.arango_adapter.store_document(document)
            return True
        
        # If no adapter, try to update through ingestor
        elif self.ingestor and hasattr(self.ingestor, 'update_code_node_embedding'):
            # Convert numpy array to list if needed
            if hasattr(embedding, 'tolist'):
                embedding = embedding.tolist()
                
            return bool(self.ingestor.update_code_node_embedding(node_id, embedding, metadata))
        
        return False
        
    def process_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process a list of documents using the ISNE pipeline.
        
        This method takes pre-processor outputs and converts them to
        ISNE documents, generates embeddings, and returns the enhanced
        documents with embeddings.
        
        Args:
            documents: List of document dictionaries from pre-processors
            
        Returns:
            List of documents with embeddings
        """
        logger.info(f"Processing {len(documents)} documents with ISNE pipeline")
        
        if not self.isne_pipeline:
            logger.error("ISNE pipeline is not initialized")
            return documents  # Return original documents as fallback
        
        try:
            # Convert pre-processor documents to ISNE IngestDocument format
            isne_docs = []
            for doc in documents:
                doc_id = doc.get('id', str(uuid.uuid4()))
                
                # Create ISNE document
                isne_doc = IngestDocument(
                    id=doc_id,
                    content=doc.get('content', ''),
                    source=doc.get('path', 'unknown'),
                    document_type=doc.get('type', 'code'),
                    metadata={
                        key: value for key, value in doc.items()
                        if key not in ['id', 'content', 'path', 'type', 'relationships']
                    }
                )
                isne_docs.append(isne_doc)
            
            # Create dataset
            dataset_name = f"dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            dataset_id = f"id_{dataset_name}"
            dataset = IngestDataset(id=dataset_id, name=dataset_name, documents=isne_docs)
            
            # Add relationships
            for doc in documents:
                if 'relationships' in doc and doc['relationships']:
                    for rel in doc['relationships']:
                        # Get source and target IDs
                        from_id = rel['from']
                        to_id = rel['to']
                        
                        # Create relation
                        relation = DocumentRelation(
                            source_id=from_id,
                            target_id=to_id,
                            relation_type=RelationType(rel['type']),
                            weight=rel.get('weight', 0.5),
                            metadata={k: v for k, v in rel.items() if k not in ['from', 'to', 'type', 'weight']}
                        )
                        
                        # Add to dataset
                        dataset.add_relation(relation)
            
            # Process with ISNE pipeline
            processed_dataset = self.isne_pipeline.process_dataset(dataset)
            
            if processed_dataset is None:
                logger.error("ISNE pipeline returned None for processed dataset")
                return documents  # Return original documents as fallback
            
            # Convert back to dictionaries
            result_docs = []
            for isne_doc in processed_dataset.documents:
                # Find original document
                original_doc = next((d for d in documents if d.get('id') == isne_doc.id), None)
                if not original_doc:
                    continue
                
                # Create enhanced document
                enhanced_doc = dict(original_doc)  # Make a copy of the original document
                
                # Add embedding if available
                if isne_doc.embedding is not None:
                    # Convert numpy array to list if needed
                    if hasattr(isne_doc.embedding, 'tolist'):
                        enhanced_doc['embedding'] = isne_doc.embedding.tolist()
                    else:
                        enhanced_doc['embedding'] = isne_doc.embedding
                
                # Add metadata
                enhanced_doc['isne_enhanced'] = True
                enhanced_doc['embedding_model'] = isne_doc.embedding_model or self.embedding_model
                
                result_docs.append(enhanced_doc)
            
            logger.info(f"Successfully processed {len(result_docs)} documents with ISNE")
            return result_docs
            
        except Exception as e:
            logger.error(f"Error processing documents with ISNE: {e}")
            return documents  # Return original documents as fallback

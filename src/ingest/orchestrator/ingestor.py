"""
Main repository ingest orchestrator for HADES-PathRAG.

This module provides the main entry point for the ingestion pipeline,
orchestrating file discovery, preprocessing, embedding generation,
and storage in a type-safe way.
"""

import logging
import time
from typing import Dict, List, Any, Optional, Union, cast
from pathlib import Path
from datetime import datetime

from src.types.common import (
    IngestStats, PreProcessorConfig, NodeData,
    EdgeData, EmbeddingVector, StorageConfig
)
from src.db.arango_connection import ArangoConnection
from src.ingest.repository.repository_interfaces import UnifiedRepository
from src.ingest.repository.arango_repository import ArangoRepository
from src.ingest.processing.file_processor import FileProcessor
from src.ingest.processing.preprocessor_manager import PreprocessorManager
from src.isne.processors.base_processor import ProcessorConfig
from src.isne.processors.embedding_processor import EmbeddingProcessor
from src.isne.types.models import EmbeddingConfig
from src.types.common import NodeData, EdgeData, NodeID, EdgeID

# Set up logging
logger = logging.getLogger(__name__)


class RepositoryIngestor:
    """
    Main orchestrator for the HADES-PathRAG ingestion pipeline.
    
    This class coordinates the entire ingestion process, delegating to specialized
    components for file discovery, preprocessing, embedding, and storage.
    """
    
    def __init__(self, 
                 storage_config: Optional[StorageConfig] = None,
                 preprocessor_config: Optional[PreProcessorConfig] = None):
        """
        Initialize the repository ingestor.
        
        Args:
            storage_config: Configuration for storage backend
            preprocessor_config: Configuration for preprocessors
        """
        self.storage_config = storage_config or StorageConfig()
        self.preprocessor_config = preprocessor_config or PreProcessorConfig()
        
        # Initialize components
        self.repository: Optional[UnifiedRepository] = None
        self.embedding_service: Optional[EmbeddingProcessor] = None
        self.file_processor: Optional[FileProcessor] = None
        self.preprocessor_manager: Optional[PreprocessorManager] = None
        
        self._initialize_components()
    
    def _initialize_components(self) -> None:
        """Initialize all components needed for ingestion."""
        # Initialize repository
        self._initialize_repository()
        
        # Initialize file processor
        self.file_processor = FileProcessor(self.preprocessor_config)
        
        # Initialize preprocessor manager
        self.preprocessor_manager = PreprocessorManager(self.preprocessor_config)
        
        # Initialize embedding service
        self._initialize_embedding_service()
    
    def _initialize_repository(self) -> None:
        """Initialize the storage repository."""
        try:
            # Extract configuration
            host = self.storage_config.get('host', 'localhost')
            # Prepare URL with port if needed
            if 'port' in self.storage_config:
                port = self.storage_config['port']
                if isinstance(port, int):
                    host = f"{host}:{port}"
            
            username = self.storage_config.get('username', 'root')
            password = self.storage_config.get('password', '')
            database = self.storage_config.get('database', 'pathrag')
            
            # Collection configuration
            collection_prefix = self.storage_config.get('collection_prefix', '')
            node_collection = f"{collection_prefix}nodes" if collection_prefix else "nodes"
            edge_collection = f"{collection_prefix}edges" if collection_prefix else "edges"
            graph_name = f"{collection_prefix}pathrag" if collection_prefix else "pathrag"
            
            # Connect to ArangoDB
            self.connection = ArangoConnection(
                db_name=database,
                host=host,
                username=username,
                password=password
            )
            
            # Create repository
            self.repository = ArangoRepository(
                connection=self.connection,
                node_collection=node_collection,
                edge_collection=edge_collection,
                graph_name=graph_name
            )
            
            logger.info(f"Initialized repository with database {database}")
            
        except Exception as e:
            logger.error(f"Error initializing repository: {e}")
            raise
    
    def _initialize_embedding_service(self) -> None:
        """Initialize the embedding service."""
        embedding_config_raw = self.storage_config.get("embedding")
        
        # Ensure we have a mapping for ** unpacking
        if not isinstance(embedding_config_raw, dict):
            logger.warning("Embedding configuration is not a mapping; skipping embedding service init")
            self.embedding_service = None
            return
        embedding_config_dict: Dict[str, Any] = embedding_config_raw  # type: ignore[assignment]
        
        try:
            # Instantiate EmbeddingConfig object
            embedding_config_obj = EmbeddingConfig(**embedding_config_dict)
            self.embedding_service = EmbeddingProcessor(embedding_config=embedding_config_obj)
            logger.info("Initialized embedding service.")
        except Exception as e:
            logger.error(f"Failed to initialize embedding service: {e}")
            self.embedding_service = None # Ensure it's None on failure

    def ingest(self, directory: Union[str, Path], dataset_name: Optional[str] = None) -> IngestStats: 
        """
        Ingest content from a directory into the repository.
        
        Args:
            directory: Directory containing files to ingest
            dataset_name: Optional name for the dataset
            
        Returns:
            Statistics about the ingestion process
        """
        # Normalize to Path
        if isinstance(directory, str):
            directory = Path(directory)
        
        # Set dataset name if not provided
        if dataset_name is None:
            dataset_name = directory.name
        
        # Initialize stats
        stats = IngestStats(start_time=datetime.now().isoformat())
        
        # Discover files & create batches
        logger.info(f"Starting file discovery in {directory}...")
        file_batches: List[Dict[str, List[Path]]] = []
        if self.file_processor:
            batch_size_any = self.preprocessor_config.get('batch_size', 100)
            batch_size_cfg: Optional[int]
            if isinstance(batch_size_any, int):
                batch_size_cfg = batch_size_any
            else:
                batch_size_cfg = None
            file_batches = self.file_processor.process_directory(directory, batch_size=batch_size_cfg)
            total_files_discovered = sum(len(p) for batch in file_batches for p in batch.values())
            stats['files_discovered'] = total_files_discovered
            logger.info(f"Discovered {total_files_discovered} files in {len(file_batches)} batches.")
        else:
            logger.warning("File processor not initialized. Skipping file discovery.")
            stats['files_discovered'] = 0

        if not file_batches:
            logger.warning("No files discovered. Aborting ingest.")
            stats['status'] = 'aborted_no_files'
            stats['end_time'] = datetime.now().isoformat()
            return stats

        # Preprocess batches and aggregate
        logger.info("Starting preprocessing...")
        preprocessing_results_all: Dict[str, List[Dict[str, Any]]] = {}
        if self.preprocessor_manager:
            for batch in file_batches:
                batch_results = self.preprocessor_manager.preprocess_batch(batch)
                # Merge results by type
                for ftype, results in batch_results.items():
                    preprocessing_results_all.setdefault(ftype, []).extend(results)
            total_processed = sum(len(v) for v in preprocessing_results_all.values())
            stats['files_processed'] = total_processed
            logger.info(f"Preprocessing completed for {total_processed} files.")
        else:
            logger.warning("Preprocessor manager not initialized. Skipping preprocessing.")
            stats['files_processed'] = 0
        
        # Extract entities and relationships
        entities: List[Dict[str, Any]] = []
        relationships: List[Dict[str, Any]] = []
        if self.preprocessor_manager:
            extraction = self.preprocessor_manager.extract_entities_and_relationships(preprocessing_results_all)
            entities = extraction.get('entities', [])
            relationships = extraction.get('relationships', [])
        
        # Store entities and generate embeddings
        logger.info("Storing entities...")
        if self.repository: 
            self._store_entities(entities, dataset_name or Path(directory).name)
            stats['entities_stored'] = len(entities) # Assuming all are stored if no error
        else:
            logger.warning("Repository not initialized. Skipping entity storage.")
            stats['entities_stored'] = 0

        # Store relationships
        logger.info("Storing relationships...")
        if self.repository: 
            self._store_relationships(relationships)
            stats['relationships_stored'] = len(relationships) # Assuming all are stored
        else:
            logger.warning("Repository not initialized. Skipping relationship storage.")
            stats['relationships_stored'] = 0

        # Finalize and get stats
        stats['end_time'] = datetime.now().isoformat()
        stats['status'] = 'completed'
        
        try:
            if self.repository: 
                repo_stats = self.repository.collection_stats()
                stats['repository_stats'] = repo_stats
            else:
                 stats['repository_stats'] = {}
        except Exception as e:
            logger.error(f"Error retrieving repository stats: {e}")
            stats['status'] = 'completed_with_errors'

        logger.info(f"Ingestion completed. Stats: {stats}")
        return stats

    def _store_entities(self, entities: List[Dict[str, Any]], dataset_name: str) -> None:
        """Store entities and create embeddings."""
        if not self.repository: 
            logger.warning("Repository not initialized. Cannot store entities.")
            return
            
        for entity in entities:
            try:
                node_data: Dict[str, Any] = {
                    'type': entity.get('type', 'unknown'),
                    'content': entity.get('content', ''),
                    'title': entity.get('title'),
                    'source': entity.get('source') or dataset_name,
                    'metadata': entity.get('metadata', {})
                }
                
                # Add any additional properties
                for key, value in entity.items():
                    if key not in ['type', 'content', 'title', 'source', 'metadata']:
                        node_data[key] = value
                
                # Store the document
                node_id = self.repository.store_document(cast(NodeData, node_data))
                
                # Generate and store embedding if content is available
                if node_data.get('content') and self.embedding_service and self.embedding_service.embedding_fn:
                    content = node_data['content']
                    try:
                        # Call the embedding function (expects a list, returns a list)
                        embedding_list = self.embedding_service.embedding_fn([content])
                        embedding = embedding_list[0] if embedding_list else None
                        
                        if embedding is not None:
                            # Store the embedding
                            model_name = self.embedding_service.embedding_config.model_name
                            self.repository.store_embedding(
                                node_id,
                                embedding,
                                {'model': model_name} # Access model_name via config
                            )
                    except Exception as embed_error:
                        logger.error(f"Error generating or storing embedding for {node_id}: {embed_error}")
            
            except Exception as e:
                logger.error(f"Error storing entity: {e}")

    def _store_relationships(self, relationships: List[Dict[str, Any]]) -> None:
        """
        Store relationships between entities.
        
        Args:
            relationships: List of relationship data
        """
        if not self.repository: 
            logger.warning("Repository not initialized. Cannot store relationships.")
            return
            
        for rel in relationships:
            try:
                # Basic validation
                if not rel.get('source_id') or not rel.get('target_id'):
                    logger.warning(f"Skipping relationship with missing source or target ID: {rel}")
                    continue
                
                # Prepare edge data
                edge_data: Dict[str, Any] = {
                    'source_id': rel.get('source_id'),
                    'target_id': rel.get('target_id'),
                    'type': rel.get('type', 'relates_to'),
                    'weight': float(rel.get('weight', 1.0)),
                    'bidirectional': bool(rel.get('bidirectional', False)),
                    'metadata': rel.get('metadata', {})
                }
                
                # Add any additional properties
                for key, value in rel.items():
                    if key not in ['source_id', 'target_id', 'type', 'weight', 'bidirectional', 'metadata']:
                        edge_data[key] = value
                
                # Create the edge
                self.repository.create_edge(cast(EdgeData, edge_data))
                
                # Create reciprocal edge if bidirectional
                if bool(edge_data.get('bidirectional')):
                    reciprocal_edge = dict(edge_data)
                    reciprocal_edge['source_id'] = edge_data['target_id']
                    reciprocal_edge['target_id'] = edge_data['source_id']
                    
                    # Add a suffix to distinguish the reciprocal relationship type
                    if isinstance(reciprocal_edge.get('type'), str) and not reciprocal_edge['type'].endswith('_reciprocal'):
                        reciprocal_edge['type'] += '_reciprocal'
                    
                    self.repository.create_edge(cast(EdgeData, reciprocal_edge))
            
            except Exception as e:
                logger.error(f"Error creating edge: {e}")

    def get_repository(self) -> Optional[UnifiedRepository]: 
        """
        Get the repository instance.
        
        Returns:
            Repository instance if initialized, None otherwise
        """
        return self.repository

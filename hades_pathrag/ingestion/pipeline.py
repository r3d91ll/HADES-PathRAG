"""
Main ingestion pipeline for HADES-PathRAG.

This module provides the core ingestion pipeline that orchestrates the loading,
embedding, and storage of data for the PathRAG system.
"""
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, cast, ClassVar, Dict

from hades_pathrag.ingestion.models import IngestDataset, IngestDocument, DocumentRelation
from hades_pathrag.ingestion.loaders import DataLoader, TextDirectoryLoader, JSONLoader, CSVLoader
from hades_pathrag.ingestion.embeddings import ISNEEmbeddingProcessor
from hades_pathrag.embeddings.base import BaseEmbedder
from hades_pathrag.ingestion.config import load_pipeline_config
import yaml  # type: ignore
from hades_pathrag.ingestion.storage import ArangoStorage
from hades_pathrag.storage.arango import ArangoDBConnection

logger = logging.getLogger(__name__)


class IngestionPipeline:
    """
    Main ingestion pipeline for HADES-PathRAG.
    
    This class orchestrates the loading, embedding, and storage of data
    for the PathRAG system.
    """
    # Class attributes with type annotations
    db_connection: Optional[ArangoDBConnection]
    embedding_processor: Union[ISNEEmbeddingProcessor, BaseEmbedder]
    storage: Optional[ArangoStorage]
    loaders: Dict[str, DataLoader]
    
    def __init__(
        self,
        db_connection: Optional[ArangoDBConnection] = None,
        embedding_processor: Optional[Union[ISNEEmbeddingProcessor, BaseEmbedder]] = None,
        document_collection: str = "documents",
        edge_collection: str = "relationships",
        vector_collection: str = "vectors",
        graph_name: str = "knowledge_graph",
    ):
        """
        Initialize the ingestion pipeline.
        
        Args:
            db_connection: ArangoDB connection
            embedding_processor: Embedding processor to use
            document_collection: Name of the document collection
            edge_collection: Name of the edge collection
            vector_collection: Name of the vector collection
            graph_name: Name of the graph
        """
        self.db_connection = db_connection
        self.embedding_processor = embedding_processor or ISNEEmbeddingProcessor()  # type: ignore
        
        # Initialize storage as None by default
        self.storage = None
        
        # Create ArangoStorage if db_connection is provided
        if db_connection:
            self.storage = ArangoStorage(
                connection=db_connection,
                document_collection=document_collection,
                edge_collection=edge_collection,
                vector_collection=vector_collection,
                graph_name=graph_name,
            )
        
        self.loaders = {
            "text_directory": TextDirectoryLoader(),
            "json": JSONLoader(),
            "csv": CSVLoader(),
        }
    
    def register_loader(self, name: str, loader: DataLoader) -> None:
        """
        Register a custom data loader.
        
        Args:
            name: Name for the loader
            loader: The loader instance
        """
        self.loaders[name] = loader
    
    def load_data(
        self, 
        source: Union[str, Path],
        loader_type: Optional[str] = None,
        **loader_kwargs: Any
    ) -> IngestDataset:
        """
        Load data from a source using the appropriate loader.
        
        Args:
            source: Source path for the data
            loader_type: Type of loader to use (if None, auto-detect)
            **loader_kwargs: Additional arguments for the loader
            
        Returns:
            An IngestDataset containing the loaded data
        """
        source_path = Path(source) if isinstance(source, str) else source
        
        # Auto-detect loader type if not specified
        if loader_type is None:
            loader_type = self._detect_loader_type(source_path)
        
        # Get the loader
        if loader_type not in self.loaders:
            raise ValueError(f"Unknown loader type: {loader_type}")
            
        loader = self.loaders[loader_type]
        logger.info(f"Loading data from {source} using {loader_type} loader")
        
        # Load the data
        dataset = loader.load(source, **loader_kwargs)
        return dataset
    
    def _detect_loader_type(self, source_path: Path) -> str:
        """Detect appropriate loader type based on the source path."""
        if source_path.is_dir():
            return "text_directory"
        elif source_path.suffix.lower() == ".json":
            return "json"
        elif source_path.suffix.lower() in [".csv", ".tsv"]:
            return "csv"
        else:
            raise ValueError(f"Could not determine loader type for {source_path}")
    
    def compute_embeddings(self, dataset: IngestDataset) -> IngestDataset:
        """
        Compute embeddings for the documents in the dataset.
        
        Args:
            dataset: The dataset to process
            
        Returns:
            The dataset with embeddings added to documents
        """
        if not self.embedding_processor:
            logger.warning("No embedding processor available, skipping embeddings")
            return dataset
        
        logger.info(f"Computing embeddings for dataset {dataset.name}")
        # mypy union-attr fix: only ISNEEmbeddingProcessor has process()
        if isinstance(self.embedding_processor, ISNEEmbeddingProcessor):
            return self.embedding_processor.process(dataset)
        # For other embedders, implement a generic fallback or raise
        raise NotImplementedError("Embedding processor does not support batch process(). Implement per-document encoding if needed.")
    
    def store_data(self, dataset: IngestDataset) -> Dict[str, Any]:
        """
        Store the dataset in the database.
        
        Args:
            dataset: The dataset to store
            
        Returns:
            Statistics about the stored data
        """
        if not self.storage:
            logger.warning("No storage available, skipping storage")
            return {"error": "No storage available"}
        
        logger.info(f"Storing dataset {dataset.name}")
        return self.storage.store_dataset(dataset)
    
    def ingest(
        self,
        source: Union[str, Path],
        loader_type: Optional[str] = None,
        skip_embeddings: bool = False,
        skip_storage: bool = False,
        **loader_kwargs: Any
    ) -> Tuple[IngestDataset, Dict[str, Any]]:
        """
        Run the full ingestion pipeline on a data source.
        
        Args:
            source: Source path for the data
            loader_type: Type of loader to use (if None, auto-detect)
            skip_embeddings: Whether to skip computing embeddings
            skip_storage: Whether to skip storing in the database
            **loader_kwargs: Additional arguments for the loader
            
        Returns:
            Tuple of (processed dataset, statistics)
        """
        # Load data
        dataset = self.load_data(source, loader_type, **loader_kwargs)
        
        # Compute embeddings
        if not skip_embeddings:
            dataset = self.compute_embeddings(dataset)
        
        # Store data
        stats = {"storage_skipped": True}
        if not skip_storage and self.storage:
            stats = self.store_data(dataset)
        
        return dataset, stats
        

def create_pipeline_from_yaml_config(config_path: str) -> IngestionPipeline:
    """
    Create an ingestion pipeline from a YAML configuration file.
    Args:
        config_path: Path to the YAML config file.
    Returns:
        An IngestionPipeline instance
    """
    config = load_pipeline_config(config_path)

    # --- Loader selection ---
    loader_type = config.get('dataloader', {}).get('type', 'text_directory')
    loader_kwargs = config.get('dataloader', {})
    loader: DataLoader
    if loader_type == 'text_directory':
        loader = TextDirectoryLoader(
            file_extensions=loader_kwargs.get('file_extensions'),
            chunk_size=loader_kwargs.get('chunk_size'),
            chunk_overlap=loader_kwargs.get('chunk_overlap', 0)
        )
    elif loader_type == 'json':
        loader = JSONLoader()
    elif loader_type == 'csv':
        loader = CSVLoader()
    else:
        raise ValueError(f"Unsupported loader type: {loader_type}")

    # --- Embedding processor selection ---
    embedding_cfg = config.get('embedding', {})
    provider = embedding_cfg.get('provider', 'local')
    embedding_processor: Union[ISNEEmbeddingProcessor, BaseEmbedder]
    if provider == 'local':
        embedding_processor = ISNEEmbeddingProcessor()
    elif provider == 'vllm':
        # Placeholder for vLLM integration
        from hades_pathrag.embeddings.factory import create_embedder
        embedding_processor = create_embedder(
            provider='vllm',
            model_name=embedding_cfg.get('model_name'),
            endpoint=embedding_cfg.get('endpoint')
        )
    else:
        raise ValueError(f"Unsupported embedding provider: {provider}")

    # --- Storage selection ---
    storage_cfg = config.get('storage', {})
    db_url = storage_cfg.get('url', 'http://localhost:8529')
    db_host = db_url.replace('http://', '').replace('https://', '').split(':')[0]
    db_port = int(db_url.split(':')[-1]) if ':' in db_url else 8529
    db_connection = ArangoDBConnection(
        host=db_host,
        port=db_port,
        username=storage_cfg.get('username', 'root'),
        password=storage_cfg.get('password', 'password'),
        database=storage_cfg.get('database', 'pathrag')
    )

    # --- Pipeline construction ---
    pipeline = IngestionPipeline(
        db_connection=db_connection,
        embedding_processor=embedding_processor,
        document_collection=storage_cfg.get('document_collection', 'documents'),
        edge_collection=storage_cfg.get('edge_collection', 'relationships'),
        vector_collection=storage_cfg.get('vector_collection', 'vectors'),
        graph_name=storage_cfg.get('graph_name', 'knowledge_graph'),
    )
    pipeline.register_loader(loader_type, loader)
    return pipeline


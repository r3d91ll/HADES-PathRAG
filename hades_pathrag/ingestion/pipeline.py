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
    embedding_processor: ISNEEmbeddingProcessor
    storage: Optional[ArangoStorage]
    loaders: Dict[str, DataLoader]
    
    def __init__(
        self,
        db_connection: Optional[ArangoDBConnection] = None,
        embedding_processor: Optional[ISNEEmbeddingProcessor] = None,
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
        self.embedding_processor = embedding_processor or ISNEEmbeddingProcessor()
        
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
        return self.embedding_processor.process(dataset)
    
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
        

def create_pipeline_from_config(config: Any) -> IngestionPipeline:
    """
    Create an ingestion pipeline from a configuration.
    
    Args:
        config: Configuration object with database settings
        
    Returns:
        An IngestionPipeline instance
    """
    # Create database connection
    db_connection = ArangoDBConnection(
        host=config.database.url.replace('http://', '').replace('https://', '').split(':')[0],
        port=int(config.database.url.split(':')[-1]) if ':' in config.database.url else 8529,
        username=config.database.username,
        password=config.database.password,
        database=config.database.database_name
    )
    
    # Create embedding processor
    embedding_processor = ISNEEmbeddingProcessor(
        embedding_dim=128,
        weight_threshold=0.5,
    )
    
    # Create pipeline
    pipeline = IngestionPipeline(
        db_connection=db_connection,
        embedding_processor=embedding_processor,
        document_collection=config.pathrag.document_collection,
        edge_collection=config.pathrag.edge_collection,
        vector_collection=config.pathrag.vector_collection,
        graph_name=config.pathrag.graph_name,
    )
    
    return pipeline

"""
Base document loader for the ISNE pipeline.

This module defines the base loader interface and common functionality
for document loaders in the ISNE pipeline.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Any, Set, TypeVar, Generic, Protocol, Iterator, Callable
from pathlib import Path
import uuid
import logging

from src.isne.types.models import IngestDocument, IngestDataset, DocumentRelation

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class LoaderConfig:
    """
    Configuration for document loaders.
    
    This class defines common configuration parameters for document loaders
    in the ISNE pipeline.
    """
    # Content processing options
    include_metadata: bool = True
    extract_relationships: bool = True
    chunk_size: int = 1000
    chunk_overlap: int = 200
    
    # File filtering options
    file_types: Optional[List[str]] = None
    exclude_patterns: Optional[List[str]] = None
    include_patterns: Optional[List[str]] = None
    min_file_size: int = 0  # In bytes
    max_file_size: int = 10 * 1024 * 1024  # 10 MB
    
    # Relationship extraction options
    relationship_threshold: float = 0.5
    max_relationships_per_doc: int = 50
    bidirectional_relationships: bool = False
    
    # Additional options
    encoding: str = "utf-8"
    timeout: int = 30  # In seconds
    custom_options: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LoaderResult:
    """
    Result of a document loading operation.
    
    This class contains the loaded documents, relationships, and metadata
    from a document loading operation.
    """
    documents: List[IngestDocument]
    relations: List[DocumentRelation]
    dataset: Optional[IngestDataset] = None
    errors: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseLoader(ABC):
    """
    Base class for document loaders in the ISNE pipeline.
    
    This abstract base class defines the interface and common functionality
    for document loaders in the ISNE pipeline.
    """
    
    def __init__(self, config: Optional[LoaderConfig] = None) -> None:
        """
        Initialize the loader with configuration.
        
        Args:
            config: Loader configuration, uses default if not provided
        """
        self.config = config or LoaderConfig()
        
    @abstractmethod
    def load(self, source: Union[str, Path]) -> LoaderResult:
        """
        Load documents from the specified source.
        
        Args:
            source: Source to load documents from (path, URL, etc.)
            
        Returns:
            LoaderResult containing loaded documents and relationships
        """
        pass
    
    def create_dataset(
        self, 
        name: str, 
        documents: List[IngestDocument],
        relations: List[DocumentRelation],
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> IngestDataset:
        """
        Create a dataset from loaded documents and relationships.
        
        Args:
            name: Name of the dataset
            documents: List of documents to include
            relations: List of relationships between documents
            description: Optional description of the dataset
            metadata: Optional metadata for the dataset
            
        Returns:
            IngestDataset containing the documents and relationships
        """
        dataset_id = str(uuid.uuid4())
        
        dataset = IngestDataset(
            id=dataset_id,
            name=name,
            description=description,
            metadata=metadata or {}
        )
        
        # Add documents to dataset
        for doc in documents:
            dataset.add_document(doc)
        
        # Add relationships to dataset
        for rel in relations:
            dataset.add_relation(rel)
        
        return dataset
    
    def filter_files(
        self, 
        paths: List[Path], 
        config: Optional[LoaderConfig] = None
    ) -> List[Path]:
        """
        Filter files based on configuration parameters.
        
        Args:
            paths: List of file paths to filter
            config: Configuration to use for filtering (uses instance config if None)
            
        Returns:
            Filtered list of file paths
        """
        config = config or self.config
        filtered_paths: List[Path] = []
        
        for path in paths:
            # Skip directories
            if path.is_dir():
                continue
                
            # Check file extension
            if config.file_types is not None and path.suffix.lower() not in config.file_types:
                continue
                
            # Check include patterns
            if config.include_patterns is not None:
                if not any(pattern in str(path) for pattern in config.include_patterns):
                    continue
                    
            # Check exclude patterns
            if config.exclude_patterns is not None:
                if any(pattern in str(path) for pattern in config.exclude_patterns):
                    continue
                    
            # Check file size
            if path.exists():
                size = path.stat().st_size
                if size < config.min_file_size or size > config.max_file_size:
                    continue
            
            filtered_paths.append(path)
            
        return filtered_paths

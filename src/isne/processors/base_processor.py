"""
Base processor for the ISNE pipeline.

This module defines the base processor interface and common functionality
for processors in the ISNE pipeline.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Any, Set, TypeVar, Generic, Protocol, Iterator, Callable
import logging

from src.isne.types.models import IngestDocument, IngestDataset, DocumentRelation

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ProcessorConfig:
    """
    Configuration for document processors.
    
    This class defines common configuration parameters for document processors
    in the ISNE pipeline.
    """
    # Processing options
    batch_size: int = 32
    parallel_workers: int = 1
    
    # Cache options
    use_cache: bool = True
    cache_dir: Optional[str] = None
    
    # Performance options
    use_gpu: bool = True
    device: Optional[str] = None
    
    # Custom options
    custom_options: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProcessorResult:
    """
    Result of a document processing operation.
    
    This class contains the processed documents, relationships, and metadata
    from a document processing operation.
    """
    documents: List[IngestDocument]
    relations: List[DocumentRelation]
    dataset: Optional[IngestDataset] = None
    errors: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseProcessor(ABC):
    """
    Base class for document processors in the ISNE pipeline.
    
    This abstract base class defines the interface and common functionality
    for document processors in the ISNE pipeline.
    """
    
    def __init__(self, config: Optional[ProcessorConfig] = None) -> None:
        """
        Initialize the processor with configuration.
        
        Args:
            config: Processor configuration, uses default if not provided
        """
        self.config = config or ProcessorConfig()
    
    @abstractmethod
    def process(
        self, 
        documents: List[IngestDocument],
        relations: Optional[List[DocumentRelation]] = None,
        dataset: Optional[IngestDataset] = None
    ) -> ProcessorResult:
        """
        Process documents and relationships.
        
        Args:
            documents: List of documents to process
            relations: Optional list of relationships between documents
            dataset: Optional dataset containing documents and relationships
            
        Returns:
            ProcessorResult containing processed documents and relationships
        """
        pass
    
    def _get_device(self) -> str:
        """
        Get the device to use for processing.
        
        Returns:
            Device string ('cuda' or 'cpu')
        """
        if self.config.device:
            return self.config.device
        
        # Try to use GPU if requested
        if self.config.use_gpu:
            try:
                import torch
                if torch.cuda.is_available():
                    return "cuda"
            except ImportError:
                pass
        
        return "cpu"
    
    def _process_batches(
        self,
        items: List[Any],
        process_fn: Callable[[List[Any]], List[Any]],
        batch_size: Optional[int] = None
    ) -> List[Any]:
        """
        Process items in batches.
        
        Args:
            items: List of items to process
            process_fn: Function to process a batch of items
            batch_size: Batch size to use, defaults to config batch_size
            
        Returns:
            List of processed items
        """
        batch_size = batch_size or self.config.batch_size
        results = []
        
        # Process in batches
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            batch_results = process_fn(batch)
            results.extend(batch_results)
        
        return results

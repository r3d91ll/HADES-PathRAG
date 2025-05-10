"""
Base adapter interface for document processing.

This module defines the abstract base class that all format-specific adapters must implement.
The focus is on producing standardized JSON objects that can be passed between pipeline stages.

Adapters should use the centralized configuration system to determine processing behavior.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple, cast

from src.config.preprocessor_config import load_config
from src.types.common import (
    PreProcessorConfig, 
    MetadataExtractionConfig,
    EntityExtractionConfig,
    ChunkingPreparationConfig
)


class BaseAdapter(ABC):
    """
    Base interface for all document format adapters.
    
    The adapter's primary responsibility is to convert documents from various formats
    into a standardized JSON structure that can be passed between pipeline stages.
    
    The standard JSON structure includes:
    - id: Unique identifier for the document
    - source: Source path/string
    - content: Document content (extracted text)
    - format: Document format (e.g., pdf, html, text)
    - metadata: Document metadata
    - entities: Extracted entities from the content
    - raw_content: Original content when possible
    
    All adapters should use the centralized configuration system for their settings.
    """
    
    def __init__(self, format_type: Optional[str] = None):
        """
        Initialize the adapter with optional format-specific configuration.
        
        Args:
            format_type: Format this adapter handles (used to retrieve specific config)
        """
        self.format_type = format_type
        self._load_configuration()
    
    def _load_configuration(self) -> None:
        """
        Load configuration from the centralized configuration system.
        """
        try:
            # Load complete configuration
            self.config = load_config()
            
            # Get format-specific settings if format_type is provided
            self.format_config: Dict[str, Any] = {}
            if self.format_type:
                preprocessor_config = self.config.get("preprocessor_config", {})
                self.format_config = preprocessor_config.get(self.format_type, {})
            
            # Get global configuration sections
            self.metadata_config = cast(
                MetadataExtractionConfig, 
                self.config.get("metadata_extraction", {})
            )
            
            self.entity_config = cast(
                EntityExtractionConfig,
                self.config.get("entity_extraction", {})
            )
            
            self.chunking_config = cast(
                ChunkingPreparationConfig,
                self.config.get("chunking_preparation", {})
            )
            
        except Exception as e:
            # Fallback to empty config if loading fails
            self.config = {}
            self.format_config = {}
            self.metadata_config = cast(MetadataExtractionConfig, {})
            self.entity_config = cast(EntityExtractionConfig, {})
            self.chunking_config = cast(ChunkingPreparationConfig, {})
            
            import logging
            logging.getLogger(__name__).warning(f"Failed to load configuration: {e}")
    
    @abstractmethod
    def process(self, file_path: Path, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process a document file and return its standardized JSON representation.
        
        Args:
            file_path: Path to the document file
            options: Optional processing options
            
        Returns:
            Dictionary with processed content and metadata in standardized format
        """
        pass
    
    @abstractmethod
    def process_text(self, text: str, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process text content directly without a file.
        
        Args:
            text: Text content to process
            options: Optional processing options
            
        Returns:
            Dictionary with processed content and metadata in standardized format
        """
        pass
    
    @abstractmethod
    def extract_entities(self, content: Union[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract entities from document content.
        
        Args:
            content: Document content as string or dictionary
            
        Returns:
            List of extracted entities with metadata
        """
        pass
    
    @abstractmethod
    def extract_metadata(self, content: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Extract metadata from document content.
        
        Args:
            content: Document content as string or dictionary
            
        Returns:
            Dictionary of metadata
        """
        pass
    
    def prepare_for_chunking(self, content: str) -> str:
        """
        Prepare content for chunking based on configuration settings.
        
        This method applies transformations specified in the chunking_preparation
        configuration to make the content more suitable for the chunking stage.
        
        Args:
            content: Document content to prepare
            
        Returns:
            Prepared content ready for chunking
        """
        if not self.chunking_config:
            return content
            
        prepared_content = content
        
        # Add section markers if configured
        if self.chunking_config.get('add_section_markers', False):
            prepared_content = self._add_section_markers(prepared_content)
            
        # Mark potential chunk boundaries if configured
        if self.chunking_config.get('mark_chunk_boundaries', False):
            prepared_content = self._mark_chunk_boundaries(prepared_content)
            
        return prepared_content
    
    def _add_section_markers(self, content: str) -> str:
        """
        Add section markers to content to improve chunking.
        
        Args:
            content: Document content
            
        Returns:
            Content with section markers
        """
        # Default implementation - can be overridden by format-specific adapters
        return content
    
    def _mark_chunk_boundaries(self, content: str) -> str:
        """
        Mark natural chunk boundaries in the content.
        
        Args:
            content: Document content
            
        Returns:
            Content with chunk boundary markers
        """
        # Default implementation - can be overridden by format-specific adapters
        return content
    
    # Note: convert_to_markdown and convert_to_text methods were removed
    # as they are outside the core functionality of the document processing pipeline
    # and represent unnecessary complexity for adapters

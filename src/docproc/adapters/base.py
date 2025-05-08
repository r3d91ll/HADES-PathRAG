"""
Base adapter interface for document processing.

This module defines the abstract base class that all format-specific adapters must implement.
The focus is on producing standardized JSON objects that can be passed between pipeline stages.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple


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
    """
    
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
    
    # Note: convert_to_markdown and convert_to_text methods were removed
    # as they are outside the core functionality of the document processing pipeline
    # and represent unnecessary complexity for adapters

"""
Base adapter interface for document processing.

This module defines the abstract base class that all format-specific adapters must implement.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional, Union, List


class BaseAdapter(ABC):
    """Base interface for all document format adapters."""
    
    @abstractmethod
    def process(self, file_path: Path, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process a document file and return its standardized representation.
        
        Args:
            file_path: Path to the document file
            options: Optional processing options
            
        Returns:
            Dictionary with processed content and metadata
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
            Dictionary with processed content and metadata
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
    
    @abstractmethod
    def convert_to_markdown(self, content: Union[str, Dict[str, Any]]) -> str:
        """
        Convert document content to markdown format.
        
        Args:
            content: Document content as string or dictionary
            
        Returns:
            Markdown representation of the content
        """
        pass
    
    @abstractmethod
    def convert_to_text(self, content: Union[str, Dict[str, Any]]) -> str:
        """
        Convert document content to plain text.
        
        Args:
            content: Document content as string or dictionary
            
        Returns:
            Plain text representation of the content
        """
        pass

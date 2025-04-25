"""
DoclingPreProcessor: Typed pre-processor wrapper for Docling integration in HADES-PathRAG.

This class provides a unified interface for document parsing using Docling, compatible with the pre-processor pipeline.
"""
from typing import Any, Dict, Optional, Union
from pathlib import Path
from src.ingest.adapters.docling_adapter import DoclingAdapter
from .base_pre_processor import BasePreProcessor

class DoclingPreProcessor(BasePreProcessor):
    """
    Pre-processor for documents using Docling.
    """
    def __init__(self, options: Optional[Dict[str, Any]] = None) -> None:
        super().__init__()
        self.adapter = DoclingAdapter(options)

    def analyze_text(self, text: str) -> Any:
        """
        Analyze text using Docling.
        
        Args:
            text: Text content to analyze
            
        Returns:
            Docling analysis result
        """
        return self.adapter.analyze_text(text)
        
    def analyze_file(self, file_path: Union[str, Path]) -> Any:
        """
        Analyze a file using Docling.
        
        Args:
            file_path: Path to the file to analyze
            
        Returns:
            Docling analysis result
        """
        return self.adapter.analyze_file(file_path)
        
    def extract_entities(self, file_path: Union[str, Path]) -> list:
        """
        Extract entities from a file.
        
        Args:
            file_path: Path to the file to analyze
            
        Returns:
            List of entities
        """
        return self.adapter.extract_entities(file_path)
        
    def extract_relationships(self, text: str) -> list:
        """
        Extract relationships from text.
        
        Args:
            text: Text to analyze
            
        Returns:
            List of relationships
        """
        return self.adapter.extract_relationships(text)
        
    def extract_keywords(self, text: str) -> list:
        """
        Extract keywords from text.
        
        Args:
            text: Text to analyze
            
        Returns:
            List of keywords with scores
        """
        return self.adapter.extract_keywords(text)

    def process_file(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Process a document file and return its parsed structure.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Dictionary with parsed content and metadata
        """
        import os
        
        # Verify file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        # Assume the adapter has a parse method that we need to mock in testing
        # In a real implementation, this would use the Docling adapter to parse the file
        try:
            with open(file_path, 'r') as f:
                content = f.read()
                
            # Analyze the content
            analysis = self.analyze_text(content)
            
            # Extract entities and keywords
            entities = self.adapter.extract_entities(file_path)
            keywords = self.adapter.extract_keywords(content)
            
            return {
                "path": str(file_path),
                "content": content,
                "entities": entities,
                "keywords": keywords,
                "analysis": analysis
            }
        except Exception as e:
            raise Exception(f"Error processing file {file_path}: {str(e)}")

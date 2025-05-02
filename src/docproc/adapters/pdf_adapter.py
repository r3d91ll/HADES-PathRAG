"""
PDF adapter for document processing.

This module provides functionality to process PDF documents using Docling.
"""

import hashlib
import re
from pathlib import Path
from typing import Dict, Any, Optional, Union, List

# Import Docling if available
try:
    from docling.document_converter import DocumentConverter
    from docling.datamodel.base_models import InputFormat
    DOCLING_AVAILABLE = True
except ImportError:
    DOCLING_AVAILABLE = False
    # Define placeholder types for type checking
    class DocumentConverter:  # type: ignore
        """Placeholder for DocumentConverter when docling is not available."""
        def convert(self, file_path: Union[str, Path], **kwargs: Any) -> Any:
            """Placeholder convert method."""
            raise ImportError("Docling is not installed")
            
    class InputFormat:  # type: ignore
        """Placeholder for InputFormat when docling is not available."""
        pass

from .base import BaseAdapter
from .registry import register_adapter


class PDFAdapter(BaseAdapter):
    """Adapter for processing PDF documents."""
    
    def __init__(self, options: Optional[Dict[str, Any]] = None):
        """
        Initialize the PDF adapter.
        
        Args:
            options: Optional configuration options
        """
        self.options = options or {}
        if not DOCLING_AVAILABLE:
            raise ImportError("Docling is required for PDF processing. Please install docling.")
        self.converter = DocumentConverter()
    
    def process(self, file_path: Path, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process a PDF document file.
        
        Args:
            file_path: Path to the PDF file
            options: Optional processing options
            
        Returns:
            Dictionary with processed content and metadata
        """
        process_options = {**self.options, **(options or {})}
        
        # Verify the file exists
        if not file_path.exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")
        
        # Generate a stable document ID
        file_path_str = str(file_path)
        file_name = file_path.name
        file_name_sanitized = re.sub(r"[^A-Za-z0-9_:\-@\.\(\)\+\,=;\$!\*'%]+", "_", file_name)
        doc_id = f"pdf_{hashlib.md5(file_path_str.encode()).hexdigest()[:8]}_{file_name_sanitized}"
        
        try:
            # Process the PDF with Docling
            # The DocumentConverter automatically detects PDF files by extension
            converter = DocumentConverter()
            result = converter.convert(file_path)
            doc = result.document
            
            # Extract content as markdown
            markdown_content = doc.export_to_markdown()
            
            # Extract metadata from the document
            metadata = self._extract_doc_metadata(doc)
            
            # Extract entities
            entities = self.extract_entities(doc)
            
            return {
                "id": doc_id,
                "source": str(file_path),
                "content": markdown_content,
                "content_type": "markdown",
                "format": "pdf",
                "metadata": metadata,
                "entities": entities,
                "docling_document": doc  # Store the original Docling document for further processing
            }
            
        except Exception as e:
            # Handle Docling processing errors
            raise ValueError(f"Error processing PDF file {file_path}: {e}")
    
    def process_text(self, text: str, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        PDF adapter cannot process text directly.
        
        Args:
            text: Text content to process
            options: Optional processing options
            
        Raises:
            NotImplementedError: Always raised as PDF processing requires a file
        """
        raise NotImplementedError("PDF adapter cannot process text directly. Use a file instead.")
    
    def extract_entities(self, content: Union[str, Dict[str, Any], 'Any']) -> List[Dict[str, Any]]:
        """
        Extract entities from PDF content.
        
        Args:
            content: Document content as Docling document, dict or string
            
        Returns:
            List of extracted entities with metadata
        """
        entities = []
        
        # Handle different content types
        if hasattr(content, 'export_to_markdown'):
            # This is a Docling document
            # Use Docling's entity extraction if available
            if hasattr(content, 'get_entities'):
                raw_entities = content.get_entities()
                for entity in raw_entities:
                    entities.append({
                        "type": entity.entity_type,
                        "value": entity.value,
                        "confidence": getattr(entity, "confidence", 0.9),
                        "location": getattr(entity, "location", None)
                    })
            
            # If no entities were extracted, implement basic extraction
            if not entities:
                text = content.export_to_markdown()
                entities = self._extract_entities_from_text(text)
                
        elif isinstance(content, dict) and "content" in content:
            # Extract from content field in dict
            entities = self._extract_entities_from_text(content["content"])
            
        elif isinstance(content, str):
            # Extract from text
            entities = self._extract_entities_from_text(content)
            
        return entities
    
    def extract_metadata(self, content: Union[str, Dict[str, Any], 'Any']) -> Dict[str, Any]:
        """
        Extract metadata from PDF content.
        
        Args:
            content: Document content as Docling document, dict or string
            
        Returns:
            Dictionary of metadata
        """
        metadata = {}
        
        # Extract metadata based on content type
        if hasattr(content, 'get_metadata'):
            # This is a Docling document
            raw_metadata = content.get_metadata()
            # Convert Docling metadata to dictionary
            for key, value in raw_metadata.items():
                metadata[key] = value
                
        elif isinstance(content, dict):
            # If content is already a dict, extract metadata field if present
            metadata = content.get("metadata", {})
            
        return metadata
    
    def convert_to_markdown(self, content: Union[str, Dict[str, Any]]) -> str:
        """
        Convert PDF content to markdown format.
        
        Args:
            content: Document content as Docling document, dict or string
            
        Returns:
            Markdown representation of the content
        """
        if hasattr(content, 'export_to_markdown'):
            # This is a Docling document
            return content.export_to_markdown()
        elif isinstance(content, dict) and "content" in content:
            # Return content field if it's already markdown
            if content.get("content_type") == "markdown":
                return content["content"]
            # Otherwise, assume it's plain text and add basic markdown formatting
            return self._text_to_markdown(content["content"])
        elif isinstance(content, str):
            # Assume string is plain text and add basic markdown formatting
            return self._text_to_markdown(content)
        else:
            raise ValueError(f"Cannot convert content of type {type(content)} to markdown")
    
    def convert_to_text(self, content: Union[str, Dict[str, Any]]) -> str:
        """
        Convert PDF content to plain text.
        
        Args:
            content: Document content as Docling document, dict or string
            
        Returns:
            Plain text representation of the content
        """
        if hasattr(content, 'export_to_text'):
            # This is a Docling document
            return content.export_to_text()
        elif isinstance(content, dict) and "content" in content:
            # Return content field if it's already plain text
            if content.get("content_type") == "text":
                return content["content"]
            # If it's markdown, strip markdown formatting
            return self._markdown_to_text(content["content"])
        elif isinstance(content, str):
            # Assume string is markdown and strip formatting
            return self._markdown_to_text(content)
        else:
            raise ValueError(f"Cannot convert content of type {type(content)} to text")
    
    def _extract_doc_metadata(self, doc) -> Dict[str, Any]:
        """Extract metadata from a Docling document."""
        metadata = {}
        
        # Extract metadata using Docling's API if available
        if hasattr(doc, 'get_metadata'):
            raw_metadata = doc.get_metadata()
            # Convert Docling metadata to dictionary
            for key, value in raw_metadata.items():
                metadata[key] = value
        
        # Extract document statistics
        if hasattr(doc, 'get_statistics'):
            statistics = doc.get_statistics()
            metadata['statistics'] = statistics
            
        # Basic fallback metadata if nothing else is available
        if not metadata:
            metadata = {
                'page_count': getattr(doc, 'page_count', 1),
                'title': getattr(doc, 'title', ''),
                'author': getattr(doc, 'author', ''),
                'created_date': getattr(doc, 'created_date', ''),
                'modified_date': getattr(doc, 'modified_date', '')
            }
        
        return metadata
    
    def _extract_entities_from_text(self, text: str) -> List[Dict[str, Any]]:
        """Extract entities from text content."""
        # This is a basic placeholder implementation
        # In a production system, this would use NLP techniques or Docling's API
        entities = []
        
        # Extract dates using a simple regex pattern
        date_pattern = r'\b\d{1,2}/\d{1,2}/\d{2,4}\b|\b\d{4}-\d{1,2}-\d{1,2}\b'
        for match in re.finditer(date_pattern, text):
            entities.append({
                "type": "date",
                "value": match.group(0),
                "confidence": 0.8,
                "location": match.span()
            })
        
        # Extract emails
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        for match in re.finditer(email_pattern, text):
            entities.append({
                "type": "email",
                "value": match.group(0),
                "confidence": 0.9,
                "location": match.span()
            })
            
        return entities
    
    def _text_to_markdown(self, text: str) -> str:
        """Convert plain text to basic markdown."""
        # Split into lines
        lines = text.split('\n')
        result = []
        
        for line in lines:
            line = line.rstrip()
            if not line:
                # Empty line
                result.append('')
            elif line.upper() == line and len(line) < 80:
                # All caps, likely a heading
                result.append(f'## {line}')
            else:
                # Regular paragraph
                result.append(line)
                
        return '\n'.join(result)
    
    def _markdown_to_text(self, markdown: str) -> str:
        """Strip markdown formatting from text."""
        # Remove headings
        text = re.sub(r'#+\s+', '', markdown)
        # Remove bold/italic
        text = re.sub(r'\*\*?(.*?)\*\*?', r'\1', text)
        # Remove links
        text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text)
        # Remove code blocks
        text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
        # Remove inline code
        text = re.sub(r'`(.*?)`', r'\1', text)
        
        return text


# Register the adapter
register_adapter('pdf', PDFAdapter)

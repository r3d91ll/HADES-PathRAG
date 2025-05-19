"""
Mock PDF adapter for document processing (testing only).

This module provides a mocked version of the PDF adapter that doesn't depend
on Docling for testing purposes.
"""

import hashlib
import re
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple

from .base import BaseAdapter
from .registry import register_adapter


class MockPDFAdapter(BaseAdapter):
    """Mock adapter for PDF documents (for testing without Docling)."""
    
    def __init__(self, options: Optional[Dict[str, Any]] = None):
        """
        Initialize the mock PDF adapter.
        
        Args:
            options: Optional configuration options
        """
        self.options = options or {}
    
    def process(self, file_path: Path, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process a PDF document file (mock implementation).
        
        Args:
            file_path: Path to the PDF file
            options: Optional processing options
            
        Returns:
            Dictionary with processed content and metadata
        """
        options = options or {}
        merged_options = {**self.options, **options}
        
        try:
            # Just read the file as text for testing purposes
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            document_id = hashlib.md5(str(file_path).encode('utf-8')).hexdigest()
            
            return {
                'content': content,
                'metadata': {
                    'format': 'pdf',
                    'file_path': str(file_path),
                    'document_id': document_id,
                    'options': merged_options,
                    'page_count': 1
                }
            }
        except Exception as e:
            # Return error information
            return {
                'content': '',
                'metadata': {
                    'format': 'pdf',
                    'file_path': str(file_path),
                    'error': str(e),
                    'options': merged_options
                }
            }
    
    def process_text(self, text: str, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process PDF content from text.
        
        Args:
            text: Text content to process
            options: Optional processing options
            
        Returns:
            Dictionary with processed content and metadata
        """
        options = options or {}
        merged_options = {**self.options, **options}
        
        document_id = hashlib.md5(text[:100].encode('utf-8')).hexdigest()
        
        return {
            'content': text,
            'metadata': {
                'format': 'pdf',
                'document_id': document_id,
                'options': merged_options,
                'page_count': 1
            }
        }
    
    def extract_entities(self, content: Union[str, Dict[str, Any]]) -> list:
        """
        Extract entities from PDF content.
        
        Args:
            content: Document content as text or dict
            
        Returns:
            List of extracted entities with metadata
        """
        text = content if isinstance(content, str) else str(content)
        
        # Simple entity extraction based on patterns
        entities = []
        
        # Extract headings (lines ending with a colon)
        heading_pattern = r"^(.+):$"
        headings = re.findall(heading_pattern, text, re.MULTILINE)
        for heading in headings:
            entities.append({
                'type': 'heading',
                'text': heading,
                'level': 1
            })
        
        # Extract sections based on blank line separation
        section_pattern = r"\n\n(.+?)(?:\n\n|$)"
        sections = re.findall(section_pattern, text)
        for section in sections:
            if len(section.strip()) > 20:  # Only sections with substantial content
                entities.append({
                    'type': 'section',
                    'text': section.strip()
                })
        
        return entities
    
    def extract_metadata(self, content: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Extract metadata from PDF content.
        
        Args:
            content: Document content as text or dict
            
        Returns:
            Dictionary of metadata
        """
        text = content if isinstance(content, str) else str(content)
        
        # Extract basic metadata
        metadata = {
            'format': 'pdf',
            'length': len(text),
            'section_count': len(re.findall(r"\n\n", text)) + 1
        }
        
        # Try to extract title (first line)
        lines = text.strip().split('\n')
        if lines:
            metadata['title'] = lines[0].strip()
        
        return metadata
    
    def convert_to_markdown(self, content: Union[str, Dict[str, Any]]) -> str:
        """
        Convert PDF content to markdown format.
        
        Args:
            content: Document content as text or dict
            
        Returns:
            Markdown representation of the content
        """
        text = content if isinstance(content, str) else str(content)
        lines = text.strip().split('\n')
        markdown_lines = []
        
        # Simple conversion: first line as title, blank lines for paragraph separation
        if lines:
            markdown_lines.append(f"# {lines[0].strip()}")
            markdown_lines.append("")
        
        in_paragraph = False
        for line in lines[1:]:
            if not line.strip():
                if in_paragraph:
                    markdown_lines.append("")
                    in_paragraph = False
            elif line.strip().endswith(':'):
                # Heading
                if in_paragraph:
                    markdown_lines.append("")
                    in_paragraph = False
                markdown_lines.append(f"## {line.strip()}")
                markdown_lines.append("")
            else:
                # Regular paragraph
                markdown_lines.append(line)
                in_paragraph = True
        
        return '\n'.join(markdown_lines)
    
    def convert_to_text(self, content: Union[str, Dict[str, Any]]) -> str:
        """
        Convert PDF content to plain text.
        
        Args:
            content: Document content as text or dict
            
        Returns:
            Plain text representation of the content
        """
        if isinstance(content, str):
            return content
        elif isinstance(content, dict) and 'content' in content:
            return str(content['content'])
        else:
            return str(content)

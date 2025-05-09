"""
Markdown document adapter for HADES-PathRAG document processing.

This adapter specializes in processing markdown documents, extracting structured
information such as headings, code blocks, tables, and links.
"""

import re
from pathlib import Path
from typing import Dict, Any, List, Optional

# Import mistune conditionally to avoid issues when unit testing
try:
    import mistune
except ImportError:
    mistune = None


class MarkdownEntityExtractor:
    """Mistune renderer that extracts entities from markdown content."""
    
    def __init__(self):
        super().__init__()
        self.entities = []
        self.current_heading_level = 0
        self.heading_stack = []
        self.heading_counter = 0
        
    def heading(self, text, level, raw=None):
        """Extract headings as entities."""
        heading_id = f"heading-{self.heading_counter}"
        self.heading_counter += 1
        
        # Pop headings from stack if we're at a lower level
        while self.heading_stack and self.heading_stack[-1]["level"] >= level:
            self.heading_stack.pop()
        
        # Create parent relationship
        parent_id = self.heading_stack[-1]["id"] if self.heading_stack else None
        
        entity = {
            "id": heading_id,
            "type": f"h{level}",
            "name": text,
            "level": level,
            "parent_id": parent_id,
            "start_index": len(self.output) if hasattr(self, "output") else 0
        }
        
        self.heading_stack.append(entity)
        self.entities.append(entity)
        return super().heading(text, level, raw)
    
    def block_code(self, code, lang=None):
        """Extract code blocks as entities."""
        entity = {
            "type": "code_block",
            "name": lang or "code",
            "language": lang or "text",
            "content": code,
            "start_index": len(self.output) if hasattr(self, "output") else 0
        }
        self.entities.append(entity)
        return super().block_code(code, lang)
    
    def table(self, header, body):
        """Extract tables as entities."""
        entity = {
            "type": "table",
            "name": "table",
            "start_index": len(self.output) if hasattr(self, "output") else 0
        }
        self.entities.append(entity)
        return super().table(header, body)
    
    def link(self, link, title, text):
        """Extract links as entities."""
        entity = {
            "type": "link",
            "name": text,
            "url": link,
            "title": title,
            "start_index": len(self.output) if hasattr(self, "output") else 0
        }
        self.entities.append(entity)
        return super().link(link, title, text)


class MarkdownAdapter:
    """Adapter for processing markdown documents."""
    
    def __init__(self):
        self.markdown_parser = mistune.Markdown(renderer=MarkdownEntityExtractor())
        
    def process(self, file_path: Path, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process a markdown document.
        
        Args:
            file_path: Path to the markdown file
            options: Processing options
            
        Returns:
            Dictionary with processed content and metadata
        """
        options = options or {}
        
        # Generate a unique document ID
        doc_id = str(file_path.stem)
        
        # Read the file
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            print(f"Error reading markdown file: {e}")
            content = ""
            
        # Extract entities using our custom renderer
        entity_extractor = MarkdownEntityExtractor()
        markdown_parser = mistune.Markdown(renderer=entity_extractor)
        parsed_content = markdown_parser(content)
        
        # Extract metadata
        metadata = self._extract_metadata_from_markdown(content, file_path)
        
        # Build result
        return {
            "id": doc_id,
            "source": str(file_path),
            "format": "markdown",
            "content": content,
            "parsed_content": parsed_content,
            "metadata": metadata,
            "entities": entity_extractor.entities
        }
    
    def _extract_metadata_from_markdown(self, content: str, file_path: Path) -> Dict[str, Any]:
        """
        Extract metadata from markdown content.
        
        Args:
            content: Markdown content
            file_path: Path to the markdown file
            
        Returns:
            Dictionary with extracted metadata
        """
        metadata = {
            "format": "markdown",
            "file_path": str(file_path),
            "doc_type": "markdown",
            "source": str(file_path)
        }
        
        # Extract title from first heading or filename
        title_match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
        if title_match:
            metadata["title"] = title_match.group(1).strip()
        else:
            metadata["title"] = file_path.stem
        
        # Look for author information (common patterns in markdown docs)
        author_patterns = [
            r'(?:Author|Authors):\s*(.+?)(?:\n|$)',  # Author: Name
            r'By\s+(.+?)(?:\n|$)',                    # By Name
            r'\*\s*(.+?)\s*\*\s*$'                   # *Name* (italics at end of line)
        ]
        
        authors = []
        for pattern in author_patterns:
            author_match = re.search(pattern, content, re.MULTILINE | re.IGNORECASE)
            if author_match:
                author_text = author_match.group(1).strip()
                # Split authors if comma-separated
                if ',' in author_text:
                    authors.extend([a.strip() for a in author_text.split(',')])
                else:
                    authors.append(author_text)
        
        metadata["authors"] = authors
        
        # Look for date information
        date_patterns = [
            r'(?:Date|Published|Updated):\s*(.+?)(?:\n|$)',  # Date: YYYY-MM-DD
            r'(?:\d{1,2}\s+\w+\s+\d{4})',                    # 15 January 2023
            r'(?:\d{4}-\d{2}-\d{2})'                         # 2023-01-15
        ]
        
        for pattern in date_patterns:
            date_match = re.search(pattern, content, re.MULTILINE | re.IGNORECASE)
            if date_match:
                metadata["date_published"] = date_match.group(0).strip()
                break
        else:
            metadata["date_published"] = "UNK"
        
        # Default publisher
        metadata["publisher"] = "UNK"
        
        return metadata


def create_adapter() -> MarkdownAdapter:
    """Create and return a new MarkdownAdapter instance."""
    return MarkdownAdapter()

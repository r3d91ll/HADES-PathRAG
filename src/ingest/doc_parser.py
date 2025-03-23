"""
Documentation parser module for extracting structured information from documentation files.
"""
import os
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple
import logging
import markdown
from bs4 import BeautifulSoup

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentationElement:
    """Class representing a documentation element"""
    def __init__(self, title: str, content: str, source_file: str, 
                 section_type: str = "generic", line_start: int = 0, 
                 line_end: int = 0, references: Optional[List[str]] = None):
        self.title = title
        self.content = content
        self.source_file = source_file
        self.section_type = section_type  # e.g., "header", "code_example", "api_reference"
        self.line_start = line_start
        self.line_end = line_end
        self.references = references if references else []
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "title": self.title,
            "content": self.content,
            "source_file": self.source_file,
            "section_type": self.section_type,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "references": self.references
        }

class DocumentationFile:
    """Class representing a documentation file"""
    def __init__(self, file_path: str, relative_path: str, elements: Optional[List[DocumentationElement]] = None):
        self.file_path = file_path
        self.relative_path = relative_path
        self.elements = elements if elements else []
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "file_path": self.file_path,
            "relative_path": self.relative_path,
            "elements": [element.to_dict() for element in self.elements]
        }

class DocParser:
    """
    Class to parse documentation files and extract structured information.
    """
    
    # Supported documentation file extensions
    SUPPORTED_EXTENSIONS = {'.md', '.rst', '.txt', '.html'}
    
    # Patterns for finding references to code entities in documentation
    CODE_REFERENCE_PATTERNS = [
        r'`([a-zA-Z0-9_\.]+\.[a-zA-Z0-9_]+)`',  # `module.function` or `class.method`
        r'`([a-zA-Z0-9_]+)`',                   # `function` or `class`
        r'([a-zA-Z0-9_]+)\(\)',                 # function()
        r'class\s+([a-zA-Z0-9_]+)',             # class ClassName
        r'def\s+([a-zA-Z0-9_]+)',               # def function_name
        r'import\s+([a-zA-Z0-9_\.]+)'           # import module
    ]
    
    def __init__(self, repo_path: Path):
        """
        Initialize DocParser with repository path.
        
        Args:
            repo_path: Path to the repository
        """
        self.repo_path = repo_path
        
    def parse_markdown_file(self, file_path: Path) -> Optional[DocumentationFile]:
        """
        Parse a Markdown file into structured elements.
        
        Args:
            file_path: Path to the Markdown file
            
        Returns:
            DocumentationFile object or None if parsing failed
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            relative_path = file_path.relative_to(self.repo_path)
            elements = []
            
            # Convert markdown to HTML for better structure analysis
            html = markdown.markdown(content)
            soup = BeautifulSoup(html, 'html.parser')
            
            # Extract elements based on headers
            current_section = None
            current_content = []
            current_start = 0
            
            # First, try to extract the title from an h1 if available
            title = file_path.stem.replace('_', ' ').title()
            h1 = soup.find('h1')
            if h1:
                title = h1.text.strip()
            
            # Process line by line to track line numbers
            lines = content.splitlines()
            line_counter = 0
            
            # Create a preliminary element for the whole file
            file_element = DocumentationElement(
                title=title,
                content=content,
                source_file=str(relative_path),
                section_type="document",
                line_start=1,
                line_end=len(lines),
                references=self._extract_references(content)
            )
            elements.append(file_element)
            
            # Process each header to divide into sections
            for header in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
                # Find this header in the original markdown to get line number
                header_text = header.text.strip()
                header_line = -1
                
                for i, line in enumerate(lines):
                    if header_text in line and ('#' in line or '===' in line or '---' in line):
                        header_line = i + 1  # 1-indexed line numbers
                        break
                
                if header_line == -1:
                    continue  # Skip if we can't find the header
                
                # If we had a previous section, save it
                if current_section is not None:
                    section_content = '\n'.join(current_content)
                    
                    elements.append(DocumentationElement(
                        title=current_section,
                        content=section_content,
                        source_file=str(relative_path),
                        section_type=f"h{header.name[1]}",
                        line_start=current_start,
                        line_end=header_line - 1,
                        references=self._extract_references(section_content)
                    ))
                
                current_section = header_text
                current_content = []
                current_start = header_line
            
            # Add the last section
            if current_section is not None:
                section_content = '\n'.join(current_content)
                
                elements.append(DocumentationElement(
                    title=current_section,
                    content=section_content,
                    source_file=str(relative_path),
                    section_type="section",
                    line_start=current_start,
                    line_end=len(lines),
                    references=self._extract_references(section_content)
                ))
            
            # Extract code blocks as separate elements
            code_blocks = soup.find_all('code')
            for code_block in code_blocks:
                # Find code block in original markdown
                code_text = code_block.text.strip()
                code_line = -1
                
                for i, line in enumerate(lines):
                    if code_text in line:
                        code_line = i + 1
                        break
                
                if code_line != -1:
                    elements.append(DocumentationElement(
                        title="Code Example",
                        content=code_text,
                        source_file=str(relative_path),
                        section_type="code_example",
                        line_start=code_line,
                        line_end=code_line + code_text.count('\n') + 1,
                        references=self._extract_references(code_text)
                    ))
            
            return DocumentationFile(
                file_path=str(file_path),
                relative_path=str(relative_path),
                elements=elements
            )
            
        except Exception as e:
            logger.error(f"Error parsing markdown file {file_path}: {e}")
            return None
    
    def parse_documentation(self) -> Dict[str, DocumentationFile]:
        """
        Parse all documentation files in the repository.
        
        Returns:
            Dictionary mapping relative file paths to DocumentationFile objects
        """
        doc_files: Dict[str, DocumentationFile] = {}
        
        # Find all documentation files
        for ext in self.SUPPORTED_EXTENSIONS:
            files = list(self.repo_path.glob(f"**/*{ext}"))
            
            for file_path in files:
                # Skip files in virtual environments or hidden directories
                if any(part.startswith('.') or part in ['venv', 'env', 'node_modules'] 
                       for part in file_path.parts):
                    continue
                
                relative_path = file_path.relative_to(self.repo_path)
                
                # Parse based on file type
                if file_path.suffix == '.md':
                    doc_file = self.parse_markdown_file(file_path)
                    if doc_file:
                        doc_files[str(relative_path)] = doc_file
                # Add handlers for other formats as needed
                # elif file_path.suffix == '.rst':
                #     doc_file = self.parse_rst_file(file_path)
                
        return doc_files
    
    def extract_doc_code_relationships(self, doc_files: Dict[str, DocumentationFile]) -> List[Dict[str, Any]]:
        """
        Extract relationships between documentation and code elements.
        
        Args:
            doc_files: Dictionary of parsed documentation files
            
        Returns:
            List of relationships
        """
        relationships = []
        
        for path, doc_file in doc_files.items():
            for element in doc_file.elements:
                for reference in element.references:
                    relationships.append({
                        "source": f"{doc_file.relative_path}::{element.title}",
                        "target": reference,
                        "type": "documents",
                        "weight": 0.8  # High weight for documentation references
                    })
        
        return relationships
    
    def _extract_references(self, text: str) -> List[str]:
        """
        Extract references to code elements from text.
        
        Args:
            text: Text to extract references from
            
        Returns:
            List of references
        """
        references = []
        
        for pattern in self.CODE_REFERENCE_PATTERNS:
            matches = re.findall(pattern, text)
            references.extend(matches)
        
        # Remove duplicates and filter out common words that might be false positives
        common_words = {'the', 'and', 'to', 'for', 'in', 'on', 'at', 'with', 'by', 'from'}
        filtered_refs = [ref for ref in set(references) if ref.lower() not in common_words]
        
        return filtered_refs

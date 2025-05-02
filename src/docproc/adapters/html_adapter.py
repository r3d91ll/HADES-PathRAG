"""
HTML adapter for document processing.

This module provides functionality to process HTML documents, cleaning and extracting
meaningful content while preserving structure.
"""

import hashlib
import re
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from urllib.parse import urlparse

# Try to import BeautifulSoup for HTML parsing
try:
    from bs4 import BeautifulSoup, Tag, NavigableString
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False

# Try to import Docling if available
try:
    from docling.document_converter import DocumentConverter
    from docling.datamodel.base_models import InputFormat
    DOCLING_AVAILABLE = True
except ImportError:
    DOCLING_AVAILABLE = False

from .base import BaseAdapter
from .registry import register_adapter


class HTMLAdapter(BaseAdapter):
    """Adapter for processing HTML documents."""
    
    def __init__(self, options: Optional[Dict[str, Any]] = None):
        """
        Initialize the HTML adapter.
        
        Args:
            options: Optional configuration options
        """
        self.options = options or {}
        self.clean_html = self.options.get('clean_html', True)
        self.extract_code = self.options.get('extract_code', True)
        self.extract_links = self.options.get('extract_links', True)
        
        # Check if required dependencies are available
        if not BS4_AVAILABLE:
            raise ImportError("BeautifulSoup is required for HTML processing. Install with 'pip install beautifulsoup4'")
    
    def process(self, file_path: Path, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process an HTML document file.
        
        Args:
            file_path: Path to the HTML file
            options: Optional processing options
            
        Returns:
            Dictionary with processed content and metadata
        """
        process_options = {**self.options, **(options or {})}
        
        # Verify the file exists
        if not file_path.exists():
            raise FileNotFoundError(f"HTML file not found: {file_path}")
        
        # Generate a stable document ID
        file_path_str = str(file_path)
        file_name = file_path.name
        file_name_sanitized = re.sub(r"[^A-Za-z0-9_:\-@\.\(\)\+\,=;\$!\*'%]+", "_", file_name)
        doc_id = f"html_{hashlib.md5(file_path_str.encode()).hexdigest()[:8]}_{file_name_sanitized}"
        
        try:
            # Read the HTML file
            with open(file_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            # Parse and process the HTML
            return self._process_html_content(html_content, doc_id, str(file_path), process_options)
            
        except UnicodeDecodeError:
            # Try reading with different encodings if UTF-8 fails
            try:
                with open(file_path, 'r', encoding='latin-1') as f:
                    html_content = f.read()
                return self._process_html_content(html_content, doc_id, str(file_path), process_options)
            except Exception as e:
                raise ValueError(f"Error processing HTML file {file_path}: {e}")
        
        except Exception as e:
            raise ValueError(f"Error processing HTML file {file_path}: {e}")
    
    def process_text(self, text: str, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process HTML content directly.
        
        Args:
            text: HTML content to process
            options: Optional processing options
            
        Returns:
            Dictionary with processed content and metadata
        """
        process_options = {**self.options, **(options or {})}
        
        # Generate a stable document ID based on content hash
        content_hash = hashlib.md5(text.encode()).hexdigest()[:12]
        doc_id = f"html_text_{content_hash}"
        
        # Process the HTML content
        return self._process_html_content(text, doc_id, "html_content", process_options)
    
    def extract_entities(self, content: Union[str, Dict[str, Any], 'BeautifulSoup']) -> List[Dict[str, Any]]:
        """
        Extract entities from HTML content.
        
        Args:
            content: Document content as HTML string or processed dict
            
        Returns:
            List of extracted entities with metadata
        """
        entities = []
        
        # Handle different content types
        if isinstance(content, dict) and "content" in content:
            # Get the content from the dict
            html_content = content.get("raw_content", content["content"])
        elif isinstance(content, str):
            # Use the raw HTML string
            html_content = content
        else:
            return entities  # Empty list if content type is unsupported
        
        # Parse HTML if not already parsed
        if not isinstance(html_content, BeautifulSoup) and BS4_AVAILABLE:
            try:
                soup = BeautifulSoup(html_content, 'html.parser')
            except:
                return entities  # Return empty list if parsing fails
        else:
            soup = html_content
        
        # Extract entities if we have a valid soup object
        if soup:
            # Extract email addresses
            email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            for match in re.finditer(email_pattern, str(soup)):
                entities.append({
                    "type": "email",
                    "value": match.group(0),
                    "confidence": 0.9
                })
            
            # Extract links
            if isinstance(soup, BeautifulSoup):
                for link in soup.find_all('a', href=True):
                    href = link.get('href', '')
                    if href and not href.startswith('#'):
                        entities.append({
                            "type": "link",
                            "value": href,
                            "text": link.get_text(strip=True),
                            "confidence": 1.0
                        })
            
            # Extract dates
            date_pattern = r'\b\d{1,2}/\d{1,2}/\d{2,4}\b|\b\d{4}-\d{1,2}-\d{1,2}\b'
            for match in re.finditer(date_pattern, str(soup)):
                entities.append({
                    "type": "date",
                    "value": match.group(0),
                    "confidence": 0.8
                })
                
        return entities
    
    def extract_metadata(self, content: Union[str, Dict[str, Any], 'BeautifulSoup']) -> Dict[str, Any]:
        """
        Extract metadata from HTML content.
        
        Args:
            content: Document content as HTML string or processed dict
            
        Returns:
            Dictionary of metadata
        """
        metadata = {}
        
        # Handle different content types
        if isinstance(content, dict) and "metadata" in content:
            # Return existing metadata if available
            return content["metadata"]
        
        # Get HTML content
        if isinstance(content, dict) and "content" in content:
            html_content = content.get("raw_content", content["content"])
        elif isinstance(content, str):
            html_content = content
        else:
            return metadata  # Empty dict if content type is unsupported
        
        # Parse HTML if not already parsed
        if not isinstance(html_content, BeautifulSoup) and BS4_AVAILABLE:
            try:
                soup = BeautifulSoup(html_content, 'html.parser')
            except:
                return metadata  # Return empty dict if parsing fails
        else:
            soup = html_content
        
        # Extract metadata from HTML
        if soup:
            # Get title
            title_tag = soup.find('title')
            if title_tag and title_tag.string:
                metadata['title'] = title_tag.string.strip()
            
            # Get meta description
            meta_desc = soup.find('meta', attrs={'name': 'description'})
            if meta_desc:
                metadata['description'] = meta_desc.get('content', '')
            
            # Get meta keywords
            meta_keywords = soup.find('meta', attrs={'name': 'keywords'})
            if meta_keywords:
                keywords = meta_keywords.get('content', '')
                metadata['keywords'] = [k.strip() for k in keywords.split(',') if k.strip()]
            
            # Get author
            meta_author = soup.find('meta', attrs={'name': 'author'})
            if meta_author:
                metadata['author'] = meta_author.get('content', '')
            
            # Get other metadata
            for meta in soup.find_all('meta'):
                name = meta.get('name', meta.get('property', ''))
                if name and name != 'description' and name != 'keywords' and name != 'author':
                    metadata[name] = meta.get('content', '')
                    
        return metadata
    
    def convert_to_markdown(self, content: Union[str, Dict[str, Any]]) -> str:
        """
        Convert HTML content to markdown format.
        
        Args:
            content: Document content as HTML string or processed dict
            
        Returns:
            Markdown representation of the content
        """
        if isinstance(content, dict) and "content" in content:
            # If content_type is already markdown, return as is
            if content.get("content_type") == "markdown":
                return content["content"]
            # Otherwise get the raw content
            html_content = content.get("raw_content", content["content"])
        elif isinstance(content, str):
            html_content = content
        else:
            raise ValueError(f"Cannot convert content of type {type(content)} to markdown")
        
        # Use the HTML to markdown conversion
        return self._html_to_markdown(html_content)
    
    def convert_to_text(self, content: Union[str, Dict[str, Any]]) -> str:
        """
        Convert HTML content to plain text.
        
        Args:
            content: Document content as HTML string or processed dict
            
        Returns:
            Plain text representation of the content
        """
        if isinstance(content, dict) and "content" in content:
            # If content_type is already text, return as is
            if content.get("content_type") == "text":
                return content["content"]
            # If it's markdown, strip markdown formatting
            if content.get("content_type") == "markdown":
                return self._markdown_to_text(content["content"])
            # Otherwise get the raw content
            html_content = content.get("raw_content", content["content"])
        elif isinstance(content, str):
            html_content = content
        else:
            raise ValueError(f"Cannot convert content of type {type(content)} to text")
        
        # Parse HTML and extract text
        if BS4_AVAILABLE:
            try:
                soup = BeautifulSoup(html_content, 'html.parser')
                return soup.get_text(separator='\n', strip=True)
            except:
                # Fallback to basic HTML tag removal
                return self._strip_html_tags(html_content)
        else:
            # Fallback to basic HTML tag removal if BeautifulSoup is not available
            return self._strip_html_tags(html_content)
    
    def _process_html_content(self, html_content: str, doc_id: str, source: str, 
                              options: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process HTML content and return structured data.
        
        Args:
            html_content: Raw HTML content
            doc_id: Document ID
            source: Document source (file path or URL)
            options: Processing options
            
        Returns:
            Dictionary with processed content and metadata
        """
        clean_html = options.get('clean_html', self.clean_html)
        extract_code = options.get('extract_code', self.extract_code)
        
        # Parse HTML
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Extract basic metadata directly - this ensures we get title and other key metadata
        metadata = {}
        
        # Get title directly
        title_tag = soup.find('title')
        if title_tag and title_tag.string:
            metadata['title'] = title_tag.string.strip()
        
        # Get meta description
        desc_tag = soup.find('meta', attrs={'name': 'description'})
        if desc_tag and desc_tag.get('content'):
            metadata['description'] = desc_tag.get('content')
        
        # Clean HTML if requested
        if clean_html:
            self._clean_html_soup(soup)
        
        # Extract additional metadata
        metadata.update(self.extract_metadata(soup))
        
        # Extract entities
        entities = self.extract_entities(soup)
        
        # Extract code blocks if requested
        code_blocks = []
        if extract_code:
            code_blocks = self._extract_code_blocks(soup)
        
        # Convert to markdown
        markdown_content = self._html_to_markdown(str(soup))
        
        # Build result dictionary
        result = {
            "id": doc_id,
            "source": source,
            "content": markdown_content,
            "content_type": "markdown",
            "format": "html",
            "metadata": metadata,
            "entities": entities,
            "raw_content": html_content
        }
        
        if code_blocks:
            result["code_blocks"] = code_blocks
            
        return result
    
    def _clean_html_soup(self, soup: BeautifulSoup) -> None:
        """
        Clean HTML by removing boilerplate elements.
        
        Args:
            soup: BeautifulSoup object to clean (modified in place)
        """
        # Remove script and style tags
        for tag in soup(['script', 'style', 'iframe', 'noscript']):
            tag.decompose()
        
        # Remove comments
        for comment in soup.find_all(string=lambda text: isinstance(text, type(soup.Comment)) or text.strip() == ''):
            comment.extract()
        
        # Remove navigation elements
        nav_elements = ['nav', 'header', 'footer']
        for nav_tag in nav_elements:
            for nav in soup.find_all(nav_tag):
                nav.decompose()
        
        # Remove common ad classes and IDs
        ad_patterns = ['ad', 'ads', 'banner', 'sidebar', 'comment']
        for pattern in ad_patterns:
            for element in soup.find_all(class_=lambda x: x and pattern in x.lower()):
                element.decompose()
            for element in soup.find_all(id=lambda x: x and pattern in x.lower()):
                element.decompose()
    
    def _extract_code_blocks(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """
        Extract code blocks from HTML.
        
        Args:
            soup: BeautifulSoup object
            
        Returns:
            List of code blocks with metadata
        """
        code_blocks = []
        
        # Extract <pre> and <code> tags
        pre_tags = soup.find_all('pre')
        for pre in pre_tags:
            code_tag = pre.find('code')
            if code_tag:
                # Get language from class if available
                language = None
                for class_name in code_tag.get('class', []):
                    if class_name.startswith('language-'):
                        language = class_name[9:]
                        break
                
                code_blocks.append({
                    "content": code_tag.get_text(),
                    "language": language,
                    "type": "code"
                })
            else:
                # Pre without code is probably code too
                code_blocks.append({
                    "content": pre.get_text(),
                    "language": None,
                    "type": "code"
                })
        
        # Also check for code tags outside of pre
        for code in soup.find_all('code'):
            if code.parent.name != 'pre':  # Skip those already processed above
                code_blocks.append({
                    "content": code.get_text(),
                    "language": None,
                    "type": "inline_code"
                })
                
        return code_blocks
    
    def _html_to_markdown(self, html_content: str) -> str:
        """
        Convert HTML to markdown format.
        
        Args:
            html_content: HTML content
            
        Returns:
            Markdown representation
        """
        # Use Docling if available
        if DOCLING_AVAILABLE:
            try:
                converter = DocumentConverter()
                result = converter.convert_from_string(html_content, input_format=InputFormat.HTML)
                return result.document.export_to_markdown()
            except:
                # Fall back to basic conversion
                pass
        
        # Basic HTML to markdown conversion
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Process headings
        for i in range(1, 7):
            for heading in soup.find_all(f'h{i}'):
                heading.replace_with(NavigableString(f"\n{'#' * i} {heading.get_text().strip()}\n"))
        
        # Process paragraphs
        for p in soup.find_all('p'):
            p.replace_with(NavigableString(f"\n{p.get_text().strip()}\n"))
        
        # Process links
        for a in soup.find_all('a', href=True):
            text = a.get_text().strip()
            href = a['href']
            a.replace_with(NavigableString(f"[{text}]({href})"))
        
        # Process lists
        for ul in soup.find_all('ul'):
            items = []
            for li in ul.find_all('li', recursive=False):
                items.append(f"* {li.get_text().strip()}")
            ul.replace_with(NavigableString("\n" + "\n".join(items) + "\n"))
        
        for ol in soup.find_all('ol'):
            items = []
            for i, li in enumerate(ol.find_all('li', recursive=False), 1):
                items.append(f"{i}. {li.get_text().strip()}")
            ol.replace_with(NavigableString("\n" + "\n".join(items) + "\n"))
        
        # Process emphasis
        for em in soup.find_all(['em', 'i']):
            em.replace_with(NavigableString(f"*{em.get_text().strip()}*"))
        
        for strong in soup.find_all(['strong', 'b']):
            strong.replace_with(NavigableString(f"**{strong.get_text().strip()}**"))
        
        # Process code
        for code in soup.find_all('code'):
            code.replace_with(NavigableString(f"`{code.get_text().strip()}`"))
        
        # Process pre (code blocks)
        for pre in soup.find_all('pre'):
            content = pre.get_text().strip()
            pre.replace_with(NavigableString(f"\n```\n{content}\n```\n"))
        
        # Clean up the text
        markdown = soup.get_text()
        
        # Fix newlines (avoid multiple consecutive newlines)
        markdown = re.sub(r'\n{3,}', '\n\n', markdown)
        
        return markdown.strip()
    
    def _markdown_to_text(self, markdown: str) -> str:
        """
        Strip markdown formatting from text.
        
        Args:
            markdown: Markdown content
            
        Returns:
            Plain text
        """
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
        
        return text.strip()
    
    def _strip_html_tags(self, html: str) -> str:
        """
        Strip HTML tags from text.
        
        Args:
            html: HTML content
            
        Returns:
            Plain text
        """
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', ' ', html)
        # Fix whitespace
        text = re.sub(r'\s+', ' ', text)
        return text.strip()


# Register the adapter
register_adapter('html', HTMLAdapter)

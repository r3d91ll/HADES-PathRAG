"""
DoclingPreProcessor: Typed pre-processor wrapper for Docling integration in HADES-PathRAG.

This class provides a unified interface for document parsing using Docling, compatible with the pre-processor pipeline.
"""
from typing import Any, Dict, Optional, Union, List
from pathlib import Path
import hashlib
import re
from urllib.parse import urlparse

# Attempt to import BeautifulSoup for HTML parsing
try:
    from bs4 import BeautifulSoup  # type: ignore[import-not-found]
    _BS4_AVAILABLE = True
except ImportError:
    _BS4_AVAILABLE = False
    from html.parser import HTMLParser  # Fallback minimal parser

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
                
            # ------------------------------------------------------------------
            # Compute a stable document ID based on file path (matches repository)
            # ------------------------------------------------------------------
            file_path_str = str(file_path)
            file_name_raw = Path(file_path).name
            # Sanitize file name to be ArangoDB _key compatible
            file_name_sanitized = re.sub(r"[^A-Za-z0-9_:\-@\.\(\)\+\,=;\$!\*'%]+", "_", file_name_raw)
            doc_id = f"html_{hashlib.md5(file_path_str.encode()).hexdigest()[:8]}_{file_name_sanitized}"
            
            # Analyze the content (placeholder using Docling adapter)
            analysis = self.analyze_text(content)
            
            # Extract entities and keywords via adapter
            entities = self.adapter.extract_entities(file_path)
            keywords = self.adapter.extract_keywords(content)
            
            # --------------------------------------------------------------
            # Extract relationships by parsing HTML <a href="..."> links
            # --------------------------------------------------------------
            relationships: List[Dict[str, Any]] = []
            try:
                if _BS4_AVAILABLE:
                    soup = BeautifulSoup(content, "html.parser")
                    links = soup.find_all('a', href=True)
                    for a_tag in links:
                        href: str = a_tag['href']
                        # Skip empty or purely internal anchors
                        if not href or href.startswith('#'):
                            continue
                        # Identify external links (http/https). We currently skip storing them as nodes.
                        if href.startswith('http://') or href.startswith('https://'):
                            continue
                        # Resolve relative paths against current document directory
                        # Strip any URL parameters/fragments then sanitize
                        parsed_href = urlparse(href)
                        clean_path = parsed_href.path  # Remove query & fragment
                        target_path = (Path(file_path).parent / clean_path).resolve()
                        target_path_str = str(target_path)
                        target_file_name_raw = target_path.name
                        target_file_name = re.sub(r"[^A-Za-z0-9_:\-@\.\(\)\+\,=;\$!\*'%]+", "_", target_file_name_raw)
                        target_id = f"html_{hashlib.md5(target_path_str.encode()).hexdigest()[:8]}_{target_file_name}"
                        relationships.append({
                            'source_id': doc_id,
                            'target_id': target_id,
                            'type': 'references',
                            'weight': 1.0,
                            'bidirectional': False,
                            'metadata': {
                                'href': href
                            }
                        })
                else:
                    # Basic fallback parser that searches for href="..." patterns
                    for match in re.finditer(r'href=["\']([^"\']+)["\']', content):
                        href = match.group(1)
                        if not href or href.startswith('#') or href.startswith('http://') or href.startswith('https://'):
                            continue
                        parsed_href = urlparse(href)
                        clean_path = parsed_href.path
                        target_path = (Path(file_path).parent / clean_path).resolve()
                        target_path_str = str(target_path)
                        target_file_name_raw = target_path.name
                        target_file_name = re.sub(r"[^A-Za-z0-9_:\-@\.\(\)\+\,=;\$!\*'%]+", "_", target_file_name_raw)
                        target_id = f"html_{hashlib.md5(target_path_str.encode()).hexdigest()[:8]}_{target_file_name}"
                        relationships.append({
                            'source_id': doc_id,
                            'target_id': target_id,
                            'type': 'references',
                            'weight': 1.0,
                            'bidirectional': False,
                            'metadata': {
                                'href': href
                            }
                        })
            except Exception as parse_err:
                # Log but continue
                import logging
                logging.getLogger(__name__).warning(f"Failed to parse HTML links in {file_path}: {parse_err}")
                
            # ------------------------------------------------------------------
            # Build and return processed document structure
            # ------------------------------------------------------------------
            return {
                "path": str(file_path),
                "id": doc_id,
                "type": "html",
                "content": content,
                "entities": entities,
                "keywords": keywords,
                "relationships": relationships,
                "analysis": analysis
            }
        except Exception as e:
            raise Exception(f"Error processing file {file_path}: {str(e)}")

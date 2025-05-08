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
    from bs4 import BeautifulSoup, Tag, NavigableString, PageElement
    from typing import cast
    _BS4_AVAILABLE = True
except ImportError:
    _BS4_AVAILABLE = False
    from html.parser import HTMLParser  # Fallback minimal parser
    # Define placeholders for type checking
    class Tag:  # type: ignore
        """Placeholder for bs4.Tag when bs4 is not available."""
        pass
        
    class NavigableString:  # type: ignore
        """Placeholder for bs4.NavigableString when bs4 is not available."""
        pass
        
    class PageElement:  # type: ignore
        """Placeholder for bs4.PageElement when bs4 is not available."""
        pass

from src.ingest.adapters.docling_adapter import DoclingAdapter
from .base_pre_processor import BasePreProcessor, DocProcAdapter

class DoclingPreProcessor(BasePreProcessor):
    """
    Pre-processor for documents using Docling.
    
    This implementation now uses the new DocProcAdapter internally while maintaining
    the same interface for backward compatibility.
    """
    def __init__(self, options: Optional[Dict[str, Any]] = None) -> None:
        super().__init__()
        # Keep the original adapter for backward compatibility with tests
        self.adapter = DoclingAdapter(options)
        # Use new docproc adapter for actual processing
        self.doc_adapter = DocProcAdapter()

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
        Process a single file using docproc via DocProcAdapter.
        
        Args:
            file_path: Path to the file to process
            
        Returns:
            Processed document data
        """
        # Convert to string if it's a Path object
        file_path_str = str(file_path)
        
        # Use the DocProcAdapter to process the file
        try:
            # Process the file using the new docproc module
            result = self.doc_adapter.process_file(file_path_str)
            
            # If the adapter couldn't process the file, fall back to the legacy adapter
            if result is None:
                # Keep original behavior for backward compatibility
                file_path_obj = Path(file_path)
                file_path_str = str(file_path_obj.absolute())
                
                # Check if file exists
                if not file_path_obj.exists():
                    raise FileNotFoundError(f"File not found: {file_path_str}")
                
                # Extract file metadata
                file_name_raw = file_path_obj.name
                file_ext = file_path_obj.suffix.lower()
                
                # Sanitize file name to be ArangoDB _key compatible
                file_name_sanitized = re.sub(r"[^A-Za-z0-9_:\-@\.\(\)\+\,=;\$!\*'%]+", "_", file_name_raw)
                
                # Determine the document type prefix based on file extension
                doc_type_prefix = "html"  # Default
                if file_ext == ".pdf":
                    doc_type_prefix = "pdf"
                elif file_ext == ".py":
                    doc_type_prefix = "python"
                elif file_ext == ".md":
                    doc_type_prefix = "markdown"
                
                # Compute a stable document ID based on file path
                doc_id = f"{doc_type_prefix}_{hashlib.md5(file_path_str.encode()).hexdigest()[:8]}_{file_name_sanitized}"
                
                # Use the original adapter for tests
                parsed_doc = self.adapter.analyze_file(file_path)
                
                # Extract text content based on what's available in the parsed document
                if isinstance(parsed_doc, dict):
                    if 'text' in parsed_doc:
                        content = parsed_doc['text']
                    elif 'content' in parsed_doc:
                        content = parsed_doc['content']
                    else:
                        # Use the entire parsed document as a string
                        content = str(parsed_doc)
                else:
                    # If not a dict, convert to string
                    content = str(parsed_doc)
                
                # Extract entities and keywords
                entities = self.adapter.extract_entities(file_path)
                keywords = self.adapter.extract_keywords(content)
                
                # Store the analysis if available
                analysis = parsed_doc.get('analysis', {'type': 'basic_analysis'})
                
                # Build the basic result object
                result = {
                    "id": doc_id,
                    "path": file_path_str,
                    "content": content,
                    "metadata": {
                        "file_name": file_name_raw,
                        "file_type": doc_type_prefix
                    },
                    "entities": entities,
                    "keywords": keywords,
                    "analysis": analysis
                }
            
            return result
            
        except Exception as e:
            error_message = f"Error processing file {file_path_str}: {str(e)}"
            raise Exception(error_message)

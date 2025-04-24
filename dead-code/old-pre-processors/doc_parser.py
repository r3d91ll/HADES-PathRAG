"""
Orchestrates documentation file parsing using format-specific pre-processors.
"""
import os
from typing import Dict, Any, List
from .base_parser import BaseParser
from src.ingest.pre_processor.markdown_pre_processor import MarkdownPreProcessor
# from src.ingest.pre_processor.html_pre_processor import HTMLPreProcessor  # For future

class DocParser(BaseParser):
    """Orchestrates documentation file parsing using format-specific pre-processors."""

    EXTENSION_MAP = {
        ".md": MarkdownPreProcessor,
        # ".html": HTMLPreProcessor,  # Example for future
    }

    def parse(self, doc_path: str) -> Dict[str, Any]:
        results = {}
        for root, _, files in os.walk(doc_path):
            for fname in files:
                ext = os.path.splitext(fname)[1]
                preproc_cls = self.EXTENSION_MAP.get(ext)
                if preproc_cls:
                    preproc = preproc_cls()
                    fpath = os.path.join(root, fname)
                    results[fpath] = preproc.process_file(fpath)
        return results

    def parse_documentation(self, doc_path: str) -> Dict[str, Any]:
        """
        Process a documentation directory or file, creating hierarchical graphs.
        
        Args:
            doc_path: Path to document or directory
            
        Returns:
            Dictionary mapping file paths to document data
        """
        # This is a wrapper around parse for backward compatibility with the ingestor
        return self.parse(doc_path)
    
    @staticmethod
    def get_supported_extensions() -> List[str]:
        return list(DocParser.EXTENSION_MAP.keys())
        
    def extract_doc_code_relationships(self, doc_files: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract relationships between documentation and code files.
        
        Args:
            doc_files: Dictionary of document data
            
        Returns:
            List of relationships between docs and code
        """
        relationships = []
        
        for file_path, doc_data in doc_files.items():
            # Extract code references from content
            if isinstance(doc_data, dict) and "content" in doc_data:
                content = doc_data["content"]
                # Look for code references using simple pattern matching
                # This is a placeholder for more sophisticated extraction
                import re
                
                # Pattern for code file references like `file.py`, etc.
                code_refs = []
                
                # Pattern for Python file references
                py_files = re.findall(r'`([^`]*\.py)`', content)
                code_refs.extend(py_files)
                
                # Pattern for class references
                classes = re.findall(r'`([A-Z][a-zA-Z]*)`', content)
                code_refs.extend(classes)
                
                # Add relationships
                for ref in code_refs:
                    relationships.append({
                        "from_doc": file_path,
                        "to_code": ref,
                        "type": "references",
                        "weight": 0.7
                    })
                
        return relationships

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

    def process_file(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Process a document file and return its parsed structure.
        Args:
            file_path: Path to the document file
        Returns:
            Dictionary with parsed content and metadata
        """
        result = self.adapter.parse(file_path)
        return {
            "source": result["source"],
            "content": result["content"],
            "format": result.get("format"),
            "docling_document": result["docling_document"]
        }

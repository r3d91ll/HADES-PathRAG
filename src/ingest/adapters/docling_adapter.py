"""
Docling Adapter for Document Parsing in HADES-PathRAG

This module provides a typed interface to Docling for converting various document formats (PDF, HTML, DOCX, etc.) into a unified structure for downstream processing.
"""
from typing import Any, Dict, Optional, List, Union
from pathlib import Path

try:
    from docling.document_converter import DocumentConverter  # type: ignore[import-not-found]
    from docling.datamodel.base_models import InputFormat  # type: ignore[import-not-found]
except ImportError:
    DocumentConverter = None
    InputFormat = None

class DoclingAdapter:
    """
    Adapter for Docling document parsing.
    """
    def __init__(self, options: Optional[Dict[str, Any]] = None) -> None:
        self.options = options or {}
        if DocumentConverter is None:
            raise ImportError("Docling is not installed. Please install docling to use this adapter.")
        self.converter = DocumentConverter()

    def parse(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Parse a document using Docling and return a unified structure.

        Args:
            file_path: Path to the document file (PDF, HTML, etc.)
        Returns:
            Dict with keys: source (str), content (str), docling_document (Any), format (Optional[str])
        """
        file_path = Path(file_path)
        # Infer input format if possible
        input_format = self._infer_format(file_path)
        result = self.converter.convert(str(file_path), input_format=input_format)
        doc = result.document
        return {
            "source": str(file_path),
            "content": doc.export_to_markdown(),
            "docling_document": doc,
            "format": getattr(input_format, "name", None) if input_format else None
        }

    def _infer_format(self, file_path: Path) -> Optional[Any]:
        """
        Infer the input format for Docling based on file extension.
        """
        ext = file_path.suffix.lower()
        if InputFormat is None:
            return None
        if ext == ".pdf":
            return getattr(InputFormat, "PDF", None)
        if ext in {".html", ".htm"}:
            return getattr(InputFormat, "HTML", None)
        if ext == ".md":
            return getattr(InputFormat, "MARKDOWN", None)
        if ext == ".docx":
            return getattr(InputFormat, "DOCX", None)
        return None

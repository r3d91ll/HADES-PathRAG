"""
Website and PDF specific pre-processors built on Docling.

These are thin wrappers around `DoclingPreProcessor` that tag the resulting
`doc["type"]` for explicit routing (e.g., "html", "pdf").
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Union, Any

from .docling_pre_processor import DoclingPreProcessor


class WebsitePreProcessor(DoclingPreProcessor):
    """Pre-process HTML / website documents via Docling."""

    DEFAULT_TYPE = "html"

    def process_file(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        doc = super().process_file(file_path)
        doc["type"] = self.DEFAULT_TYPE
        return doc


class PDFPreProcessor(DoclingPreProcessor):
    """Pre-process PDF documents via Docling."""

    DEFAULT_TYPE = "pdf"

    def process_file(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        doc = super().process_file(file_path)
        doc["type"] = self.DEFAULT_TYPE
        return doc

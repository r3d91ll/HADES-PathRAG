"""
Website and PDF specific pre-processors.

These classes use the new docproc module adapters to process website and PDF documents.
They maintain backward compatibility with the previous interface while using the
new document processing capabilities.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Union, Any, Optional

from .base_pre_processor import DocProcAdapter


class WebsitePreProcessor(DocProcAdapter):
    """Pre-process HTML / website documents via the docproc module."""

    def __init__(self) -> None:
        # Initialize with html format
        super().__init__(format_override="html")

    def process_file(self, file_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
        # Get the processed document from the adapter
        result = super().process_file(str(file_path))
        
        # Ensure the type is set to html
        if result:
            result["type"] = "html"
        return result


class PDFPreProcessor(DocProcAdapter):
    """Pre-process PDF documents via the docproc module."""

    def __init__(self) -> None:
        # Initialize with pdf format
        super().__init__(format_override="pdf")

    def process_file(self, file_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
        # Get the processed document from the adapter
        result = super().process_file(str(file_path))
        
        # Ensure the type is set to pdf
        if result:
            result["type"] = "pdf"
        return result

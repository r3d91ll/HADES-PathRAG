"""
Abstract base parser for all modality orchestrators (code, doc, image, etc.)
"""
from typing import Dict, Any, List

class BaseParser:
    """Abstract base class for all modality parsers (code, doc, image, etc.)"""

    def parse(self, input_path: str) -> Dict[str, Any]:
        """
        Main entry point to parse an input (file, directory, etc.).
        Should be overridden by subclasses.
        """
        raise NotImplementedError

    @staticmethod
    def get_supported_extensions() -> List[str]:
        """Return a list of supported file extensions for this parser."""
        return []

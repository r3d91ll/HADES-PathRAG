"""
Configuration models for the pre-processor.

This module defines the data models for pre-processor configuration.
"""
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class PreProcessorConfig:
    """
    Configuration for the pre-processor.
    
    Attributes:
        input_dir: Directory containing files to process
        output_dir: Directory to write output files to
        python: Python file pre-processor configuration
        markdown: Markdown file pre-processor configuration
        docling: Docling pre-processor configuration
    """
    input_dir: str
    output_dir: str
    python: Dict[str, Any] = None
    markdown: Dict[str, Any] = None
    docling: Dict[str, Any] = None
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'PreProcessorConfig':
        """
        Create a PreProcessorConfig instance from a dictionary.
        
        Args:
            config_dict: Configuration dictionary
            
        Returns:
            PreProcessorConfig instance
        """
        return cls(
            python=config_dict.get("python", {"enabled": True}),
            markdown=config_dict.get("markdown", {"enabled": True}),
            docling=config_dict.get("docling", {"enabled": False}),
            output_dir=config_dict.get("output_dir", None)
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to a dictionary.
        
        Returns:
            Dictionary representation
        """
        return {
            "python": self.python,
            "markdown": self.markdown,
            "docling": self.docling,
            "output_dir": self.output_dir
        }

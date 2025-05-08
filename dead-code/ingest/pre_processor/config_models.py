"""
Configuration models for the pre-processor.

This module defines the data models for pre-processor configuration.
"""
from typing import Dict, Any, Optional
from dataclasses import dataclass, field


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
    python: Dict[str, Any] = field(default_factory=lambda: {"enabled": True})
    markdown: Dict[str, Any] = field(default_factory=lambda: {"enabled": True})
    docling: Dict[str, Any] = field(default_factory=lambda: {"enabled": True})
    
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
            input_dir=config_dict.get("input_dir", "."),
            output_dir=config_dict.get("output_dir", "./output"),
            python=config_dict.get("python", {"enabled": True}),
            markdown=config_dict.get("markdown", {"enabled": True}),
            docling=config_dict.get("docling", {"enabled": True})
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to a dictionary.
        
        Returns:
            Dictionary representation
        """
        return {
            "input_dir": self.input_dir,
            "output_dir": self.output_dir,
            "python": self.python,
            "markdown": self.markdown,
            "docling": self.docling
        }

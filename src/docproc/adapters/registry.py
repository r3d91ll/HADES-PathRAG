"""
Registry for document format adapters.

This module provides functionality to register and retrieve format-specific adapters.
"""

from typing import Dict, Type, Optional
from .base import BaseAdapter

# Registry of format adapters
_ADAPTER_REGISTRY: Dict[str, Type[BaseAdapter]] = {}


def register_adapter(format_type: str, adapter_class: Type[BaseAdapter]) -> None:
    """
    Register an adapter for a specific format.
    
    Args:
        format_type: Document format identifier (e.g., "pdf", "html")
        adapter_class: Adapter class to register
    """
    global _ADAPTER_REGISTRY
    _ADAPTER_REGISTRY[format_type.lower()] = adapter_class


def get_adapter_class(format_type: str) -> Type[BaseAdapter]:
    """
    Get the adapter class for a specific format.
    
    Args:
        format_type: Document format identifier
        
    Returns:
        Adapter class for the specified format
        
    Raises:
        ValueError: If no adapter is registered for the format
    """
    format_type = format_type.lower()
    if format_type not in _ADAPTER_REGISTRY:
        raise ValueError(f"No adapter registered for format: {format_type}")
    
    return _ADAPTER_REGISTRY[format_type]


def get_adapter_for_format(format_type: str) -> BaseAdapter:
    """
    Get an instance of the adapter for a specific format.
    
    Args:
        format_type: Document format identifier
        
    Returns:
        Adapter instance for the specified format
    """
    adapter_class = get_adapter_class(format_type)
    return adapter_class()


def get_supported_formats() -> list:
    """
    Get a list of all supported document formats.
    
    Returns:
        List of format identifiers
    """
    return list(_ADAPTER_REGISTRY.keys())

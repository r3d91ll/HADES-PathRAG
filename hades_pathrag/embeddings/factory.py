"""
Factory methods for creating embedding models.

This module provides factory methods for creating embedding models with
different implementations and configurations.
"""
from typing import Dict, List, Optional, Type, Any, Union, TypeVar, cast

import logging
import importlib
from pathlib import Path

import numpy as np
import networkx as nx

from .base import BaseEmbedder
from .interfaces import TrainableEmbedder, ComparableEmbedder, EmbeddingStats, EmbeddingCache
from ..core.config import PathRAGConfig

# Import concrete embedder implementations
from .isne import ISNEEmbedder
from .enhanced_isne import EnhancedISNEEmbedder

# Type variables
T = TypeVar('T', bound=BaseEmbedder)

logger = logging.getLogger(__name__)


class EmbedderRegistry:
    """Registry of available embedder implementations."""
    
    _registry: Dict[str, Type[BaseEmbedder]] = {
        "isne": ISNEEmbedder,
        "enhanced_isne": EnhancedISNEEmbedder,
    }
    
    @classmethod
    def register(cls, name: str, embedder_cls: Type[BaseEmbedder]) -> None:
        """
        Register a new embedder implementation.
        
        Args:
            name: Name for the embedder type
            embedder_cls: Embedder class to register
        """
        cls._registry[name.lower()] = embedder_cls
        logger.info(f"Registered embedder: {name}")
    
    @classmethod
    def get(cls, name: str) -> Optional[Type[BaseEmbedder]]:
        """
        Get embedder class by name.
        
        Args:
            name: Name of the embedder type
            
        Returns:
            Embedder class if found, None otherwise
        """
        return cls._registry.get(name.lower())
    
    @classmethod
    def list_available(cls) -> List[str]:
        """
        List available embedder implementations.
        
        Returns:
            List of registered embedder names
        """
        return list(cls._registry.keys())


def register_external_embedder(module_path: str, class_name: str, registry_name: Optional[str] = None) -> bool:
    """
    Register an embedder implementation from an external module.
    
    Args:
        module_path: Import path to the module
        class_name: Name of the class to import
        registry_name: Optional name for registration, defaults to class name
        
    Returns:
        True if registration succeeded, False otherwise
    """
    try:
        module = importlib.import_module(module_path)
        embedder_cls = getattr(module, class_name)
        
        # Ensure it's a BaseEmbedder subclass
        if not issubclass(embedder_cls, BaseEmbedder):
            logger.error(f"Class {class_name} is not a BaseEmbedder subclass")
            return False
        
        # Register with provided name or default to class name
        name = registry_name or class_name.lower()
        EmbedderRegistry.register(name, embedder_cls)
        return True
    except (ImportError, AttributeError) as e:
        logger.error(f"Error registering external embedder: {e}")
        return False


def create_embedder(
    embedder_type: str,
    config: Optional[Union[PathRAGConfig, Dict[str, Any]]] = None,
    **kwargs: Any
) -> BaseEmbedder:
    """
    Create an embedder instance of the specified type.
    
    Args:
        embedder_type: Type of embedder to create
        config: Configuration object or dictionary
        **kwargs: Additional keyword arguments for the embedder
        
    Returns:
        Configured embedder instance
        
    Raises:
        ValueError: If the embedder type is not supported
    """
    # Get configuration
    config_dict: Dict[str, Any] = {}
    if isinstance(config, PathRAGConfig):
        config_dict = {
            "embedding_dim": config.embedding_dim,
            "learning_rate": config.learning_rate,
            "epochs": config.training_epochs,
            "batch_size": config.training_batch_size,
            "negative_samples": config.negative_samples
        }
    elif isinstance(config, dict):
        config_dict = config
    
    # Override with explicit kwargs
    config_dict.update(kwargs)
    
    # Get embedder class
    embedder_cls = EmbedderRegistry.get(embedder_type)
    if not embedder_cls:
        available = ", ".join(EmbedderRegistry.list_available())
        raise ValueError(
            f"Unsupported embedder type: {embedder_type}. "
            f"Available types: {available}"
        )
    
    # Create and return instance
    try:
        return embedder_cls(**config_dict)
    except Exception as e:
        logger.error(f"Error creating {embedder_type} embedder: {e}")
        raise ValueError(f"Failed to create embedder of type {embedder_type}: {e}")


def create_embedder_from_file(model_path: str) -> BaseEmbedder:
    """
    Load an embedder model from a saved file.
    
    Args:
        model_path: Path to the saved model
        
    Returns:
        Loaded embedder model
        
    Raises:
        ValueError: If the model could not be loaded
    """
    path = Path(model_path)
    if not path.exists():
        raise ValueError(f"Model file not found: {model_path}")
    
    # Try to infer model type from filename or metadata
    model_type = None
    if "enhanced_isne" in path.name.lower():
        model_type = "enhanced_isne"
    elif "isne" in path.name.lower():
        model_type = "isne"
    
    if not model_type:
        # Try each registered embedder
        for name, cls in EmbedderRegistry._registry.items():
            try:
                # Attempt to load with this class
                return cls.load(model_path)
            except Exception:
                # Try next class
                continue
        
        raise ValueError(f"Could not determine embedder type for {model_path}")
    
    # Load with inferred type
    embedder_cls = EmbedderRegistry.get(model_type)
    if not embedder_cls:
        raise ValueError(f"Unsupported embedder type: {model_type}")
    
    try:
        return embedder_cls.load(model_path)
    except Exception as e:
        logger.error(f"Error loading {model_type} embedder from {model_path}: {e}")
        raise ValueError(f"Failed to load embedder model: {e}")


# Register embedders from external libraries if available
try:
    # Try to register Node2Vec if available
    register_external_embedder("node2vec", "Node2Vec", "node2vec")
except Exception:
    pass

try:
    # Try to register SentenceTransformer if available
    register_external_embedder("sentence_transformers", "SentenceTransformer", "sentence-bert")
except Exception:
    pass

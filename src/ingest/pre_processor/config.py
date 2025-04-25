"""
Configuration module for the pre-processor.

This module provides functions for loading and saving pre-processor
configuration from/to files.
"""
import json
import os
from typing import Dict, Any, cast
from dataclasses import asdict


def load_config(config_path: str) -> 'PreProcessorConfig':
    """
    Load configuration from a JSON file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Configuration object
        
    Raises:
        FileNotFoundError: If the config file does not exist
        json.JSONDecodeError: If the config file is not valid JSON
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
        
    try:
        with open(config_path, "r") as f:
            config_data = json.load(f)
            from .config_models import PreProcessorConfig
            return PreProcessorConfig(**config_data)
    except json.JSONDecodeError as e:
        raise e


def save_config(config: 'PreProcessorConfig', config_path: str) -> bool:
    """
    Save configuration to a JSON file.
    
    Args:
        config: Configuration object
        config_path: Path to save the configuration file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        os.makedirs(os.path.dirname(os.path.abspath(config_path)), exist_ok=True)
        
        # Convert dataclass to dict
        config_dict = asdict(config)
        
        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=4)
        return True
    except (IOError, TypeError) as e:
        print(f"Error saving config: {e}")
        return False


def get_default_config() -> 'PreProcessorConfig':
    """
    Get default configuration.
    
    Returns:
        Default configuration object
    """
    from .config_models import PreProcessorConfig
    
    return PreProcessorConfig(
        input_dir=".",
        output_dir="./output",
        python={
            "enabled": True,
            "chunk_size": 1000,
            "overlap": 200
        },
        markdown={
            "enabled": True,
            "chunk_size": 1500,
            "overlap": 300
        },
        docling={
            "enabled": True
        }
    )

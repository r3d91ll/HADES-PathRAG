"""
Configuration loading utilities for HADES-PathRAG.

This module provides functions to load various configuration files,
including the pipeline configuration and component-specific configs.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union

# Define the paths to the configuration files
CONFIG_DIR = Path(__file__).parent
PIPELINE_CONFIG_PATH = CONFIG_DIR / 'pipeline_config.yaml'


def load_yaml_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load a YAML configuration file.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        Dictionary with configuration values
        
    Raises:
        FileNotFoundError: If the configuration file does not exist
        yaml.YAMLError: If the YAML file is invalid
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
        
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
        
    return config or {}


def load_pipeline_config(config_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """
    Load the pipeline configuration.
    
    Args:
        config_path: Path to the pipeline configuration file (optional, uses default if not provided)
        
    Returns:
        Dictionary with pipeline configuration values
    """
    path = Path(config_path) if config_path else PIPELINE_CONFIG_PATH
    
    try:
        config = load_yaml_config(path)
        return config
    except Exception as e:
        # Log the error but don't crash
        import logging
        logging.getLogger(__name__).warning(f"Error loading pipeline config: {e}")
        return {}


def get_device_config() -> Dict[str, Any]:
    """
    Get the device configuration settings from the pipeline configuration.
    
    Returns:
        Dictionary with device configuration values
    """
    config = load_pipeline_config()
    
    # Check if GPU execution is enabled
    if 'gpu_execution' in config and config['gpu_execution'].get('enabled', False):
        return {
            'mode': 'gpu',
            'config': config['gpu_execution']
        }
    
    # Check if CPU execution is enabled
    if 'cpu_execution' in config and config['cpu_execution'].get('enabled', False):
        return {
            'mode': 'cpu',
            'config': config['cpu_execution']
        }
    
    # Default to GPU if both are enabled or neither is enabled
    if 'gpu_execution' in config:
        return {
            'mode': 'gpu',
            'config': config['gpu_execution']
        }
    
    # Fall back to empty config if nothing is specified
    return {
        'mode': 'auto',
        'config': {}
    }


def get_component_device(component_name: str) -> Optional[str]:
    """
    Get the device configuration for a specific component.
    
    Args:
        component_name: Name of the component (e.g., 'docproc', 'chunking', 'embedding')
        
    Returns:
        Device string (e.g., 'cuda:1') or None if not configured
    """
    device_config = get_device_config()
    
    if device_config['mode'] == 'gpu':
        component_config = device_config['config'].get(component_name, {})
        return component_config.get('device')
    elif device_config['mode'] == 'cpu':
        component_config = device_config['config'].get(component_name, {})
        return component_config.get('device')
    
    return None

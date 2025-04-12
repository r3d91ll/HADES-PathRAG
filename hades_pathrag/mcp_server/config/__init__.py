"""
Configuration handling for the MCP server.
"""
import os
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any

from hades_pathrag.mcp_server.config.settings import MCPServerConfig

logger = logging.getLogger(__name__)


def load_config(config_path: Optional[str] = None) -> MCPServerConfig:
    """
    Load the server configuration from various sources.
    
    Order of precedence:
    1. Explicit config file path
    2. Environment variable MCP_CONFIG_PATH
    3. Default config file locations
    4. Environment variables
    5. Default values
    
    Args:
        config_path: Optional path to config file
        
    Returns:
        MCPServerConfig object
    """
    # Check for explicitly provided config path
    if config_path and os.path.exists(config_path):
        return _load_from_file(config_path)
    
    # Check for config path in environment
    env_config_path = os.environ.get("MCP_CONFIG_PATH")
    if env_config_path and os.path.exists(env_config_path):
        return _load_from_file(env_config_path)
    
    # Check default locations
    default_locations = [
        os.path.join(os.getcwd(), "mcp_config.json"),
        os.path.join(os.getcwd(), "config", "mcp_config.json"),
        os.path.expanduser("~/.config/hades_pathrag/mcp_config.json"),
        "/etc/hades_pathrag/mcp_config.json",
    ]
    
    for location in default_locations:
        if os.path.exists(location):
            return _load_from_file(location)
    
    # Fall back to environment variables
    logger.info("No config file found, using environment variables and defaults")
    return MCPServerConfig.from_env()


def _load_from_file(filepath: str) -> MCPServerConfig:
    """Load configuration from a file."""
    try:
        logger.info(f"Loading config from {filepath}")
        with open(filepath, "r") as f:
            config_dict = json.load(f)
        return MCPServerConfig.from_dict(config_dict)
    except (json.JSONDecodeError, IOError) as e:
        logger.error(f"Error loading config from {filepath}: {e}")
        logger.warning("Falling back to default configuration")
        return MCPServerConfig()


# Singleton config instance
_config: Optional[MCPServerConfig] = None


def get_config() -> MCPServerConfig:
    """Get the singleton config instance."""
    global _config
    if _config is None:
        _config = load_config()
    return _config


def set_config(config: MCPServerConfig) -> None:
    """Set the singleton config instance."""
    global _config
    _config = config

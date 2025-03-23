"""
MCP Server Configuration for HADES-PathRAG.

This module contains configuration settings for the Model Context Protocol (MCP) server.
"""

import os
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

# MCP Server settings
MCP_HOST = os.getenv("MCP_HOST", "localhost")
MCP_PORT = int(os.getenv("MCP_PORT", "8765"))
MCP_FASTAPI_PORT = int(os.getenv("MCP_FASTAPI_PORT", "8123"))

# Database settings for XnX-PathRAG
ARANGO_URL = os.getenv("ARANGO_URL", f"http://{os.getenv('ARANGO_HOST', 'localhost')}:{os.getenv('ARANGO_PORT', '8529')}")
ARANGO_DB = os.getenv("ARANGO_DB", "hades_knowledge")
ARANGO_USER = os.getenv("ARANGO_USER", os.getenv("ARANGO_USERNAME", "root"))
ARANGO_PASSWORD = os.getenv("ARANGO_PASSWORD", "")

# PathRAG settings
PATHRAG_CACHE_DIR = os.getenv("PATHRAG_CACHE_DIR", "./PathRAG_cache")
PATHRAG_EMBEDDING_CACHE = os.getenv("PATHRAG_EMBEDDING_CACHE", "True").lower() in ["true", "1", "yes"]

# MCP configuration
MCP_CONFIG: Dict[str, Any] = {
    "server": {
        "host": MCP_HOST,
        "port": MCP_PORT,
        "fastapi_port": MCP_FASTAPI_PORT,
    },
    "arango": {
        "url": ARANGO_URL,
        "db": ARANGO_DB,
        "user": ARANGO_USER,
        "password": ARANGO_PASSWORD,
    },
    "pathrag": {
        "cache_dir": PATHRAG_CACHE_DIR,
        "embedding_cache": PATHRAG_EMBEDDING_CACHE,
        "kv_storage": os.getenv("PATHRAG_KV_STORAGE", "JsonKVStorage"),
        "vector_storage": os.getenv("PATHRAG_VECTOR_STORAGE", "NanoVectorDBStorage"),
        "graph_storage": os.getenv("PATHRAG_GRAPH_STORAGE", "NetworkXStorage"),
    }
}

def get_mcp_config() -> Dict[str, Any]:
    """Get the current MCP server configuration."""
    return MCP_CONFIG

def get_arango_config() -> Dict[str, Any]:
    """Get ArangoDB-specific configuration."""
    return MCP_CONFIG["arango"]

def get_pathrag_config() -> Dict[str, Any]:
    """Get PathRAG-specific configuration."""
    return MCP_CONFIG["pathrag"]

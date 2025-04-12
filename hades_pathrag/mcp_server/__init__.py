"""
Model Context Protocol (MCP) Server for HADES-PathRAG.

This module implements the Model Context Protocol (MCP) specification defined at:
https://modelcontextprotocol.io/

It provides a standardized interface for LLMs and other agents
to interact with the HADES-PathRAG system through tools that enable
path-based retrieval, document embedding, and semantic search.
"""

from hades_pathrag.mcp_server.mcp_standalone import app as mcp_app

__all__ = ["mcp_app"]

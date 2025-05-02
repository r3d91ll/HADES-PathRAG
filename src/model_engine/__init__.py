"""
HADES-PathRAG Model Engine.

This module provides a unified interface for working with language models,
including embedding generation, completion, and chat functionality.
"""

from src.model_engine.adapters.vllm_adapter import VLLMAdapter, start_vllm_server
from src.model_engine.server_manager import ServerManager, get_server_manager

__all__ = [
    "VLLMAdapter", 
    "start_vllm_server",
    "ServerManager",
    "get_server_manager"
]

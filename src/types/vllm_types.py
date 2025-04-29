"""
Type definitions for vLLM integration in HADES-PathRAG.

This module provides type annotations specific to the vLLM integration
to ensure consistency and improve type safety.
"""

from typing import Dict, List, Any, Optional, Union, TypedDict, Literal
from pydantic import BaseModel, Field


class VLLMServerConfigType(TypedDict, total=False):
    """Configuration for the vLLM server."""
    host: str
    port: int
    tensor_parallel_size: int
    gpu_memory_utilization: float
    max_model_len: Optional[int]
    dtype: str


class VLLMModelConfigType(TypedDict, total=False):
    """Configuration for a vLLM model."""
    model_id: str
    embedding_model_id: Optional[str]
    max_tokens: int
    temperature: float
    top_p: float
    top_k: int
    context_window: int
    truncate_input: bool


class VLLMConfigType(TypedDict, total=False):
    """Configuration for the vLLM integration."""
    server: VLLMServerConfigType
    ingestion_models: Dict[str, VLLMModelConfigType]
    inference_models: Dict[str, VLLMModelConfigType]


# Model mode type
ModelMode = Literal["inference", "ingestion"]

# Function type for checking server status
ServerStatusType = Dict[str, Any]

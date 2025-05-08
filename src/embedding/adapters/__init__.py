"""Embedding adapters for different model backends.

This module contains implementations of the EmbeddingAdapter interface
for different model engines including vLLM and Ollama.
"""

from __future__ import annotations

# Import all adapters so they register themselves
from .vllm_adapter import VLLMEmbeddingAdapter
from .ollama_adapter import OllamaEmbeddingAdapter

# Re-export adapter classes
__all__: list[str] = [
    "VLLMEmbeddingAdapter",
    "OllamaEmbeddingAdapter",
]

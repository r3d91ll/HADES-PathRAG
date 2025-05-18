"""Embedding adapters for different model backends.

This module contains implementations of the EmbeddingAdapter interface
for different model engines including vLLM, Ollama, and ModernBERT.
"""

from __future__ import annotations

# Import all adapters so they register themselves
from .cpu_adapter import CPUEmbeddingAdapter
from .vllm_adapter import VLLMEmbeddingAdapter
from .ollama_adapter import OllamaEmbeddingAdapter
from .modernbert_adapter import ModernBERTEmbeddingAdapter

# Re-export adapter classes
__all__: list[str] = [
    "CPUEmbeddingAdapter",
    "VLLMEmbeddingAdapter",
    "OllamaEmbeddingAdapter",
    "ModernBERTEmbeddingAdapter",
]

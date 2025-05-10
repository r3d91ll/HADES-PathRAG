from __future__ import annotations

"""Semantic text chunker using [Chonky](https://github.com/mithril-security/chonky).

This implementation integrates Chonky with the Haystack model engine for efficient model loading
and management. The model is lazily initialized and managed by the Haystack engine.

The helper ``chunk_text`` mirrors the signature of ``chunk_python_code`` so the
`PreprocessorManager` can treat code and text files uniformly.

Features:
- Uses the high-quality mirth/chonky_modernbert_large_1 model
- Token-aware chunking with configurable parameters
- Efficient model loading through Haystack model engine
- Parallel processing of multiple documents
"""

from typing import Any, Dict, List, Optional, Union, Tuple, Generator, Callable
import hashlib
import uuid
import logging
import json
import os
import torch
from functools import lru_cache
from multiprocessing.pool import ThreadPool
from contextlib import contextmanager

# Optional imports to handle gracefully
try:
    from transformers import AutoTokenizer
except ImportError as exc:  # pragma: no cover
    # Not raising here, as the engine may be loading transformers itself
    AutoTokenizer = None  # type: ignore

try:
    from chonky import ParagraphSplitter  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "Package 'chonky' is required for semantic text chunking.\n"
        "Install with:  pip install chonky"
    ) from exc

# Import Haystack model engine
from src.model_engine.engines.haystack import HaystackModelEngine
from src.config.chunker_config import get_chunker_config

logger = logging.getLogger(__name__)

# Initialize global model engine and tokenizer
_MODEL_ENGINE: Optional[HaystackModelEngine] = None
_TOKENIZER: Any = None
_SPLITTER_CACHE: Dict[str, ParagraphSplitter] = {}


def get_model_engine() -> HaystackModelEngine:
    """Get the global model engine instance, initializing if necessary."""
    global _MODEL_ENGINE  # pylint: disable=global-statement
    if _MODEL_ENGINE is None:
        _MODEL_ENGINE = HaystackModelEngine()
        # Start the service if not already running
        if not _MODEL_ENGINE.status().get("running", False):
            logger.info("Starting Haystack model engine service")
            _MODEL_ENGINE.start()
    return _MODEL_ENGINE


@lru_cache(maxsize=2)
def get_tokenizer(model_id: str) -> Any:
    """Get a tokenizer for the specified model ID with caching."""
    if AutoTokenizer is None:
        raise ImportError(
            "Package 'transformers' is required for token counting.\n"
            "Install with:  pip install transformers"
        )
    return AutoTokenizer.from_pretrained(model_id)


def _count_tokens(text: str, tokenizer: Any) -> int:
    """Count the number of tokens in a text string."""
    if not text.strip():
        return 0
    return len(tokenizer.encode(text))


def _split_text_by_tokens(text: str, max_tokens: int, min_tokens: int, 
                         tokenizer: Any, overlap: int = 100) -> List[str]:
    """Split text into chunks respecting token limits.
    
    Args:
        text: The text to split
        max_tokens: Maximum tokens per chunk
        min_tokens: Minimum tokens per chunk 
        tokenizer: Tokenizer to use for counting tokens
        overlap: Number of tokens to overlap between chunks
        
    Returns:
        List of text chunks
    """
    if not text.strip():
        return []
        
    # Short text that fits in one chunk
    token_count = _count_tokens(text, tokenizer)
    if token_count <= max_tokens:
        return [text]
    
    # Split on paragraph boundaries first
    paragraphs = text.split("\n\n")
    chunks: List[str] = []
    current_chunk: List[str] = []
    current_token_count = 0
    
    for para in paragraphs:
        para_token_count = _count_tokens(para, tokenizer)
        
        # If paragraph itself exceeds max tokens, split it on sentence boundaries
        if para_token_count > max_tokens:
            sentences = para.replace(".", ".\n").replace("!", "!\n").replace("?", "?\n").split("\n")
            for sentence in sentences:
                if not sentence.strip():
                    continue
                
                sentence_token_count = _count_tokens(sentence, tokenizer)
                
                # Handle very long sentences by splitting on tokens directly
                if sentence_token_count > max_tokens:
                    words = sentence.split()
                    sub_chunk: List[str] = []
                    sub_token_count = 0
                    
                    for word in words:
                        word_token_count = _count_tokens(word, tokenizer)
                        if sub_token_count + word_token_count > max_tokens:
                            if sub_chunk and sub_token_count >= min_tokens:
                                chunks.append(" ".join(sub_chunk))
                            sub_chunk = [word]
                            sub_token_count = word_token_count
                        else:
                            sub_chunk.append(word)
                            sub_token_count += word_token_count
                    
                    if sub_chunk and sub_token_count >= min_tokens:
                        chunks.append(" ".join(sub_chunk))
                else:
                    # Normal sentence handling
                    if current_token_count + sentence_token_count > max_tokens:
                        if current_chunk and current_token_count >= min_tokens:
                            chunks.append("\n".join(current_chunk))
                        current_chunk = [sentence]
                        current_token_count = sentence_token_count
                    else:
                        current_chunk.append(sentence)
                        current_token_count += sentence_token_count
        else:
            # Normal paragraph handling
            if current_token_count + para_token_count > max_tokens:
                if current_chunk and current_token_count >= min_tokens:
                    chunks.append("\n\n".join(current_chunk))
                current_chunk = [para]
                current_token_count = para_token_count
            else:
                current_chunk.append(para)
                current_token_count += para_token_count
    
    # Add the final chunk
    if current_chunk and current_token_count >= min_tokens:
        chunks.append("\n\n".join(current_chunk))
    
    return chunks


def _get_splitter_with_engine(model_id: str = "mirth/chonky_modernbert_large_1", 
                             device: str = "cuda:0") -> ParagraphSplitter:
    """Get a Chonky paragraph splitter using the Haystack model engine.
    
    This implementation uses the configured model engine to load the model,
    leveraging our model caching and resource management.
    """
    # Use cache to avoid re-creating splitters
    cache_key = f"{model_id}_{device}"
    if cache_key in _SPLITTER_CACHE:
        return _SPLITTER_CACHE[cache_key]
    
    # Get model engine and ensure it's running
    engine = get_model_engine()
    
    # Load the model using the engine
    load_result = engine.load_model(model_id)
    
    # Extract success status and error message safely without assuming dict type
    success = False
    error_msg = "Unknown error"
    
    try:
        # Try to access as dictionary
        success = bool(load_result.get("success", False)) if hasattr(load_result, "get") else False
        if not success and hasattr(load_result, "get"):
            error_msg = str(load_result.get("error", "Unknown error"))
    except (AttributeError, TypeError):
        # Handle non-dictionary response
        error_msg = f"Unexpected response type: {type(load_result).__name__}"
    
    if not success:
        logger.error("Failed to load Chonky model: %s", error_msg)
        raise RuntimeError(f"Failed to load Chonky model: {error_msg}")
    
    logger.info("Loading Chonky paragraph splitter '%s'", model_id)
    splitter = ParagraphSplitter(model_id=model_id, device=device)
    _SPLITTER_CACHE[cache_key] = splitter
    return splitter


def _hash_path(path: str) -> str:
    """Create a short hash from a file path for chunk IDs."""
    return hashlib.md5(path.encode()).hexdigest()[:8]


@contextmanager
def ensure_model_engine() -> Generator[HaystackModelEngine, None, None]:
    """Context manager to ensure model engine is started and properly cleaned up."""
    engine = get_model_engine()
    try:
        yield engine
    finally:
        # We don't stop the engine here to allow reuse across calls
        pass


def chunk_text(
    document: Dict[str, Any], *, max_tokens: int = 2048, output_format: str = "python"
) -> Union[List[Dict[str, Any]], str]:  # noqa: D401
    """Split plain-text/Markdown document into semantic paragraphs using Chonky.

    Parameters
    ----------
    document:
        Pre-processed document dict. Must contain ``content`` and ``path`` keys.
    max_tokens:
        Token budget for each chunk (default comes from config).
    output_format:
        Output format of the chunks. Can be either "python" or "json".

    Returns
    -------
    Union[List[Dict[str, Any]], str]
        List of chunk dictionaries or JSON string, depending on output_format
    """
    # Load configuration
    cfg = get_chunker_config("chonky")
    
    # Extract configuration values with defaults
    if max_tokens == 2048:
        max_tokens = cfg.get("max_tokens", 2048)
    
    min_tokens = cfg.get("min_tokens", 64)
    model_id = cfg.get("model_id", "mirth/chonky_modernbert_large_1")
    device = cfg.get("device", "cuda:0" if torch.cuda.is_available() else "cpu")
    overlap_tokens = cfg.get("overlap_tokens", 200)
    
    # Extract document content and path
    source = document.get("content") or document.get("source", "")
    path: str = document.get("path", "unknown")
    
    # Get tokenizer for token counting regardless of content
    tokenizer = get_tokenizer(model_id)
    
    # Process with model engine to ensure proper resource management even for empty docs
    with ensure_model_engine():
        if not source.strip():
            logger.warning("Chunk_text called with empty document: %s", path)
            return []

        # Get Chonky paragraph splitter using the model engine
        try:
            splitter = _get_splitter_with_engine(model_id=model_id, device=device)
            
            # Run semantic paragraph splitting
            logger.debug("Running Chonky paragraph splitter on %s", path)
            paragraphs: List[str] = list(splitter(source))
            
            # Apply token-aware splitting to handle long paragraphs
            processed_paragraphs: List[str] = []
            for para in paragraphs:
                if not para.strip():
                    continue
                    
                # Check if paragraph exceeds token limit and split if needed
                para_token_count = _count_tokens(para, tokenizer)
                if para_token_count > max_tokens:
                    logger.debug("Splitting large paragraph (%d tokens) into smaller chunks", para_token_count)
                    sub_chunks = _split_text_by_tokens(
                        para, max_tokens, min_tokens, tokenizer, overlap_tokens
                    )
                    processed_paragraphs.extend(sub_chunks)
                else:
                    processed_paragraphs.append(para)
            
            # Create chunk objects with metadata
            chunks: List[Dict[str, Any]] = []
            parent_id = f"doc:{_hash_path(path)}"
            
            for idx, para in enumerate(processed_paragraphs):
                if not para.strip():
                    continue
                    
                # Generate unique chunk ID
                chunk_id = f"chunk:{uuid.uuid4().hex[:8]}"
                
                # Count tokens for metadata
                token_count = _count_tokens(para, tokenizer)
                
                chunks.append({
                    "id": chunk_id,
                    "parent": parent_id,
                    "path": path,
                    "type": document.get("type", "markdown"),
                    "content": para,
                    "symbol_type": "paragraph",
                    "name": f"paragraph_{idx}",
                    "line_start": 0,
                    "line_end": 0,
                    "token_count": token_count,
                    "embedding": None,  # Placeholder for future embedding
                })
            
            logger.info("Chunked %s into %d paragraphs using Chonky", path, len(chunks))
            
            # Return in requested format
            if output_format == "json":
                return json.dumps(chunks, indent=2)
            
            return chunks
            
        except Exception as e:
            logger.error("Error in Chonky chunking: %s", str(e))
            # Fall back to basic splitting if Chonky fails
            logger.warning("Falling back to basic paragraph splitting for %s", path)
            basic_paragraphs = [p for p in source.split("\n\n") if p.strip()]
            
            fallback_chunks: List[Dict[str, Any]] = []
            parent_id = f"doc:{_hash_path(path)}"
            
            for idx, para in enumerate(basic_paragraphs):
                if not para.strip():
                    continue
                chunk_id = f"chunk:{uuid.uuid4().hex[:8]}"
                fallback_chunks.append({
                    "id": chunk_id,
                    "parent": parent_id,
                    "path": path,
                    "type": document.get("type", "markdown"),
                    "content": para,
                    "symbol_type": "paragraph",
                    "name": f"paragraph_{idx}",
                    "line_start": 0,
                    "line_end": 0,
                })
            
            logger.warning("Fallback chunking produced %d chunks for %s", len(fallback_chunks), path)
            
            if output_format == "json":
                return json.dumps(fallback_chunks, indent=2)
            
            return fallback_chunks

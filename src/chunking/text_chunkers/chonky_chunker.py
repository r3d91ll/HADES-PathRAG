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
import uuid
import logging
import json
import os
import hashlib
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
_TOKENIZER = None
_SPLITTER_CACHE: Dict[str, ParagraphSplitter] = {}

# Disable mypy for this function since it has complex type checking
# mypy: disable-error-code="unreachable"
def _check_model_engine_availability() -> bool:
    """Check if the model engine is available at module initialization.
    
    Returns:
        bool: True if the model engine is available, False otherwise
    """
    # Get chunker config to check if early availability check is enabled
    config = get_chunker_config('chonky')
    if not config.get('early_availability_check', True):
        logger.debug("Early model engine availability check disabled by configuration")
        return False
        
    try:
        # Use the existing HaystackModelEngine class
        engine = HaystackModelEngine()
        
        # Get status and check if engine is running
        try:
            status_dict = engine.status()
            if not status_dict:
                logger.warning("Empty status response from model engine")
                return False
                
            # Check running status if status_dict is a dict
            if isinstance(status_dict, dict) and status_dict.get("running") is True:
                return True
        except Exception as status_err:
            logger.warning("Error getting model engine status: %s", status_err)
            return False
        
        # Check if auto-start is enabled
        if not config.get('auto_start_engine', True):
            logger.info("Auto-start disabled by configuration, not starting model engine")
            return False
            
        # Engine is not running, try to start it
        logger.info("Haystack model engine not running, attempting to start...")
        
        # Get max retry count from config
        max_retries = config.get('max_startup_retries', 3)
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                result = engine.start()
                
                # Check if start was successful
                if isinstance(result, dict) and result.get("status") == "ok":
                    logger.info("Successfully started Haystack model engine")
                    return True
                    
                logger.warning("Failed to start Haystack model engine (attempt %d/%d): %s", 
                              retry_count + 1, max_retries, result)
                retry_count += 1
            except Exception as start_err:
                logger.warning("Error starting model engine (attempt %d/%d): %s", 
                              retry_count + 1, max_retries, start_err)
                retry_count += 1
                
        logger.error("Failed to start Haystack model engine after %d attempts", max_retries)
        return False
    except Exception as e:
        logger.warning("Error checking model engine availability: %s", e)
        return False

# Run availability check at module initialization
_ENGINE_AVAILABLE = _check_model_engine_availability()


def get_model_engine() -> HaystackModelEngine:
    """Get the global model engine instance, initializing if necessary.
    
    Returns:
        HaystackModelEngine: The initialized model engine
    """
    global _MODEL_ENGINE  # pylint: disable=global-statement
    if _MODEL_ENGINE is None:
        _MODEL_ENGINE = HaystackModelEngine()
        # Start the service if not already running
        status = _MODEL_ENGINE.status()
        is_running = False
        if isinstance(status, dict):
            is_running = status.get("running", False)
        if not is_running:
            logger.info("Starting Haystack model engine service")
            _MODEL_ENGINE.start()
    return _MODEL_ENGINE


@lru_cache(maxsize=4)
def get_tokenizer(model_id: str) -> Any:
    """Get a tokenizer for the specified model ID with caching.
    
    Args:
        model_id: The model ID to load the tokenizer for
        
    Returns:
        The loaded tokenizer
        
    Raises:
        ImportError: If transformers package is not installed
        RuntimeError: If tokenizer fails to load
    """
    if AutoTokenizer is None:
        raise ImportError(
            "Package 'transformers' is required for token counting.\n"
            "Install with:  pip install transformers"
        )
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if tokenizer is None:
            raise RuntimeError(f"Failed to load tokenizer for model {model_id}")
        return tokenizer
    except Exception as e:
        logger.error(f"Error loading tokenizer for model {model_id}: {e}")
        raise RuntimeError(f"Failed to load tokenizer for model {model_id}: {e}") from e


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


def _get_splitter_with_engine(model_id: str, device: str = "cuda") -> ParagraphSplitter:
    """Get a Chonky ParagraphSplitter using the Haystack model engine.
    
    This function ensures the model is loaded in the Haystack engine and
    returns a ParagraphSplitter configured to use it.
    
    Args:
        model_id: The model ID to load
        device: The device to use (e.g., "cuda", "cpu")
        
    Returns:
        A configured ParagraphSplitter instance
        
    Raises:
        RuntimeError: If the model fails to load
    """
    # Get chunker config to check if device-specific caching is enabled
    config = get_chunker_config('chonky')
    cache_with_device = config.get('cache_with_device', True)
    
    # Create cache key based on configuration
    if cache_with_device:
        cache_key = f"{model_id}_{device}"    
    else:
        cache_key = model_id
        
    if cache_key in _SPLITTER_CACHE:
        logger.debug("Using cached ParagraphSplitter for %s", cache_key)
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
) -> Union[List[Dict[str, Any]], str]:
    """Chunk a text document into semantic paragraphs using Chonky.
    
    This function processes a text document and splits it into semantic chunks using the Chonky
    neural chunking model. It preserves the original text casing and formatting while identifying
    natural paragraph and section boundaries.
    
    The function supports two modes of operation:
    1. With Chonky model engine: Uses neural-based semantic chunking
    2. Fallback mode: Uses basic paragraph splitting if Chonky is unavailable
    
    Configuration is loaded from src/config/chunker_config.yaml and can be customized to control
    chunking behavior, overlap context, caching, and model engine settings.
    
    Args:
        document: Document dictionary with the following keys:
            - content: The text content of the document
            - path: Path to the document (used for ID generation)
            - type: Document type (e.g., "markdown", "text")
        max_tokens: Maximum tokens per chunk (default: 2048)
        output_format: Output format, either "python" for Python objects or "json" for JSON string
        
    Returns:
        If output_format is "python": List of chunk dictionaries with the following structure:
            - id: Unique chunk ID
            - parent: Parent document ID
            - path: Document path
            - type: Document type
            - content: Original chunk text
            - overlap_context: Dictionary with pre/post context and position information
            - symbol_type: Always "paragraph" for text chunks
            - name: Paragraph identifier (e.g., "paragraph_0")
            - line_start/line_end: Always 0 for text chunks
            - token_count: Number of tokens in the chunk
            - content_hash: MD5 hash of the content
            - embedding: None (placeholder for future embedding)
        If output_format is "json": JSON string representation of the above
        
    Raises:
        ValueError: If document is missing required fields or has invalid content
        RuntimeError: If there are issues with the model engine (will fall back to basic chunking)
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
            
            # Track the character positions in the original text
            original_text = source
            boundaries = []
            current_pos = 0
            
            # Initialize the list to store processed paragraphs for fallback cases
            processed_paragraphs: List[str] = []
            
            # Find the boundaries of each paragraph in the original text
            for para in paragraphs:
                if not para.strip():
                    continue
                
                # Normalize the paragraph text for comparison (lowercase)
                para_lower = para.lower()
                
                # Find this paragraph in the original text
                # Start searching from the current position to handle repeated text
                search_pos = original_text.lower().find(para_lower, current_pos)
                
                if search_pos >= 0:
                    # Found the paragraph in the original text
                    start_pos = search_pos
                    end_pos = start_pos + len(para)
                    boundaries.append((start_pos, end_pos))
                    current_pos = end_pos
                else:
                    # Fallback: use the Chonky output if we can't find the exact match
                    logger.warning("Could not find exact match for paragraph in original text, using Chonky output")
                    processed_paragraphs.append(para)
                    continue
            
            # Extract the original text using the boundaries
            original_paragraphs = [original_text[start:end] for start, end in boundaries]
            
            # Reset processed_paragraphs for token-aware splitting
            processed_paragraphs = []
            
            # Apply token-aware splitting to handle long paragraphs
            for para in original_paragraphs:
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
            
            # Calculate overlap size in characters
            overlap_chars = cfg.get("overlap_chars", 200)
            
            # Create a mapping of paragraph index to position in original text
            para_positions = []
            current_pos = 0
            for para in processed_paragraphs:
                para_len = len(para)
                para_positions.append((current_pos, current_pos + para_len))
                current_pos += para_len
            
            # Combine all processed paragraphs into a single text for easier overlap handling
            full_text = "".join(processed_paragraphs)
            
            for idx, para in enumerate(processed_paragraphs):
                if not para.strip():
                    continue
                    
                # Generate unique chunk ID
                chunk_id = f"chunk:{uuid.uuid4().hex[:8]}"
                
                # Get position of this paragraph in the full text
                start_pos, end_pos = para_positions[idx]
                
                # Calculate overlap positions
                overlap_start = max(0, start_pos - overlap_chars)
                overlap_end = min(len(full_text), end_pos + overlap_chars)
                
                # Get overlap context configuration
                config = get_chunker_config('chonky')
                overlap_context_config = config.get('overlap_context', {})
                use_overlap_context = overlap_context_config.get('enabled', True)
                store_pre_context = overlap_context_config.get('store_pre_context', True)
                store_post_context = overlap_context_config.get('store_post_context', True)
                max_pre_context_chars = overlap_context_config.get('max_pre_context_chars', 1000)
                max_post_context_chars = overlap_context_config.get('max_post_context_chars', 1000)
                store_position_info = overlap_context_config.get('store_position_info', True)
                
                # Calculate overlap positions with limits
                pre_context_start = max(0, start_pos - min(overlap_chars, max_pre_context_chars))
                pre_context_end = start_pos
                post_context_start = end_pos
                post_context_end = min(len(full_text), end_pos + min(overlap_chars, max_post_context_chars))
                
                # Count tokens for metadata
                token_count = _count_tokens(para, tokenizer)
                
                # Generate a content hash for efficient storage and retrieval
                content_hash = hashlib.md5(para.encode()).hexdigest()
                
                # Create the base chunk metadata
                chunk_metadata = {
                    "id": chunk_id,
                    "parent": parent_id,
                    "path": path,
                    "type": document.get("type", "markdown"),
                    "content": para,  # Store the original paragraph without overlap
                    "symbol_type": "paragraph",
                    "name": f"paragraph_{idx}",
                    "line_start": 0,
                    "line_end": 0,
                    "token_count": token_count,
                    "content_hash": content_hash,
                    "embedding": None,  # Placeholder for future embedding
                }
                
                # Add overlap context if enabled
                if use_overlap_context:
                    overlap_context = {}
                    
                    # Add pre-context if enabled
                    if store_pre_context and pre_context_end > pre_context_start:
                        overlap_context["pre_context"] = full_text[pre_context_start:pre_context_end]
                    
                    # Add post-context if enabled
                    if store_post_context and post_context_end > post_context_start:
                        overlap_context["post_context"] = full_text[post_context_start:post_context_end]
                    
                    # Add position information if enabled
                    if store_position_info:
                        overlap_context["pre_context_start"] = pre_context_start
                        overlap_context["pre_context_end"] = pre_context_end
                        overlap_context["post_context_start"] = post_context_start
                        overlap_context["post_context_end"] = post_context_end
                    
                    # Add overlap context to chunk metadata
                    chunk_metadata["overlap_context"] = overlap_context
                else:
                    # For backward compatibility, include the full content with overlap
                    content_with_overlap = full_text[pre_context_start:post_context_end]
                    content_offset = start_pos - pre_context_start
                    content_length = end_pos - start_pos
                    
                    chunk_metadata["content_with_overlap"] = content_with_overlap
                    chunk_metadata["content_offset"] = content_offset
                    chunk_metadata["content_length"] = content_length
                
                # Add the chunk to the list
                chunks.append(chunk_metadata)
            
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

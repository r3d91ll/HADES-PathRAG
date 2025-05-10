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
- Direct support for Pydantic model instances
"""

from typing import Any, Dict, List, Optional, Union, Tuple, Generator, Callable, Type, cast
import hashlib
import uuid
import logging
import json
import os
import torch
from functools import lru_cache
from multiprocessing.pool import ThreadPool
from contextlib import contextmanager

from pydantic import BaseModel

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

# Import schema models
from src.schema.document_schema import DocumentSchema, ChunkMetadata
from src.docproc.schemas.base import BaseDocument

logger = logging.getLogger(__name__)

# Initialize global model engine and tokenizer
_MODEL_ENGINE: Optional[HaystackModelEngine] = None
_TOKENIZER = None
_SPLITTER_CACHE = {}


def get_model_engine() -> HaystackModelEngine:
    """Get or initialize the Haystack model engine."""
    global _MODEL_ENGINE
    if _MODEL_ENGINE is None:
        from src.model_engine.engines.haystack import create_haystack_engine
        _MODEL_ENGINE = create_haystack_engine()
        # Start the engine if not already running
        if not _MODEL_ENGINE.status().get("running", False):
            _MODEL_ENGINE.start()
    return _MODEL_ENGINE


def _get_splitter_with_engine(model_id: str, device: str = "cuda") -> ParagraphSplitter:
    """Get a Chonky ParagraphSplitter using the Haystack model engine.
    
    This function ensures the model is loaded in the Haystack engine and
    returns a ParagraphSplitter configured to use it.
    """
    cache_key = f"{model_id}_{device}"
    if cache_key in _SPLITTER_CACHE:
        return _SPLITTER_CACHE[cache_key]
    
    engine = get_model_engine()
    
    # Load the model in the engine
    load_result = engine.load_model(model_id)
    
    # Check if the model was loaded successfully
    success = False
    error_msg = "Unknown error"
    
    if isinstance(load_result, dict):
        success = load_result.get("success", False)
        error_msg = load_result.get("error", "Unknown error")
    else:
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
    document: Union[Dict[str, Any], BaseDocument, DocumentSchema], 
    *, 
    max_tokens: int = 2048, 
    output_format: str = "python"
) -> Union[List[Dict[str, Any]], List[ChunkMetadata], str]:  # noqa: D401
    """Split plain-text/Markdown document into semantic paragraphs using Chonky.

    Parameters
    ----------
    document:
        Pre-processed document dict or Pydantic model instance. 
        Must contain ``content`` and ``path`` keys or attributes.
    max_tokens:
        Token budget for each chunk (default comes from config).
    output_format:
        Output format of the chunks. Can be "python" (dict), "pydantic" (ChunkMetadata), or "json" (str).

    Returns
    -------
    Union[List[Dict[str, Any]], List[ChunkMetadata], str]
        List of chunk dictionaries, ChunkMetadata instances, or JSON string, depending on output_format
    """
    # Load configuration
    cfg = get_chunker_config("chonky")
    
    # Override max_tokens from config if provided
    if max_tokens <= 0:
        max_tokens = cfg.get("max_tokens", 2048)
    
    # Extract content and path from document
    if isinstance(document, BaseModel):
        # Handle Pydantic model instance
        content = getattr(document, "content", "")
        path = getattr(document, "path", "") or getattr(document, "source", "")
        doc_id = getattr(document, "id", "")
    else:
        # Handle dictionary
        content = document.get("content", "")
        path = document.get("path", "") or document.get("source", "")
        doc_id = document.get("id", "")
    
    # Ensure we have content to chunk
    if not content:
        logger.warning("No content to chunk in document %s", doc_id or path)
        return [] if output_format != "json" else "[]"
    
    # Create a short hash from the path for chunk IDs
    path_hash = _hash_path(str(path))
    
    # Get the paragraph splitter
    model_id = cfg.get("model_id", "mirth/chonky_modernbert_large_1")
    device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    
    with ensure_model_engine():
        splitter = _get_splitter_with_engine(model_id, device)
        
        # Split the content into paragraphs
        paragraphs = splitter(content)
    
    # Create chunks from paragraphs
    chunks = []
    for i, para in enumerate(paragraphs):
        chunk = {
            "id": f"chunk_{path_hash}_{i:04d}",
            "content": para,
            "symbol_type": "paragraph",
            "name": f"paragraph_{i}",
            "token_count": len(para.split())  # Approximate token count
        }
        chunks.append(chunk)
    
    # Convert to ChunkMetadata instances if requested
    if output_format == "pydantic":
        chunk_metadata_list = []
        for i, chunk in enumerate(chunks):
            chunk_metadata = ChunkMetadata(
                start_offset=0,  # Placeholder, would be calculated from actual position
                end_offset=len(chunk["content"]),
                chunk_type="text",
                chunk_index=i,
                parent_id=doc_id,
                metadata={
                    "content": chunk["content"],
                    "symbol_type": chunk["symbol_type"],
                    "name": chunk["name"],
                    "token_count": chunk["token_count"]
                }
            )
            chunk_metadata_list.append(chunk_metadata)
        return chunk_metadata_list
    
    # Convert to JSON string if requested
    if output_format == "json":
        return json.dumps(chunks, ensure_ascii=False)
    
    # Return as list of dictionaries (default)
    return chunks


def chunk_document(
    document: Union[Dict[str, Any], BaseDocument, DocumentSchema],
    *,
    max_tokens: int = 2048,
    return_pydantic: bool = True,
    save_to_disk: bool = False,
    output_dir: Optional[str] = None
) -> Union[DocumentSchema, Dict[str, Any]]:
    """
    Chunk a document and update it with the generated chunks.
    
    This function processes a document (either as a dictionary or Pydantic model),
    generates chunks using the appropriate chunker, and returns an updated document
    with the chunks attached.
    
    Parameters
    ----------
    document:
        Document to chunk (dictionary or Pydantic model)
    max_tokens:
        Maximum tokens per chunk
    return_pydantic:
        Whether to return a Pydantic model (True) or dictionary (False)
    save_to_disk:
        Whether to save the chunked document to disk (for debugging)
    output_dir:
        Directory to save the chunked document (if save_to_disk is True)
        
    Returns
    -------
    Union[DocumentSchema, Dict[str, Any]]
        Updated document with chunks attached
    """
    # Convert to dictionary if it's a Pydantic model
    if isinstance(document, BaseModel):
        doc_dict = document.model_dump()
        doc_id = getattr(document, "id", "unknown")
        doc_type = getattr(document, "document_type", "text")
    else:
        doc_dict = document
        doc_id = document.get("id", "unknown")
        doc_type = document.get("document_type", "text")
    
    # Choose the appropriate chunker based on document type
    if doc_type == "code" or doc_dict.get("format") == "python":
        from src.chunking import chunk_code
        chunks = chunk_code(doc_dict, max_tokens=max_tokens, output_format="python")
    else:
        chunks = chunk_text(doc_dict, max_tokens=max_tokens, output_format="python")
    
    # Convert chunks to ChunkMetadata format
    chunk_metadata_list = []
    for i, chunk in enumerate(chunks):
        chunk_content = chunk.get("content", "")
        chunk_metadata = {
            "start_offset": chunk.get("line_start", 0) if "line_start" in chunk else 0,
            "end_offset": chunk.get("line_end", 0) if "line_end" in chunk else len(chunk_content),
            "chunk_type": "code" if doc_type == "code" else "text",
            "chunk_index": i,
            "parent_id": doc_id,
            "metadata": {
                "content": chunk_content,
                "symbol_type": chunk.get("symbol_type", "paragraph"),
                "name": chunk.get("name", f"chunk_{i}"),
                "token_count": chunk.get("token_count", 0),
                "line_start": chunk.get("line_start", 0) if "line_start" in chunk else None,
                "line_end": chunk.get("line_end", 0) if "line_end" in chunk else None
            }
        }
        chunk_metadata_list.append(chunk_metadata)
    
    # Update the document with chunks
    doc_dict["chunks"] = chunk_metadata_list
    
    # Save to disk if requested (for debugging)
    if save_to_disk and output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{doc_id}_chunked.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(doc_dict, f, indent=2, ensure_ascii=False)
    
    # Return as Pydantic model or dictionary
    if return_pydantic:
        return DocumentSchema.model_validate(doc_dict)
    return doc_dict

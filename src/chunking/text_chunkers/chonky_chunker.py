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

# Define our own document models to avoid import issues
class BaseDocument(BaseModel):
    """Base document class for document processing."""
    content: str
    path: str
    type: str = "text"
    id: str = ""
    chunks: List[Dict[str, Any]] = []

class ChunkMetadata(BaseModel):
    """Metadata for document chunks."""
    id: str
    parent: str
    content: str
    path: str
    type: str = "text"
    overlap_context: Optional[Dict[str, Any]] = None
    symbol_type: str = "paragraph"
    name: str = ""
    line_start: int = 0
    line_end: int = 0
    token_count: int = 0
    content_hash: str = ""
    embedding: Optional[List[float]] = None

# Define our document types
from typing import Type, TypeVar, cast, Any as TypeAny, Union as TypeUnion

# Initial type definition for document types
DocumentBaseType = TypeUnion[Dict[str, TypeAny], BaseDocument]

# Define a type for DocumentSchema
class DocumentSchemaBase(BaseModel):
    """Base document schema for document processing."""
    content: str
    path: str
    type: str = "text"
    id: str = ""
    chunks: List[Dict[str, TypeAny]] = []

# Try to import DocumentSchema from schema module
try:
    from src.schema.document_schema import DocumentSchema as ImportedDocumentSchema
    # Use the imported DocumentSchema
    DocumentSchema = ImportedDocumentSchema
except ImportError:
    # Use our own DocumentSchema if not available
    class DocumentSchema(DocumentSchemaBase):  # type: ignore
        pass

# Final type definition including DocumentSchema
DocumentSchemaType = TypeUnion[Dict[str, TypeAny], BaseDocument, DocumentSchema]

logger = logging.getLogger(__name__)

# Initialize global model engine
_MODEL_ENGINE: Optional[HaystackModelEngine] = None
_TOKENIZER = None
_SPLITTER_CACHE: Dict[str, ParagraphSplitter] = {}
_ENGINE_AVAILABLE = False


def get_model_engine() -> Optional[HaystackModelEngine]:
    """Get or initialize the Haystack model engine.
    
    Returns:
        The initialized model engine, or None if initialization fails
    """
    global _MODEL_ENGINE, _ENGINE_AVAILABLE
    
    # Return existing engine if already initialized
    if _MODEL_ENGINE is not None:
        return _MODEL_ENGINE
    
    # Default to engine not available    
    _ENGINE_AVAILABLE = False
    
    try:
        # Create a new model engine instance
        _MODEL_ENGINE = HaystackModelEngine()
        logger.info("Created Haystack model engine instance")
        
        # Get chunker configuration
        config = get_chunker_config('chonky')
        
        # Check if auto-start is enabled
        auto_start = config.get('auto_start_engine', True)
        if not auto_start:
            logger.info("Auto-start disabled, returning unstarted engine")
            return _MODEL_ENGINE
        
        # Try to start the engine
        start_success = False
        try:
            _MODEL_ENGINE.start()
            logger.info("Started Haystack model engine")
            start_success = True
        except Exception as e:
            logger.warning(f"Failed to start Haystack model engine: {e}")
        
        if not start_success:
            return _MODEL_ENGINE
            
        # Client checks
        has_client = hasattr(_MODEL_ENGINE, "client")
        if not has_client:
            logger.warning("Haystack model engine has no client attribute")
            return _MODEL_ENGINE
            
        client_exists = _MODEL_ENGINE.client is not None
        if not client_exists:
            logger.warning("Haystack model engine client is None")
            return _MODEL_ENGINE
            
        # Get client reference (safe now that we've checked)
        client = _MODEL_ENGINE.client
        
        # Check ping method
        has_ping = hasattr(client, "ping")
        if not has_ping:
            logger.warning("Haystack model engine client has no ping method")
            return _MODEL_ENGINE
            
        # Try to ping
        ping_success = False
        try:
            # We've already checked client exists and has ping attribute
            # Just need to check if it's callable
            ping_method = getattr(client, "ping")
            if callable(ping_method):
                ping_result = ping_method()
                if ping_result == "pong":
                    logger.info("Haystack model engine is available")
                    _ENGINE_AVAILABLE = True
                    ping_success = True
                else:
                    logger.warning(f"Haystack model engine ping returned {ping_result}")
            else:
                logger.warning("Ping method is not callable")
        except Exception as e:
            logger.warning(f"Haystack model engine ping failed: {e}")
            
        # Return the engine regardless of ping result
        return _MODEL_ENGINE
        
    except Exception as e:
        logger.warning(f"Error initializing Haystack model engine: {e}")
        _MODEL_ENGINE = None
        return None


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
    global _SPLITTER_CACHE
    
    # Check if we have a cached splitter for this model
    cache_key = f"{model_id}_{device}"
    if cache_key in _SPLITTER_CACHE:
        return _SPLITTER_CACHE[cache_key]
    
    # Get chunker configuration
    config = get_chunker_config('chonky')
    
    # Check if caching is enabled
    use_cache = config.get('use_cache', True)
    
    # Get model engine
    engine = get_model_engine()
    if engine is None:
        raise RuntimeError("Failed to initialize model engine")
    
    # Load the model
    try:
        # Log model loading
        logger.info(f"Loading model {model_id} on {device}")
        
        # Load the model with the engine
        result = engine.load_model(model_id, device=device)
        logger.info(f"Model loading result: {result}")
        
        # Create the splitter
        splitter = ParagraphSplitter(model_id)
        
        # Cache the splitter if caching is enabled
        if use_cache:
            _SPLITTER_CACHE[cache_key] = splitter
        
        return splitter
    except Exception as e:
        logger.error(f"Error loading model {model_id}: {e}")
        raise RuntimeError(f"Failed to load model {model_id}: {e}")


def _hash_path(path: str) -> str:
    """Create a short hash from a file path for chunk IDs."""
    return hashlib.md5(path.encode()).hexdigest()[:8]


@contextmanager
def ensure_model_engine() -> Generator[Optional[HaystackModelEngine], None, None]:
    """Context manager to ensure model engine is started and properly cleaned up."""
    engine = get_model_engine()
    try:
        yield engine
    finally:
        # We don't stop the engine here to allow reuse
        pass


def chunk_text(
    document: Union[Dict[str, Any], BaseDocument, DocumentSchema], 
    *, 
    max_tokens: int = 2048, 
    output_format: str = "python"
) -> Union[List[Dict[str, Any]], str]:
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
    # Get chunker configuration
    config = get_chunker_config('chonky')
    
    # Extract document content and path
    if isinstance(document, dict):
        content = document.get("content", "")
        path = document.get("path", "unknown")
        doc_type = document.get("type", "text")
        doc_id = document.get("id", str(uuid.uuid4()))
    else:
        # Handle Pydantic model instance
        content = getattr(document, "content", "")
        path = getattr(document, "path", "unknown")
        doc_type = getattr(document, "type", "text")
        doc_id = getattr(document, "id", str(uuid.uuid4()))
    
    # Validate content
    if not content or not isinstance(content, str):
        raise ValueError("Document must contain non-empty string content")
    
    # Generate a document ID if not provided
    if not doc_id:
        doc_id = f"doc_{_hash_path(path)}"
    
    # Use the configured model ID or default
    model_id = config.get('model_id', 'mirth/chonky_modernbert_large_1')
    
    # Use the configured device or default to CUDA
    device = config.get('device', 'cuda')
    
    # Get the configured max tokens or use the provided value
    max_tokens = config.get('max_tokens', max_tokens)
    
    # Determine if we should use the model engine
    use_engine = config.get('use_model_engine', True) and _ENGINE_AVAILABLE
    
    # Initialize chunks list
    chunks = []
    
    try:
        if use_engine:
            # Use the model engine for chunking
            with ensure_model_engine():
                # Get the splitter
                splitter = _get_splitter_with_engine(model_id, device)
                
                # Split the document
                paragraphs = splitter.split(content)
                
                # Create chunks from paragraphs
                for i, para in enumerate(paragraphs):
                    # Generate a unique ID for the chunk
                    chunk_id = f"{doc_id}_p{i}"
                    
                    # Estimate start/end offsets based on paragraph position
                    # This is approximate since we don't have exact character positions from the splitter
                    if i > 0 and i-1 < len(paragraphs):
                        prev_content_length = sum(len(p.text) for p in paragraphs[:i])
                    else:
                        prev_content_length = 0
                        
                    start_offset = prev_content_length
                    end_offset = start_offset + len(para.text)
                    
                    # Create the chunk
                    chunk = {
                        "id": chunk_id,
                        "parent": doc_id,
                        "parent_id": doc_id,  # Required by schema
                        "path": path,
                        "type": doc_type,
                        "content": para.text,
                        "overlap_context": {
                            "pre": para.pre_context,
                            "post": para.post_context,
                            "position": i,
                            "total": len(paragraphs)
                        },
                        "symbol_type": "paragraph",
                        "name": f"paragraph_{i}",
                        "chunk_index": i,  # Required by schema
                        "start_offset": start_offset,  # Required by schema
                        "end_offset": end_offset,  # Required by schema 
                        "line_start": 0,
                        "line_end": 0,
                        "token_count": len(para.text.split()),
                        "content_hash": hashlib.md5(para.text.encode()).hexdigest(),
                        "embedding": None
                    }
                    
                    chunks.append(chunk)
        else:
            # Fallback to basic paragraph splitting
            logger.warning("Model engine not available, using fallback chunking")
            
            # Split by paragraphs (double newlines)
            paragraphs = [p for p in content.split("\n\n") if p.strip()]
            
            # Create chunks from paragraphs
            cumulative_length = 0
            for i, para in enumerate(paragraphs):
                # Generate a unique ID for the chunk
                chunk_id = f"{doc_id}_p{i}"
                
                # Calculate offsets
                start_offset = cumulative_length
                end_offset = start_offset + len(para)
                cumulative_length = end_offset + 2  # +2 for the removed "\n\n"
                
                # Create the chunk
                chunk = {
                    "id": chunk_id,
                    "parent": doc_id,
                    "parent_id": doc_id,  # Required by schema
                    "path": path,
                    "type": doc_type,
                    "content": para,
                    "overlap_context": {
                        "pre": "",
                        "post": "",
                        "position": i,
                        "total": len(paragraphs)
                    },
                    "symbol_type": "paragraph",
                    "name": f"paragraph_{i}",
                    "chunk_index": i,  # Required by schema
                    "start_offset": start_offset,  # Required by schema
                    "end_offset": end_offset,  # Required by schema
                    "line_start": 0,
                    "line_end": 0,
                    "token_count": len(para.split()),
                    "content_hash": hashlib.md5(para.encode()).hexdigest(),
                    "embedding": None
                }
                
                chunks.append(chunk)
    except Exception as e:
        logger.error(f"Error chunking document: {e}")
        # Fallback to basic paragraph splitting
        logger.warning("Using fallback chunking due to error")
        
        # Split by paragraphs (double newlines)
        paragraphs = [p for p in content.split("\n\n") if p.strip()]
        
        # Create chunks from paragraphs
        for i, para in enumerate(paragraphs):
            # Generate a unique ID for the chunk
            chunk_id = f"{doc_id}_p{i}"
            
            # Create the chunk
            chunk = {
                "id": chunk_id,
                "parent": doc_id,
                "path": path,
                "type": doc_type,
                "content": para,
                "overlap_context": {
                    "pre": "",
                    "post": "",
                    "position": i,
                    "total": len(paragraphs)
                },
                "symbol_type": "paragraph",
                "name": f"paragraph_{i}",
                "line_start": 0,
                "line_end": 0,
                "token_count": len(para.split()),
                "content_hash": hashlib.md5(para.encode()).hexdigest(),
                "embedding": None
            }
            
            chunks.append(chunk)
    
    # Handle different output formats
    if output_format == "json":
        return json.dumps(chunks)
    else:
        # Default to Python dictionaries
        return chunks


def chunk_document_to_json(
    document: Union[Dict[str, Any], BaseDocument],
    *,
    max_tokens: int = 2048,
    save_to_disk: bool = False,
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """Process a document and return a JSON dict.

    Args:
        document: The document to process
        max_tokens: Maximum number of tokens per chunk
        save_to_disk: Whether to save the document to disk
        output_dir: Directory to save the document to

    Returns:
        A dict with the document and chunks
    """
    # Process the document
    result = chunk_document(
        document,
        max_tokens=max_tokens,
        return_pydantic=False,
        save_to_disk=save_to_disk,
        output_dir=output_dir
    )

    # Convert to dict if needed
    if isinstance(result, dict):
        return result
    elif hasattr(result, 'dict') and callable(getattr(result, 'dict')):
        # Convert BaseDocument or DocumentSchema to dict
        return result.dict()
    else:
        # Should never happen, but just in case
        raise TypeError(f"Unexpected result type: {type(result)}")


def chunk_document_to_schema(
    document: Union[Dict[str, Any], BaseDocument],
    *,
    max_tokens: int = 2048,
    save_to_disk: bool = False,
    output_dir: Optional[str] = None
) -> DocumentSchema:
    """Process a document and return a DocumentSchema.

    Args:
        document: The document to process
        max_tokens: Maximum number of tokens per chunk
        save_to_disk: Whether to save the document to disk
        output_dir: Directory to save the document to

    Returns:
        A DocumentSchema with the document and chunks
    """
    # Process the document
    result = chunk_document(
        document,
        max_tokens=max_tokens,
        return_pydantic=True,
        save_to_disk=save_to_disk,
        output_dir=output_dir
    )

    # Convert to DocumentSchema if needed
    if isinstance(result, dict):
        return DocumentSchema(**result)
    elif hasattr(result, 'dict') and callable(getattr(result, 'dict')):
        # Convert BaseDocument to dict, then to DocumentSchema
        return DocumentSchema(**result.dict())
    elif isinstance(result, DocumentSchema):
        # Already a DocumentSchema
        return result
    else:
        # Unexpected type, create a default DocumentSchema
        logger.warning(f"Unexpected result type in chunk_document_to_schema: {type(result)}")
        # Create a minimal valid DocumentSchema with required fields
        # Use a cast to help mypy understand the type
        return cast(DocumentSchema, DocumentSchemaBase(content="", path=""))


def chunk_document(
    document: Union[Dict[str, Any], BaseDocument], 
    *, 
    max_tokens: int = 2048, 
    return_pydantic: bool = True, 
    save_to_disk: bool = False, 
    output_dir: Optional[str] = None
) -> Union[Dict[str, Any], BaseDocument, DocumentSchema]:
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
    # Convert to dictionary if needed
    if not isinstance(document, dict):
        doc_dict = document.model_dump() if hasattr(document, "model_dump") else document.dict() if hasattr(document, "dict") else {
            k: getattr(document, k) for k in dir(document) 
            if not k.startswith("_") and not callable(getattr(document, k))
        }
    else:
        doc_dict = document.copy()
    
    # Get document type
    doc_type = doc_dict.get("type", "text")
    
    # Choose the appropriate chunker based on document type
    if doc_type in ["python", "js", "typescript", "java", "c", "cpp", "csharp"]:
        # For code documents, use the code chunker
        from src.chunking.code_chunkers.ast_chunker import chunk_python_code
        chunks = chunk_python_code(doc_dict, max_tokens=max_tokens)
    else:
        # For text documents, use the text chunker
        chunks = chunk_text(doc_dict, max_tokens=max_tokens)
    
    # Update the document with chunks
    doc_dict["chunks"] = chunks
    
    # Save to disk if requested
    if save_to_disk:
        if output_dir is None:
            output_dir = "chunks"
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate a filename
        doc_id = doc_dict.get("id", "unknown")
        filename = f"{doc_id}.json"
        filepath = os.path.join(output_dir, filename)
        
        # Save to disk
        with open(filepath, "w") as f:
            json.dump(doc_dict, f, indent=2)
        
        logger.info(f"Saved chunked document to {filepath}")
    
    # Return as Pydantic model if requested
    if return_pydantic:
        try:
            # If we're using our own DocumentSchema, we can create it directly
            if hasattr(DocumentSchema, "__module__") and DocumentSchema.__module__ == __name__:
                return DocumentSchema(**doc_dict)
            else:
                # We're using the imported DocumentSchema, which might have different field requirements
                # Try to adapt our document to match the expected schema
                adapted_dict = {
                    "content": doc_dict.get("content", ""),
                    "path": doc_dict.get("path", ""),
                    "id": doc_dict.get("id", ""),
                    "chunks": doc_dict.get("chunks", []),
                    # Add any additional fields that might be required by the imported schema
                    "source": doc_dict.get("path", ""),  # Use path as source if not provided
                    "document_type": doc_dict.get("type", "text"),  # Use type as document_type
                }
                # Create the document schema instance
                return DocumentSchema(**adapted_dict)
        except Exception as e:
            logger.error(f"Error converting to Pydantic model: {e}")
            # Fall back to dictionary
            return doc_dict
    else:
        return doc_dict

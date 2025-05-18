"""
Embedding processors for the HADES-PathRAG system.

This module contains functions for processing chunked documents by adding
embeddings to their chunks. It provides the bridge between the chunking and
storage stages of the ingestion pipeline.
"""

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple, cast, Iterable

import numpy as np

from src.types.common import EmbeddingVector
from src.embedding.base import EmbeddingAdapter, get_adapter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def add_embeddings_to_document(
    document: Dict[str, Any],
    adapter_name: str = "cpu",
    adapter_options: Optional[Dict[str, Any]] = None,
    chunk_content_key: str = "content",
    embedding_key: str = "embedding"
) -> Dict[str, Any]:
    """Add embeddings to a document's chunks.
    
    Args:
        document: Document dictionary with chunks
        adapter_name: Name of the embedding adapter to use
        adapter_options: Options for the embedding adapter
        chunk_content_key: Key in each chunk containing the content to embed
        embedding_key: Key to store the embedding under in each chunk
        
    Returns:
        Document with embeddings added to chunks
    """
    # Create a copy to avoid modifying the original
    document_with_embeddings = document.copy()
    
    # Get chunks from the document
    chunks = document_with_embeddings.get("chunks", [])
    if not chunks:
        logger.warning(f"Document {document.get('id', 'unknown')} has no chunks to embed")
        return document_with_embeddings
    
    # Get adapter with options
    adapter_options = adapter_options or {}
    adapter = get_adapter(adapter_name, **adapter_options)
    
    logger.info(f"Embedding {len(chunks)} chunks from document {document.get('id', 'unknown')}")
    
    # Extract content from chunks
    chunk_contents = [chunk.get(chunk_content_key, "") for chunk in chunks]
    
    # Skip empty contents
    valid_indices = [i for i, content in enumerate(chunk_contents) if content.strip()]
    valid_contents = [chunk_contents[i] for i in valid_indices]
    
    if not valid_contents:
        logger.warning(f"No valid chunk content found in document {document.get('id', 'unknown')}")
        return document_with_embeddings
    
    # Generate embeddings for all chunks
    try:
        embeddings = await adapter.embed(valid_contents)
        
        # Add embeddings to chunks
        for idx, embedding in zip(valid_indices, embeddings):
            chunks[idx][embedding_key] = embedding
        
        # Add a summary of embedding information to the document
        document_with_embeddings["_embedding_info"] = {
            "adapter": adapter_name,
            "chunk_count": len(chunks),
            "embedded_chunk_count": len(valid_indices),
            "embedding_dimensions": len(embeddings[0]) if embeddings else 0,
            "timestamp": str(os.path.getmtime(document.get("path", ""))) if document.get("path") else None
        }
        
        logger.info(f"Added embeddings to {len(valid_indices)} chunks in document {document.get('id', 'unknown')}")
    except Exception as e:
        logger.error(f"Error generating embeddings: {e}")
        # Add error information to document
        document_with_embeddings["_embedding_error"] = str(e)
    
    return document_with_embeddings


async def process_chunked_document_file(
    file_path: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    adapter_name: str = "cpu",
    adapter_options: Optional[Dict[str, Any]] = None
) -> Tuple[Dict[str, Any], Optional[Path]]:
    """Process a chunked document file by adding embeddings.
    
    Args:
        file_path: Path to chunked document JSON file
        output_dir: Directory to save the output (None = don't save)
        adapter_name: Name of the embedding adapter to use
        adapter_options: Options for the embedding adapter
        
    Returns:
        Tuple of (document with embeddings, output file path if saved)
    """
    path_obj = Path(file_path) if isinstance(file_path, str) else file_path
    
    if not path_obj.exists():
        raise FileNotFoundError(f"File not found: {path_obj}")
    
    # Load the chunked document
    with open(path_obj, 'r', encoding='utf-8') as f:
        document = json.load(f)
    
    # Add embeddings to the document
    document_with_embeddings = await add_embeddings_to_document(
        document=document,
        adapter_name=adapter_name,
        adapter_options=adapter_options
    )
    
    # Save the document with embeddings if output_dir is provided
    output_path = None
    if output_dir:
        output_dir_obj = Path(output_dir) if isinstance(output_dir, str) else output_dir
        output_dir_obj.mkdir(parents=True, exist_ok=True)
        
        # Generate output file name
        doc_id = document.get("id", path_obj.stem)
        output_path = output_dir_obj / f"{doc_id}_embedded.json"
        
        # Save to file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(document_with_embeddings, f, indent=2)
        
        logger.info(f"Saved document with embeddings to {output_path}")
    
    return document_with_embeddings, output_path


async def process_chunked_documents_batch(
    file_paths: List[Union[str, Path]],
    output_dir: Optional[Union[str, Path]] = None,
    adapter_name: str = "cpu",
    adapter_options: Optional[Dict[str, Any]] = None,
    batch_size: int = 10,
    max_concurrent: int = 5
) -> Dict[str, Any]:
    """Process multiple chunked document files in batches.
    
    Args:
        file_paths: List of paths to chunked document JSON files
        output_dir: Directory to save the outputs (None = don't save)
        adapter_name: Name of the embedding adapter to use
        adapter_options: Options for the embedding adapter
        batch_size: Number of chunks to process in a batch
        max_concurrent: Maximum number of documents to process concurrently
        
    Returns:
        Dictionary with processing statistics
    """
    logger.info(f"Processing {len(file_paths)} chunked documents with {adapter_name} adapter")
    
    # Prepare output directory if provided
    output_path = None
    if output_dir:
        output_path = Path(output_dir) if isinstance(output_dir, str) else output_dir
        output_path.mkdir(parents=True, exist_ok=True)
    
    # Process documents in batches
    results = []
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_with_semaphore(file_path):
        async with semaphore:
            try:
                document, output_file = await process_chunked_document_file(
                    file_path=file_path,
                    output_dir=output_path,
                    adapter_name=adapter_name,
                    adapter_options=adapter_options
                )
                
                return {
                    "file_path": str(file_path),
                    "document_id": document.get("id", "unknown"),
                    "chunks": len(document.get("chunks", [])),
                    "embedded_chunks": len([c for c in document.get("chunks", []) if "embedding" in c]),
                    "success": True,
                    "output_file": str(output_file) if output_file else None
                }
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                return {
                    "file_path": str(file_path),
                    "error": str(e),
                    "success": False,
                    "output_file": None
                }
    
    # Process all files concurrently (with semaphore to limit concurrency)
    tasks = [process_with_semaphore(file_path) for file_path in file_paths]
    results = await asyncio.gather(*tasks)
    
    # Compile statistics
    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]
    
    stats = {
        "total": len(file_paths),
        "successful": len(successful),
        "failed": len(failed),
        "details": results
    }
    
    # Save statistics if output_dir is provided
    if output_path:
        stats_file = output_path / "embedding_stats.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Saved statistics to {stats_file}")
    
    return stats

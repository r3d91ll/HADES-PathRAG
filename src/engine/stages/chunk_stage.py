"""
Chunking stage.

This module implements the chunking stage of the pipeline.
It uses the Chonky chunker for text documents and the AST chunker for code documents.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union, Set

import torch

from ..batch_types import PipelineBatch
from .base import StageBase
from src.chunking.text_chunkers.chonky_chunker import chunk_text
from src.chunking.code_chunkers.ast_chunker import chunk_python_code
from src.config.chunker_config import get_chunker_config


logger = logging.getLogger(__name__)


class ChunkStage(StageBase):
    """
    Chunking stage.
    
    This stage chunks documents using the appropriate chunker based on document type.
    """
    
    name = "Chunk"
    
    def __init__(
        self, 
        device: Union[str, torch.device],
        max_tokens: int = 2048,
        use_overlap: bool = True,
        **kwargs: Any
    ) -> None:
        """
        Initialize the chunking stage.
        
        Args:
            device: The device to run this stage on
            max_tokens: Maximum tokens per chunk
            use_overlap: Whether to use overlap context for better retrieval
            **kwargs: Additional keyword arguments
        """
        super().__init__(device)
        self.max_tokens = max_tokens
        self.use_overlap = use_overlap
        
        # Load chunker configurations
        self.text_config = get_chunker_config("chonky")
        self.code_config = get_chunker_config("ast")
        
        # Set up document type to chunker mapping
        self.code_types = {"python", "java", "javascript", "typescript", "c", "cpp", "csharp"}
        self.text_types = {
            "text", "markdown", "pdf", "html", "json", "yaml", "toml", "xml", "csv"
        }
    
    async def chunk_document_async(
        self, doc: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Chunk a document asynchronously.
        
        Args:
            doc: The document to chunk
            
        Returns:
            List of chunks
        """
        # Get document type
        doc_type = doc.get("type", "").lower()
        
        # Use the appropriate chunker based on document type
        loop = asyncio.get_event_loop()
        
        if doc_type in self.code_types:
            # Use AST chunker for code documents
            result = await loop.run_in_executor(
                None, 
                lambda d, m, o: chunk_python_code(d, max_tokens=m, output_format=o),
                doc, self.max_tokens, "python"
            )
            return result if isinstance(result, list) else []
        else:
            # Use Chonky chunker for text documents
            result = await loop.run_in_executor(
                None, 
                lambda d, m, o: chunk_text(d, max_tokens=m, output_format=o),
                doc, self.max_tokens, "python"
            )
            return result if isinstance(result, list) else []
    
    async def process_batch(self, batch: PipelineBatch) -> PipelineBatch:
        """
        Process a batch of documents.
        
        Args:
            batch: The batch to process
            
        Returns:
            The processed batch with chunks
        """
        logger.info(f"Chunking batch {batch.batch_id} with {len(batch.docs)} documents")
        
        # Check if documents have content
        docs_to_chunk = []
        for doc in batch.docs:
            if "content" not in doc:
                batch.add_error(
                    self.name, 
                    ValueError("Document missing content"),
                    {"doc_id": doc.get("id")}
                )
            else:
                docs_to_chunk.append(doc)
        
        if not docs_to_chunk:
            logger.warning(f"No documents to chunk in batch {batch.batch_id}")
            return batch
        
        # Process each document in parallel
        tasks = []
        for doc in docs_to_chunk:
            tasks.append(self.chunk_document_async(doc))
        
        # Wait for all tasks to complete
        try:
            with torch.cuda.device(self.device):
                # Use the compute stream for chunking
                if self.compute_stream:
                    with torch.cuda.stream(self.compute_stream):
                        chunk_results = await asyncio.gather(*tasks, return_exceptions=True)
                else:
                    chunk_results = await asyncio.gather(*tasks, return_exceptions=True)
        except Exception as e:
            batch.add_error(self.name, e)
            return batch
        
        # Collect all chunks
        all_chunks: List[Dict[str, Any]] = []
        doc_chunk_map: Dict[str, List[str]] = {}  # Map document IDs to their chunks
        
        for i, result in enumerate(chunk_results):
            doc = docs_to_chunk[i]
            doc_id = doc.get("id", "")
            
            if isinstance(result, Exception):
                # Handle errors
                batch.add_error(self.name, result, {"doc_id": doc_id})
                continue
            
            # Add chunks to the list
            if isinstance(result, list):
                doc_chunks = result
                all_chunks.extend(doc_chunks)
                
                # Track which chunks belong to which document
                doc_chunk_map[doc_id] = [chunk.get("id", "") for chunk in doc_chunks if isinstance(chunk, dict)]
            
            # Log chunking results
            if isinstance(result, list):
                doc_chunks = result
                logger.debug(f"Document {doc_id} produced {len(doc_chunks)} chunks")
        
        # Update the batch with chunks
        batch.chunks = all_chunks
        batch.metadata["doc_chunk_map"] = doc_chunk_map
        batch.metadata["total_chunks"] = len(all_chunks)
        
        logger.info(f"Created {len(all_chunks)} chunks from {len(docs_to_chunk)} documents in batch {batch.batch_id}")
        return batch

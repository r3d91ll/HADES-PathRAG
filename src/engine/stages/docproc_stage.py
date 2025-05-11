"""
Document processing stage.

This module implements the document processing stage of the pipeline.
It uses the Docling adapter to process documents.
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Set

import torch

from ..batch_types import PipelineBatch
from .base import StageBase
from src.docproc.adapters.docling_adapter import DoclingAdapter
from src.docproc.core import process_document


logger = logging.getLogger(__name__)


class DocProcStage(StageBase):
    """
    Document processing stage.
    
    This stage processes documents using the Docling adapter.
    """
    
    name = "DocProc"
    
    def __init__(
        self, 
        device: Union[str, torch.device],
        timeout_seconds: int = 60,
        **kwargs: Any
    ) -> None:
        """
        Initialize the document processing stage.
        
        Args:
            device: The device to run this stage on
            timeout_seconds: Maximum time to wait for document processing
            **kwargs: Additional keyword arguments
        """
        super().__init__(device)
        self.timeout_seconds = timeout_seconds
        self.adapter = DoclingAdapter()
        self.supported_formats: Set[str] = {
            "pdf", "markdown", "text", "html", "python", 
            "json", "yaml", "toml", "xml", "csv"
        }
    
    async def process_document_async(self, doc_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Process a document asynchronously.
        
        Args:
            doc_path: Path to the document
            
        Returns:
            Processed document
        """
        # Use asyncio.to_thread to run the synchronous process_document function in a thread
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, process_document, str(doc_path))
        return result if isinstance(result, dict) else {}
    
    async def process_batch(self, batch: PipelineBatch) -> PipelineBatch:
        """
        Process a batch of documents.
        
        Args:
            batch: The batch to process
            
        Returns:
            The processed batch with documents
        """
        logger.info(f"Processing batch {batch.batch_id} with {len(batch.docs)} documents")
        
        # Check if documents are already processed
        if all("content" in doc for doc in batch.docs):
            logger.info(f"Batch {batch.batch_id} already processed, skipping")
            return batch
        
        # Process each document in parallel
        tasks = []
        for doc in batch.docs:
            # Skip documents that are already processed
            if "content" in doc:
                continue
                
            # Check if the document format is supported
            doc_path = doc.get("path")
            if not doc_path:
                batch.add_error(self.name, ValueError("Document missing path"), {"doc_id": doc.get("id")})
                continue
                
            # Get the file extension
            ext = os.path.splitext(doc_path)[1].lower().lstrip(".")
            if ext not in self.supported_formats:
                batch.add_error(
                    self.name, 
                    ValueError(f"Unsupported document format: {ext}"),
                    {"doc_id": doc.get("id"), "path": doc_path}
                )
                continue
                
            # Process the document
            tasks.append(self.process_document_async(doc_path))
        
        # Wait for all tasks to complete with timeout
        try:
            processed_docs = await asyncio.gather(*tasks, return_exceptions=True)
        except asyncio.TimeoutError:
            batch.add_error(self.name, asyncio.TimeoutError(f"Timed out after {self.timeout_seconds}s"))
            return batch
        
        # Update the batch with processed documents
        for i, result in enumerate(processed_docs):
            if isinstance(result, Exception):
                # Handle errors
                batch.add_error(self.name, result, {"doc_index": i})
            else:
                # Update the document
                if isinstance(result, dict):
                    batch.docs[i].update(result)
        
        logger.info(f"Processed {len(processed_docs)} documents in batch {batch.batch_id}")
        return batch

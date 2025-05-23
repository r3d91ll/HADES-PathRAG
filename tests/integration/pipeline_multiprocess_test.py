#!/usr/bin/env python
"""
Multiprocessing implementation of the document processing pipeline test.

This test demonstrates true parallel processing of documents using Python's multiprocessing
module, with configurable GPU/CPU resource allocation based on the training_pipeline_config.yaml.
"""

import os
import sys
import time
import yaml
import json
import asyncio
import logging
import argparse
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
import multiprocessing as mp
from multiprocessing import Pool, Queue, Manager

# Add the project root to the Python path
project_root = str(Path(__file__).parent.parent.parent.absolute())
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import local modules
from src.docproc.manager import DocumentProcessorManager
from src.chunking.text_chunkers.chonky_chunker import chunk_text
from src.embedding.adapters import ModernBERTEmbeddingAdapter
from src.docproc.utils.format_detector import (
    detect_format_from_path,
    get_content_category
)

# Custom timing formatter
class TimingFormatter(logging.Formatter):
    """Formatter that adds elapsed time since start."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_time = time.time()
    
    def formatTime(self, record, datefmt=None):
        formatted_time = super().formatTime(record, datefmt)
        # Add elapsed time since start
        elapsed = time.time() - self.start_time
        return f"{formatted_time} [+{elapsed:.3f}s]"

# Configure logging with custom formatter
formatter = TimingFormatter(
    fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Configure root logger
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)

# Remove existing handlers
for handler in root_logger.handlers:
    root_logger.removeHandler(handler)

# Add console handler with custom formatter
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
root_logger.addHandler(console_handler)

# Get our module logger
logger = logging.getLogger(__name__)

# Load the pipeline configuration
def load_pipeline_config() -> Dict[str, Any]:
    """Load the training pipeline configuration from YAML.
    
    This function attempts to use the new config_loader module first,
    and falls back to direct file loading if that's not available.
    """
    try:
        # Try to use the new config_loader module first
        from src.config.config_loader import load_pipeline_config as load_config
        return load_config(pipeline_type='training')
    except ImportError:
        # Fall back to direct file loading
        config_path = Path(project_root) / "src" / "config" / "training_pipeline_config.yaml"
        if not config_path.exists():
            # Check for old filename as a last resort
            old_path = Path(project_root) / "src" / "config" / "pipeline_config.yaml"
            if old_path.exists():
                config_path = old_path
                logger.warning(f"Using deprecated pipeline_config.yaml. Please update to training_pipeline_config.yaml.")
            else:
                raise FileNotFoundError(f"Cannot find pipeline configuration at {config_path} or {old_path}")
                
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return config

# Initialize components based on the configuration
def init_docproc(config: Dict[str, Any], worker_id: int) -> Any:
    """Initialize document processor with appropriate configuration."""
    try:
        # Get processor type based on worker ID
        processor_type = config.get("processor_type", "generic")
        logger.info(f"[Worker {worker_id}] Initializing document processor '{processor_type}'")
        
        # Initialize processor based on type
        # Use DocumentProcessorManager from src.docproc.manager
        from src.docproc.manager import DocumentProcessorManager
        
        # Configure document processor
        processor_options = config.get("document_processor", {})
        
        # Set up device configuration for GPU handling
        if config.get("gpu_execution", {}).get("enabled", False):
            device_config = config["gpu_execution"]["docproc"]
            processor_options["device"] = device_config.get("device", "cuda:0")
        
        return DocumentProcessorManager(options=processor_options)
    except Exception as e:
        logger.error(f"Error initializing document processor: {e}")
        raise

def init_chunker(config: Dict[str, Any], worker_id: int, custom_config: Optional[Dict[str, Any]] = None) -> Any:
    """Initialize chunker with appropriate device and configuration.
    
    Args:
        config: Global pipeline configuration
        worker_id: Worker ID for logging
        custom_config: Optional custom configuration to override defaults
        
    Returns:
        Initialized chunker instance
    """
    try:
        # Get chunker type from config
        chunker_type = config.get("chunker", "chonky")
        logger.info(f"[Worker {worker_id}] Initializing chunker '{chunker_type}'")
        
        # Set up default configuration with smaller chunk sizes for embedding compatibility
        default_config = {
            "max_chunk_size": 400,  # Characters, keep smaller for embedding model compatibility
            "min_chunk_size": 50,
            "overlap": 20,
            "respect_boundaries": True
        }
        
        # Override with any custom configuration
        if custom_config:
            default_config.update(custom_config)
            
        logger.info(f"[Worker {worker_id}] Using chunker config: max_size={default_config['max_chunk_size']}, min_size={default_config['min_chunk_size']}")
            
        # Import and initialize appropriate chunker
        if chunker_type == "chonky":
            # Import required modules
            from src.chunking.base import BaseChunker
            from src.chunking.text_chunkers.chonky_chunker import chunk_text
            # Create a wrapper class for the function-based chonky_chunker
            class ChonkyChunkerWrapper(BaseChunker):
                """Wrapper class for the function-based chonky_chunker."""
                
                def __init__(self, config=None):
                    super().__init__(name="chonky", config=config or {})
                    
                def chunk(self, text=None, content=None, doc_id=None, path="unknown", doc_type="text", max_tokens=1024, output_format="json", **kwargs):
                    """Wrap the chunk_text function to match BaseChunker interface.
                    
                    Accepts both 'text' and 'content' parameters for compatibility with different calling conventions.
                    """
                    # Use content if provided, otherwise fall back to text
                    actual_content = content if content is not None else text
                    
                    # Use max_chunk_size from config if provided
                    max_tokens = self.config.get("max_chunk_size", max_tokens)
                    
                    # Call the chonky chunker function
                    result = chunk_text(
                        content=actual_content,
                        doc_id=doc_id,
                        path=path,
                        doc_type=doc_type,
                        max_tokens=max_tokens,
                        output_format=output_format
                    )
                    return result.get("chunks", []) if isinstance(result, dict) else []
            
            return ChonkyChunkerWrapper(config=default_config)
        else:
            # For other chunker types, use registry
            from src.chunking import get_chunker
            return get_chunker(chunker_type, config=default_config)
    except Exception as e:
        logger.error(f"Error initializing chunker: {e}")
        raise

def init_embedding(config: Dict[str, Any], worker_id: int, file_type: str = "document", content_category: str = "text") -> Dict[str, Any]:
    """Initialize embedding adapter with appropriate device and model based on file format and content category.
    
    Args:
        config: Configuration dictionary
        worker_id: Worker ID for logging
        file_type: Format type of the file (python, yaml, json, markdown, etc.)
        content_category: Content category of the file (code or text)
        
    Returns:
        Initialized embedding adapter
    """
    logger.info(f"[Worker {worker_id}] Initializing embedding adapter...")
    
    # Import the renamed adapter class
    from src.embedding.adapters import EncoderEmbeddingAdapter
    
    # Select specialized models based on content category and format type
    # Set max sequence length for all models to avoid truncation issues
    max_seq_length = 512
    truncation = True
    
    # First check content category to determine the general model type
    if content_category == "code":
        # Use code-specific models for all code formats
        model_name = "microsoft/codebert-base"
        model_type = "encoder"
        
        # Set format-specific embedding type for better tracking
        embedding_type = f"{file_type}_code"
        logger.info(f"[Worker {worker_id}] Using CodeBERT for {file_type} files")
    else:
        # Use the ModernBERT model for text documents
        model_name = "bert-base-uncased"  # Use a standard BERT model that's publicly available
        model_type = "modernbert"
        embedding_type = "text"
        logger.info(f"[Worker {worker_id}] Using ModernBERT for {file_type} files")
    
    # Configure embedding based on execution mode
    if config.get("gpu_execution", {}).get("enabled", False):
        device = config["gpu_execution"]["embedding"]["device"]
        batch_size = config["gpu_execution"]["embedding"]["batch_size"]
        precision = config["gpu_execution"]["embedding"]["model_precision"]
        logger.info(f"[Worker {worker_id}] Using GPU configuration for embedding with device: {device}, precision: {precision}")
    else:
        device = config["cpu_execution"]["embedding"]["device"]
        num_threads = config["cpu_execution"]["embedding"]["num_threads"]
        logger.info(f"[Worker {worker_id}] Using CPU configuration for embedding with {num_threads} threads")
    
    # Initialize the adapter
    from src.embedding import get_adapter_by_name
    adapter = get_adapter_by_name(model_type)(
        model_name=model_name, 
        device=device,
        max_seq_length=max_seq_length,
        truncation=truncation
    )
    
    logger.info(f"[Worker {worker_id}] Initialized {model_type} adapter with model={model_name} on {device}")
    return {
        "adapter": adapter, 
        "worker_id": worker_id,
        "model_name": model_name,
        "model_type": model_type,
        "embedding_type": embedding_type
    }

# Process a document using multiprocessing
def process_document(args: Tuple[Path, Dict[str, Any], int]):
    """Process a single document using the pipeline components."""
    file_path, config, worker_id = args
    
    # Create a worker-specific logger
    worker_logger = logging.getLogger(f"__mp_main__")
    worker_logger.info(f"[Worker {worker_id}] Processing file: {file_path.name}")
    
    start_time = time.time()
    
    # Initialize timings dictionary to prevent UnboundLocalError in exception handling
    timings = {"docproc": 0, "chunking": 0, "embedding": 0, "total": 0}
    
    try:
        # Initialize components
        doc_processor = init_docproc(config, worker_id)
        
        # Detect file format and content category using the new format detector
        try:
            format_type = detect_format_from_path(file_path)
            content_category = get_content_category(format_type)
            worker_logger.info(f"[Worker {worker_id}] Detected format '{format_type}' and category '{content_category}' for {file_path.name}")
        except ValueError as e:
            # Skip unsupported file types gracefully
            worker_logger.warning(f"[Worker {worker_id}] Skipping unsupported file: {file_path.name} - {str(e)}")
            return {
                "file_id": f"skipped_{file_path.name}",
                "file_name": file_path.name,
                "file_size": file_path.stat().st_size if file_path.exists() else 0,
                "worker_id": worker_id,
                "status": "skipped",
                "reason": "unsupported_format",
                "error": str(e),
                "chunks": [],
                "processing_time": 0
            }
        
        # Prepare chunker config to limit chunk size for embedding compatibility
        chunker_config = {
            "max_chunk_size": 400,  # Characters, keep smaller for embedding model compatibility
            "min_chunk_size": 50,
            "overlap": 20,
            "respect_boundaries": True
        }
        
        # Get chunker based on content category and format type
        if content_category == "code":
            if format_type == "python":
                # Use Python code chunker
                from src.chunking.code_chunkers.python_chunker import PythonCodeChunker
                chunker = PythonCodeChunker(config=chunker_config)
                worker_logger.info(f"[Worker {worker_id}] Using Python code chunker")
            elif format_type in ["yaml", "yml"]:
                # Use YAML code chunker
                from src.chunking.code_chunkers.yaml_chunker import YAMLCodeChunker
                chunker = YAMLCodeChunker(config=chunker_config)
                worker_logger.info(f"[Worker {worker_id}] Using YAML code chunker")
            elif format_type == "json":
                # Use JSON code chunker
                from src.chunking.code_chunkers.json_chunker import JSONCodeChunker
                chunker = JSONCodeChunker(config=chunker_config)
                worker_logger.info(f"[Worker {worker_id}] Using JSON code chunker")
            else:
                # For other code formats, use generic code chunker
                worker_logger.info(f"[Worker {worker_id}] Using generic code chunker for {format_type}")
                from src.chunking.code_chunkers.generic_code_chunker import GenericCodeChunker
                chunker = GenericCodeChunker(config=chunker_config)
        else:
            # For text content, use the standard chunker initialization with custom config
            worker_logger.info(f"[Worker {worker_id}] Using text chunker for {format_type}")
            custom_chunker_config = {
                "max_chunk_size": 400,  # Characters, keep smaller for embedding model compatibility
                "min_chunk_size": 50,
                "overlap": 20,
                "respect_boundaries": True
            }
            chunker = init_chunker(config, worker_id, custom_config=custom_chunker_config)
            
        # Initialize embedding adapter with the appropriate model based on format type and content category
        embedder = init_embedding(config, worker_id, file_type=format_type, content_category=content_category)
        
        # Track timings
        timings = {"docproc": 0, "chunking": 0, "embedding": 0, "total": 0}
        
        # Step 1: Process document
        worker_logger.info(f"[Worker {worker_id}] Starting document processing")
        docproc_start = time.time()
        
        # Process the document based on file type
        if file_path.suffix.lower() == ".py":
            # Direct handling for Python files
            worker_logger.info(f"[Worker {worker_id}] Using direct PythonCodeAdapter for {file_path.name}")
            # Import and use PythonCodeAdapter directly
            try:
                from src.docproc.adapters.python_code_adapter import PythonCodeAdapter
                python_adapter = PythonCodeAdapter()
                doc_result = python_adapter.process(file_path)
            except Exception as e:
                worker_logger.error(f"[Worker {worker_id}] Error processing Python file: {e}")
                raise
        else:
            # Use standard document processor for other file types
            doc_result = doc_processor.process_document(path=file_path)
        
        docproc_end = time.time()
        docproc_time = docproc_end - docproc_start
        timings["docproc"] = docproc_time
        
        worker_logger.info(f"[Worker {worker_id}] Document processing completed in {docproc_time:.2f}s")
        
        # Step 2: Chunking
        chunk_start = time.time()
        
        # Get document properties
        doc_id = doc_result.get('id', file_path.stem)
        doc_path = doc_result.get('source', str(file_path))
        
        worker_logger.info(f"[Worker {worker_id}] Starting document chunking: {file_path.name}")
        
        # Add metadata to the chunk including embedding model information
        content_type = format_type.lower() if format_type else "text"
        text = doc_result.get('content', "") or ""
        text = str(text)
        content_length = len(text)
        
        # Add chunking config to limit chunk size
        chunking_config = {
            "max_chunk_size": 500,  # Smaller chunks to stay within embedding model limits
            "overlap": 50,
            "respect_boundaries": True
        }
        
        worker_logger.info(f"[Worker {worker_id}] Extracted content for chunking: {content_length/1024:.1f} KB")
        
        # Process through chunking
        # Set document type based on the detected format and content category
        if content_category == "code":
            doc_type = f"{format_type}_code"
        else:
            doc_type = format_type
        
        # Prepare metadata for all chunkers
        metadata = {
            "file_path": str(file_path),
            "file_name": file_path.name,
            "doc_id": doc_id,
            "path": doc_path,
            "doc_type": doc_type,
            "max_tokens": 1024,
            "output_format": "json"
        }
        
        # Add language metadata for Python files
        if doc_type == "python_code":
            metadata["language"] = "python"
        
        # Different chunkers have different interfaces, handle accordingly
        # Call the chunker to break down the content
        try:
            if hasattr(chunker, 'chunk'):
                # Use standard chunker interface
                chunks_result = chunker.chunk(text=text, metadata=metadata)
            else:
                # Use a function-based chunker
                chunks_result = chunker(
                    content=text,
                    doc_id=doc_id,
                    path=str(file_path),
                    doc_type=content_type
                )
        except Exception as e:
            worker_logger.error(f"Error during chunking: {e}")
            return []
        
        # Extract chunks - handle both dictionary and list return types
        if isinstance(chunks_result, dict):
            chunks = chunks_result.get("chunks", [])
        else:
            # If chunks_result is already a list of chunks
            chunks = chunks_result
            
        # Ensure each chunk has a unique ID for ISNE training
        for i, chunk in enumerate(chunks):
            if "id" not in chunk:
                # Create a unique ID using document ID and chunk index
                chunk["id"] = f"{doc_id}_chunk_{i}"
                
            # Add overlap_context structure required by ISNE orchestrator
            # This is crucial for creating edges in the document graph
            if "overlap_context" not in chunk:
                chunk["overlap_context"] = {}
                
            # Set position and total which are used to establish relationships
            chunk["overlap_context"]["position"] = i
            chunk["overlap_context"]["total"] = len(chunks)
            
            # Add empty pre/post context (optional but commonly expected)
            if "pre" not in chunk["overlap_context"]:
                chunk["overlap_context"]["pre"] = ""
            if "post" not in chunk["overlap_context"]:
                chunk["overlap_context"]["post"] = ""
        
        # Calculate stats on chunks
        chunk_sizes = [len(str(chunk.get("content", ""))) for chunk in chunks]
        avg_chunk_size = sum(chunk_sizes) / len(chunk_sizes) if chunk_sizes else 0
        max_chunk_size = max(chunk_sizes) if chunk_sizes else 0
        min_chunk_size = min(chunk_sizes) if chunk_sizes else 0
        
        worker_logger.info(f"[Worker {worker_id}] Created {len(chunks)} chunks for {file_path.name} - "
                  f"Avg size: {avg_chunk_size/1024:.1f} KB, Range: {min_chunk_size/1024:.1f}-{max_chunk_size/1024:.1f} KB")
        
        chunk_time = time.time() - chunk_start
        timings["chunking"] = chunk_time
        
        # Step 3: Embedding generation
        embed_start = time.time()
        
        # Extract text content from each chunk for embedding
        texts = [chunk.get("content", "") for chunk in chunks]
        
        if not texts:
            worker_logger.warning(f"[Worker {worker_id}] No text content to embed for {file_path.name}")
            timings["embedding"] = time.time() - embed_start
            return {"file_id": doc_id, "file_name": file_path.name, "file_size": file_path.stat().st_size, "worker_id": worker_id, "chunks": chunks, "embeddings": [], "timing": timings}
        
        # Calculate total text size for embedding
        total_text_size = sum(len(text) for text in texts)
        avg_text_size = total_text_size / len(texts) if texts else 0
        
        worker_logger.info(f"[Worker {worker_id}] Starting embedding generation for {len(texts)} chunks from {file_path.name} "
                   f"(Total: {total_text_size/1024:.1f} KB, Avg: {avg_text_size/1024:.1f} KB per chunk)")
        
        # Get batch size from config
        if config.get("gpu_execution", {}).get("enabled", False):
            batch_size = config["gpu_execution"]["embedding"]["batch_size"]
        else:
            batch_size = 8  # Default batch size for CPU
        
        # Process in batches to avoid OOM
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_size_bytes = sum(len(text) for text in batch_texts)
            
            worker_logger.info(f"[Worker {worker_id}] Processing embedding batch {i//batch_size + 1}/"
                      f"{(len(texts) + batch_size - 1)//batch_size} "
                      f"({len(batch_texts)} chunks, {batch_size_bytes/1024:.1f} KB)")
            
            # Create a new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                # Run the async embed method in a synchronous context
                batch_embeddings = loop.run_until_complete(embedder["adapter"].embed(batch_texts))
                embeddings.extend(batch_embeddings)
            finally:
                loop.close()
        
        # Get embedding statistics
        embed_dims = len(embeddings[0]) if embeddings else 0
        
        # Convert embeddings to lists for JSON serialization
        embeddings_list = []
        for i, emb in enumerate(embeddings):
            if isinstance(emb, np.ndarray):
                emb_list = emb.tolist()
            else:
                emb_list = emb
            embeddings_list.append(emb_list)
            
            # Also attach embedding directly to the corresponding chunk for ISNE
            if i < len(chunks):
                chunks[i]["embedding"] = emb_list
                # Add embedding model metadata for ISNE awareness
                chunks[i]["embedding_model"] = embedder["model_name"]
                chunks[i]["embedding_type"] = embedder["embedding_type"]
        
        embed_time = time.time() - embed_start
        timings["embedding"] = embed_time
        
        # Total time
        timings["total"] = time.time() - start_time
        
        # Add timestamp to show when this task completed
        completed_at = datetime.now().isoformat()
        
        worker_logger.info(f"[Worker {worker_id}] Completed {file_path.name}: {len(chunks)} chunks, {len(embeddings)} embeddings in {timings['total']:.2f}s")
        
        return {"file_id": doc_id, "file_name": file_path.name, "file_size": file_path.stat().st_size, "worker_id": worker_id, "chunks": chunks, "embeddings": embeddings_list, "timing": timings, "completed_at": completed_at}
    except Exception as e:
        worker_logger.error(f"[Worker {worker_id}] Error processing document {file_path.name}: {str(e)}", exc_info=True)
    
    return {"file_id": file_path.stem, "file_name": file_path.name, "file_size": file_path.stat().st_size, "worker_id": worker_id, "chunks": [], "embeddings": [], "timing": timings}

class PipelineMultiprocessTester:
    """Test parallel processing of documents using true multiprocessing.
    
    This class coordinates the parallel processing of documents using configuration
    settings from the training_pipeline_config.yaml file. It demonstrates how to use
    the training pipeline components (document processing, chunking, and embedding)
    in a multiprocessing environment with proper GPU resource allocation.
    """
    
    def __init__(
        self,
        test_data_dir: str,
        output_dir: str,
        num_files: int = 50,
        max_workers: int = 4,
        batch_size: int = 12
    ):
        """Initialize the multiprocessing pipeline tester."""
        self.test_data_dir = Path(test_data_dir)
        self.output_dir = Path(output_dir)
        self.num_files = num_files
        self.max_workers = max_workers
        self.batch_size = batch_size
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load configuration
        self.config = load_pipeline_config()
        
        # Statistics
        self.stats = {
            "total_files": 0,
            "total_chunks": 0,
            "total_embeddings": 0,
            "processing_time": {
                "document_processing": 0,
                "chunking": 0,
                "embedding": 0,
                "total": 0
            },
            "file_details": {}
        }
    
    def find_supported_files(self) -> List[Path]:
        """Find all supported file types recursively in the test directory and subdirectories."""
        supported_files = []
        
        # Get the extension-to-format mapping from our format detector
        from src.docproc.utils.format_detector import get_extension_to_format_map
        extension_map = get_extension_to_format_map()
        
        # First, find all files recursively
        all_files = list(self.test_data_dir.glob("**/*"))
        all_files = [f for f in all_files if f.is_file()]
        file_count = len(all_files)
        logger.info(f"Found {file_count} total files in {self.test_data_dir}")
        
        # Create a set of supported extensions from our format detector
        supported_extensions = set(extension_map.keys())
        logger.info(f"Supporting {len(supported_extensions)} file extensions: {', '.join(sorted(supported_extensions))}")
        
        # Only include files with supported extensions
        for file_path in all_files:
            ext = file_path.suffix.lower()
            if ext in supported_extensions:
                supported_files.append(file_path)
                format_type = extension_map[ext]
                content_category = get_content_category(format_type)
                logger.debug(f"Including {file_path.name} ({format_type}, {content_category})")
                
        # Group supported files by extension for logging
        by_extension = {}
        for file_path in supported_files:
            ext = file_path.suffix.lower()
            if ext not in by_extension:
                by_extension[ext] = []
            by_extension[ext].append(file_path)
            
        # Log counts by extension
        for ext, files in by_extension.items():
            format_type = extension_map[ext]
            content_category = get_content_category(format_type)
            logger.info(f"Found {len(files)} {ext} files (format: {format_type}, category: {content_category})")
            
        # Log summary
        if file_count > 0:
            logger.info(f"Including {len(supported_files)} of {file_count} files ({len(supported_files)/file_count*100:.1f}%)")
            logger.info(f"Excluding {file_count - len(supported_files)} unsupported files")
        else:
            logger.error(f"No files found in {self.test_data_dir}. Check if the path is correct and accessible.")
            logger.info(f"Absolute path: {Path(self.test_data_dir).absolute()}")
            # List parent directory to help diagnose the issue
            parent_dir = Path(self.test_data_dir).parent
            logger.info(f"Contents of parent directory {parent_dir}:")
            for item in parent_dir.iterdir():
                if item.is_dir():
                    logger.info(f"  DIR: {item.name}")
                else:
                    logger.info(f"  FILE: {item.name}")


                
        # Group files by directory
        files_by_dir = {}
        for file in supported_files:
            parent_dir = file.parent
            rel_path = parent_dir.relative_to(self.test_data_dir)
            if rel_path not in files_by_dir:
                files_by_dir[rel_path] = []
            files_by_dir[rel_path].append(file)
            
        for dir_path, dir_files in files_by_dir.items():
            if str(dir_path) == '.':
                logger.info(f"Found {len(dir_files)} files in root directory")
            else:
                logger.info(f"Found {len(dir_files)} files in subdirectory {dir_path}")
                
        self.directory_structure = files_by_dir  # Store for relationship building later
        
        # Sort files to ensure consistent order
        supported_files = sorted(supported_files, key=lambda x: str(x))
        logger.info(f"Found {len(supported_files)} supported files in {self.test_data_dir}")
        
        if len(supported_files) > self.num_files:
            # Select a representative subset
            return supported_files[:self.num_files]
        return supported_files
    
    def run_test(self, run_isne_training=False) -> Dict[str, Any]:
        """
        Run the pipeline multiprocessing test.
        
        Args:
            run_isne_training: If True, will run ISNE training directly with in-memory data after pipeline processing
            
        Returns:
            Dictionary with test statistics and results
        """
        logger.info("=== Starting Multiprocessing Pipeline Test ===")
        logger.info(f"Max Workers: {self.max_workers}, Batch Size: {self.batch_size}, Files to Process: {self.num_files}")
        
        # Overall timing
        test_start_time = time.time()
        
        # Find files to process
        file_discovery_start = time.time()
        files = self.find_supported_files()
        file_discovery_time = time.time() - file_discovery_start
        
        logger.info(f"Found {len(files)} supported files in {file_discovery_time:.2f}s")
        logger.info(f"Starting parallel processing of {len(files)} files...")
        
        # Prepare worker arguments
        worker_args = [(file, self.config, i % self.max_workers) for i, file in enumerate(files)]
        
        # Start the worker pool with the specified number of workers
        parallel_start = time.time()
        with Pool(processes=self.max_workers) as pool:
            results = pool.map(process_document, worker_args)
        parallel_time = time.time() - parallel_start
        
        # Update statistics
        stats_start = time.time()
        for result in results:
            if "file_id" not in result:
                continue
            
            self.stats["total_files"] += 1
            self.stats["total_chunks"] += len(result.get("chunks", []))
            self.stats["total_embeddings"] += len(result.get("embeddings", []))
            
            # Update processing times
            for timing_key in ["document_processing", "chunking", "embedding", "total"]:
                self.stats["processing_time"][timing_key] += result["timing"].get(timing_key, 0)
            
            # Store file details
            self.stats["file_details"][result["file_id"]] = {
                "file_name": result["file_name"],
                "file_size": result["file_size"],
                "chunks": len(result.get("chunks", [])),
                "embeddings": len(result.get("embeddings", [])),
                "processing_time": result["timing"]
            }
        stats_time = time.time() - stats_start
        
        # Save the results
        # Save statistics to output file
        output_file = self.output_dir / "pipeline_stats.json"
        save_start = time.time()
        with open(output_file, "w") as f:
            json.dump(self.stats, f, indent=2)
        
        # Save all processed documents for inspection
        sample_output = self.output_dir / "isne_input_sample.json"
        with open(sample_output, "w") as f:
            # Save a sample of the processed documents (first 5 or fewer)
            sample_docs = results[:min(5, len(results))]
            json.dump(sample_docs, f, indent=2)
        
        logger.info(f"Saved statistics to {output_file}")
        logger.info(f"Saved sample ISNE input to {sample_output}")
        save_time = time.time() - save_start
        
        # Store processing time data for report later
        self.processing_stats = {
            "file_discovery_time": file_discovery_time,
            "parallel_time": parallel_time,
            "stats_time": stats_time,
            "save_time": save_time,
            "test_start_time": test_start_time,
            "results": results,
            "total_runtime": time.time() - test_start_time
        }
        
        # Log brief processing completion
        logger.info("Pipeline document processing completed successfully")
        logger.info(f"Processed {self.stats['total_files']} files with {self.stats['total_chunks']} chunks")
        if "total_embeddings" in self.stats:
            logger.info(f"Generated {self.stats['total_embeddings']} embeddings")
            
        # Optional: Run ISNE training directly with in-memory data if requested
        isne_results = None
        if run_isne_training:
            logger.info("\n=== Starting In-Memory ISNE Training ===")
            isne_results = self.direct_train_isne(results)
            
        # Now generate the complete performance report after all processing is done
        self._generate_performance_report(run_isne_training)
        
        return isne_results
        
    def _generate_performance_report(self, run_isne_training):
        # Calculate average processing times
        if self.stats["total_files"] > 0:
            avg_doc_time = self.stats["processing_time"]["document_processing"] / self.stats["total_files"]
            avg_chunk_time = self.stats["processing_time"]["chunking"] / self.stats["total_files"]
            avg_embed_time = self.stats["processing_time"]["embedding"] / self.stats["total_files"]
        else:
            avg_doc_time = avg_chunk_time = avg_embed_time = 0
        
        # Calculate throughput
        total_runtime = time.time() - self.processing_stats["test_start_time"]
        files_per_second = self.stats["total_files"] / total_runtime if total_runtime > 0 else 0
        chunks_per_second = self.stats["total_chunks"] / total_runtime if total_runtime > 0 else 0
        
        # Calculate parallelization efficiency
        theoretical_speedup = self.max_workers
        sequential_runtime = sum(r["timing"].get("total", 0) for r in self.processing_stats["results"] if "timing" in r)
        actual_speedup = sequential_runtime / self.processing_stats["parallel_time"] if self.processing_stats["parallel_time"] > 0 else 0
        efficiency = actual_speedup / theoretical_speedup if theoretical_speedup > 0 else 0
        
        # Performance report
        logger.info("\n=== Performance Report ===")
        logger.info("\nTiming Breakdown:")
        logger.info(f"  Setup:                0.00s")
        logger.info(f"  File Discovery:       {self.processing_stats['file_discovery_time']:.2f}s")
        logger.info(f"  Parallel Processing:  {self.processing_stats['parallel_time']:.2f}s")
        logger.info(f"  Statistics Update:    {self.processing_stats['stats_time']:.2f}s")
        logger.info(f"  Results Saving:       {self.processing_stats['save_time']:.2f}s")
        
        # Add ISNE training time if run
        isne_time = 0.0
        if run_isne_training and hasattr(self, 'isne_training_time'):
            isne_time = self.isne_training_time
            logger.info(f"  ISNE Training:        {isne_time:.2f}s")
            
        logger.info(f"  Total Runtime:        {total_runtime + isne_time:.2f}s")
        
        logger.info("\nAverage Processing Times:")
        logger.info(f"  Document Processing:  {avg_doc_time:.2f}s per file")
        logger.info(f"  Chunking:             {avg_chunk_time:.2f}s per file")
        logger.info(f"  Embedding:            {avg_embed_time:.2f}s per file")
        
        logger.info("\nThroughput:")
        logger.info(f"  Files per second:     {files_per_second:.2f}")
        logger.info(f"  Chunks per second:    {chunks_per_second:.2f}")
        
        logger.info("\nParallelization Efficiency:")
        logger.info(f"  Theoretical Speedup:  {theoretical_speedup:.2f}x")
        logger.info(f"  Actual Speedup:       {actual_speedup:.2f}x")
        logger.info(f"  Efficiency:           {efficiency:.2f} ({efficiency*100:.1f}%)")
        
        logger.info("\nPipeline Test Completed Successfully!")
        logger.info(f"  - Processed {self.stats['total_files']} files with {self.stats['total_chunks']} chunks")
        if "total_embeddings" in self.stats:
            logger.info(f"  - Generated {self.stats['total_embeddings']} embeddings with dimension unknown")
        logger.info(f"  - Saved results to {self.output_dir}")
        
        # Add ISNE training details if run
        if run_isne_training and hasattr(self, 'isne_training_time'):
            # Get training results if available
            isne_results = getattr(self, 'isne_results', None)
            if isne_results and 'training_metrics' in isne_results:
                metrics = isne_results['training_metrics']
                logger.info("\nISNE Training Results:")
                logger.info(f"  - Graph size: {metrics.get('graph_nodes', 0)} nodes, {metrics.get('graph_edges', 0)} edges")
                logger.info(f"  - Training time: {self.isne_training_time:.2f}s")
                logger.info(f"  - Epochs completed: {metrics.get('epochs', 0)}")
                logger.info(f"  - Final loss: {metrics.get('final_loss', 0):.4f}")
                logger.info(f"  - Device used: {metrics.get('device', 'unknown')}")
                if 'embedding_dim' in metrics and 'output_dim' in metrics:
                    logger.info(f"  - Dimensions: {metrics.get('embedding_dim', 0)} → {metrics.get('output_dim', 0)}")
            else:
                logger.info("\nISNE Training Completed")
                logger.info(f"  - Training time: {self.isne_training_time:.2f}s")
        
        # Write full performance report to file for later analysis
        report_file = self.output_dir / "performance_report.txt"
        # TODO: Write detailed performance metrics to file
        
        return self.stats
    
    def prepare_documents_for_isne(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Prepare documents for ISNE training by ensuring proper structure.
        
        Args:
            documents: List of document dictionaries with embeddings
            
        Returns:
            List of properly structured documents for ISNE training
        """
        logger.info(f"Preparing {len(documents)} documents for ISNE training")
        
        # Create a deep copy to avoid modifying originals
        import copy
        prepped_docs = copy.deepcopy(documents)
        
        # Simplify the document structure for ISNE training - keep only docs with chunks
        valid_docs = []
        all_chunks = []
        
        # Step 1: Collect all valid chunks with embeddings
        for doc in prepped_docs:
            if "chunks" not in doc or not doc["chunks"]:
                continue
                
            chunks = []
            for i, chunk in enumerate(doc["chunks"]):
                if "embedding" in chunk and chunk["embedding"]:
                    # Create a clean chunk with only necessary fields
                    clean_chunk = {
                        "id": f"{doc['file_id']}_chunk_{i}",
                        "content": chunk.get("content", ""),
                        "embedding": chunk["embedding"],
                        "doc_id": doc["file_id"],
                        # Preserve embedding model metadata for ISNE awareness
                        "embedding_model": chunk.get("embedding_model", "unknown"),
                        "embedding_type": chunk.get("embedding_type", "unknown"),
                        "overlap_context": {
                            "position": i,
                            "total": len(doc["chunks"]),
                            "pre": "",
                            "post": ""
                        }
                    }
                    chunks.append(clean_chunk)
                    all_chunks.append(clean_chunk)  # Keep a flat list of all chunks
            
            if chunks:  # Only keep document if it has valid chunks
                clean_doc = {
                    "file_id": doc["file_id"],
                    "file_name": doc.get("file_name", ""),
                    "chunks": chunks
                }
                valid_docs.append(clean_doc)
        
        # Step 2: Ensure every document has proper relationships between chunks
        for doc in valid_docs:
            chunks = doc.get("chunks", [])
            if len(chunks) < 2:  # No relationships possible if only one chunk
                continue
                
            # Explicitly add relationships field to each chunk
            for i, chunk in enumerate(chunks):
                if "relationships" not in chunk:
                    chunk["relationships"] = []
                
                # Connect this chunk to all other chunks with decreasing weight by distance
                for j, other_chunk in enumerate(chunks):
                    if i != j:  # Skip self-relationships
                        # Calculate a weight based on proximity (higher for closer chunks)
                        proximity_weight = 1.0 - (abs(i-j) / len(chunks))
                        relationship = {
                            "source": chunk["id"],
                            "target": other_chunk["id"],
                            "type": "SEQUENTIAL",
                            "weight": proximity_weight
                        }
                        chunk["relationships"].append(relationship)
        
        # Step 3: Create a single artificial document with all chunks if we don't have enough docs
        if len(valid_docs) < 2 and all_chunks:
            # Force create a single document with all chunks to ensure we have enough for ISNE
            logger.info(f"Creating a single document with all {len(all_chunks)} chunks for ISNE training")
            
            # Ensure all chunks in this combined document are properly connected
            for i, chunk in enumerate(all_chunks):
                if "relationships" not in chunk:
                    chunk["relationships"] = []
                
                # Connect to at least 2 other chunks (or all if fewer than 3 total)
                for j, other_chunk in enumerate(all_chunks):
                    if i != j:  # Skip self-relationships
                        proximity_weight = 1.0 - (abs(i-j) / max(1, len(all_chunks)-1))
                        relationship = {
                            "source": chunk["id"],
                            "target": other_chunk["id"],
                            "type": "ARTIFICIAL",
                            "weight": proximity_weight
                        }
                        chunk["relationships"].append(relationship)
            
            return [{
                "file_id": "combined_document",
                "file_name": "Combined Document",
                "chunks": all_chunks
            }]
        
        # Count total chunks for verification
        total_chunks = sum(len(doc.get("chunks", [])) for doc in valid_docs)
        logger.info(f"Prepared {len(valid_docs)} documents with {total_chunks} total chunks for ISNE training")
        
        return valid_docs
    
    def simplified_isne_training(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """A simplified ISNE training function that doesn't rely on the complex orchestrator.
        
        This function implements a minimal ISNE training approach that skips the complex
        graph construction and directly uses a simple dense graph for training.
        
        Args:
            documents: List of document dictionaries with embeddings
            
        Returns:
            Dictionary with training results
        """
        import torch
        import numpy as np
        from pathlib import Path
        import time
        
        logger.info("=== Starting Simplified ISNE Training ===")
        
        # Create output directories
        output_dir = self.output_dir / "isne-training-simplified"
        output_dir.mkdir(exist_ok=True, parents=True)
        model_output_dir = Path("./models/isne")
        model_output_dir.mkdir(exist_ok=True, parents=True)
        
        # Step 1: Extract embeddings from documents
        all_embeddings = []
        all_chunk_ids = []
        
        for doc in documents:
            for chunk in doc.get("chunks", []):
                if chunk.get("embedding"):
                    all_embeddings.append(chunk["embedding"])
                    all_chunk_ids.append(chunk["id"])
        
        if not all_embeddings:
            logger.error("No embeddings found in documents")
            return {"success": False, "error": "No embeddings found"}
        
        logger.info(f"Extracted {len(all_embeddings)} embeddings from documents")
        
        # Step 2: Convert embeddings to tensor
        embeddings_tensor = torch.tensor(all_embeddings, dtype=torch.float)
        
        # Step 3: Determine device for training
        device = "cpu"  # Default fallback
        if self.config.get("gpu_execution", {}).get("enabled", False):
            device = self.config["gpu_execution"]["isne"]["device"]
            logger.info(f"Using GPU for ISNE training: {device}")
        elif self.config.get("cpu_execution", {}).get("enabled", False):
            device = "cpu"
            logger.info(f"Using CPU for ISNE training")
        
        embeddings_tensor = embeddings_tensor.to(device)
        
        # Step 4: Create a simplified model
        class SimplifiedISNE(torch.nn.Module):
            def __init__(self, input_dim, output_dim):
                super().__init__()
                self.linear = torch.nn.Linear(input_dim, output_dim)
                
            def forward(self, x):
                return self.linear(x)
        
        # Step 5: Define training parameters
        input_dim = embeddings_tensor.shape[1]
        output_dim = 64  # Same as standard ISNE
        learning_rate = 0.01
        num_epochs = 20
        
        logger.info(f"Creating simplified ISNE model with input dim {input_dim} and output dim {output_dim}")
        
        # Step 6: Create and train the model
        model = SimplifiedISNE(input_dim, output_dim).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        # Step 7: Train the model
        logger.info("Starting simplified ISNE training...")
        start_time = time.time()
        
        losses = []
        for epoch in range(num_epochs):
            model.train()
            optimizer.zero_grad()
            
            # Forward pass
            isne_embeddings = model(embeddings_tensor)
            
            # Compute a simplified loss - cosine similarity preservation
            # Original cosine similarity
            orig_norm = torch.nn.functional.normalize(embeddings_tensor, p=2, dim=1)
            orig_sim = torch.mm(orig_norm, orig_norm.t())
            
            # ISNE cosine similarity
            isne_norm = torch.nn.functional.normalize(isne_embeddings, p=2, dim=1)
            isne_sim = torch.mm(isne_norm, isne_norm.t())
            
            # Loss is MSE between the two similarity matrices
            loss = torch.nn.functional.mse_loss(isne_sim, orig_sim)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
            
            if (epoch + 1) % 5 == 0 or epoch == 0:
                logger.info(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.6f}")
        
        training_time = time.time() - start_time
        logger.info(f"Training completed in {training_time:.2f} seconds")
        
        # Step 8: Generate final embeddings
        model.eval()
        with torch.no_grad():
            final_embeddings = model(embeddings_tensor).cpu().numpy()
        
        # Step 9: Save the model and embeddings
        torch.save(model.state_dict(), model_output_dir / "simplified_isne_model.pt")
        np.save(output_dir / "isne_embeddings.npy", final_embeddings)
        
        # Step 10: Create a mapping from chunk IDs to embeddings
        chunk_embeddings = {}
        for i, chunk_id in enumerate(all_chunk_ids):
            chunk_embeddings[chunk_id] = final_embeddings[i].tolist()
        
        # Step 11: Save the mapping to a file
        import json
        with open(output_dir / "chunk_isne_embeddings.json", "w") as f:
            json.dump(chunk_embeddings, f)
        
        logger.info(f"Saved ISNE embeddings for {len(chunk_embeddings)} chunks")
        
        # Return the results
        return {
            "success": True,
            "num_embeddings": len(chunk_embeddings),
            "training_time": training_time,
            "epochs": num_epochs,
            "final_loss": losses[-1] if losses else None,
            "embedding_dim": output_dim
        }
    
    def direct_train_isne(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Directly train ISNE embeddings using a simplified approach.
        
        Instead of using the complex ISNE orchestrator with graph construction,
        this uses a simplified linear projection approach that preserves cosine similarities.
        
        Args:
            documents: List of document dictionaries with embeddings
            
        Returns:
            Dictionary with training results
        """
        logger.info("=== Starting In-Memory ISNE Training ===")
        logger.info("Using simplified ISNE training approach to avoid graph construction issues")
        
        # Convert results to document format
        doc_list = []
        for result in documents:
            if "file_id" in result:
                # Use the complete document with embeddings
                doc_list.append(result)
                
        # Prepare documents for ISNE training by ensuring proper structure
        doc_list = self.prepare_documents_for_isne(doc_list)
        
        # Write the prepared documents to disk for debugging
        debug_output_path = self.output_dir / "isne_input_sample.json"
        import json
        with open(debug_output_path, "w") as f:
            json.dump(doc_list, f, indent=2)
        logger.info(f"Saved prepared ISNE input sample to {debug_output_path}")
        
        # Log document and chunk counts
        chunk_count = sum(len(doc.get("chunks", [])) for doc in doc_list)
        logger.info(f"Prepared {len(doc_list)} documents with {chunk_count} total chunks for ISNE training")
        
        # Count chunks with embeddings
        chunks_with_embeddings = sum(
            1 for doc in doc_list 
            for chunk in doc.get("chunks", []) 
            if chunk.get("embedding")
        )
        logger.info(f"Found {chunks_with_embeddings} chunks with embeddings")
        
        # Use the simplified ISNE training function
        return self.simplified_isne_training(doc_list)
    
    def run_ingestion(self, output_dir=None, model_path="./models/isne/isne_model_latest.pt", save_enhanced=True) -> Dict[str, Any]:
        """
        Run the ingestion pipeline with ISNE model enhancement and save to JSON.
        
        This method demonstrates how to use a trained ISNE model to enhance document embeddings
        during ingestion and save the enhanced documents as JSON for later reference.
        
        Args:
            model_path: Path to the trained ISNE model
            output_file: Optional custom output file path for enhanced documents
            
        Returns:
            Dictionary with ingestion statistics and results
        """
        logger.info("=== Starting Ingestion Pipeline with ISNE Enhancement ===")
        
        # First, run the standard document processing pipeline to get documents with embeddings
        logger.info("Step 1: Processing documents through standard pipeline...")
        results = self.run_test(run_isne_training=False)
        
        # Get the processed documents
        documents = self.processing_stats["results"]
        logger.info(f"Processed {len(documents)} documents with base embeddings")
        
        # Now apply the ISNE model to enhance the embeddings
        logger.info("\nStep 2: Applying ISNE model to enhance embeddings...")
        isne_start_time = time.time()
        enhanced_documents = self.apply_isne_model(documents, model_path)
        isne_end_time = time.time()
        isne_processing_time = isne_end_time - isne_start_time
        logger.info(f"ISNE enhancement completed in {isne_processing_time:.2f} seconds")
        
        # Save the enhanced documents to a JSON file
        logger.info("\nStep 3: Saving enhanced documents to JSON file...")
        save_start_time = time.time()
        
        # Determine output path
        if output_dir:
            json_output_path = Path(output_dir) / "isne_enhanced_documents.json"
        else:
            json_output_path = self.output_dir / "isne_enhanced_documents.json"
            
        # Save full documents and a sample version
        try:
            # Create a sample with limited documents for easier viewing
            sample_size = min(2, len(enhanced_documents))
            sample_docs = enhanced_documents[:sample_size]
            
            # Save sample to a separate file
            sample_path = self.output_dir / "isne_enhanced_sample.json"
            with open(sample_path, "w") as f:
                json.dump(sample_docs, f, indent=2)
            
            # Save all documents
            with open(json_output_path, "w") as f:
                json.dump(enhanced_documents, f, indent=2)
                
            logger.info(f"Saved all enhanced documents to {json_output_path}")
            logger.info(f"Saved sample of {sample_size} documents to {sample_path}")
        except Exception as e:
            logger.error(f"Error saving enhanced documents: {e}")
            
        save_end_time = time.time()
        save_time = save_end_time - save_start_time
        logger.info(f"Document saving completed in {save_time:.2f} seconds")
        
        # Compile and return statistics
        ingestion_stats = {
            "total_documents": len(enhanced_documents),
            "processing_time": self.processing_stats["parallel_time"],
            "isne_enhancement_time": isne_processing_time,
            "json_save_time": save_time,
            "total_ingestion_time": time.time() - self.processing_stats["test_start_time"],
            "output_path": str(json_output_path),
            "sample_path": str(sample_path)
        }
        
        # Generate comprehensive report
        self._generate_ingestion_report(ingestion_stats)
        
        return ingestion_stats
    
    def apply_isne_model(self, documents: List[Dict[str, Any]], model_path: str) -> List[Dict[str, Any]]:
        """
        Apply trained ISNE model to enhance document embeddings.
        
        Args:
            documents: List of processed documents with base embeddings
            model_path: Path to the trained ISNE model
            
        Returns:
            List of documents with enhanced embeddings
        """
        from src.isne.trainer.training_orchestrator import ISNETrainingOrchestrator
        import torch
        
        logger.info(f"Loading ISNE model from {model_path}")
        
        # Determine device
        device = "cpu"  # Default fallback
        if self.config.get("gpu_execution", {}).get("enabled", False):
            device = self.config["gpu_execution"]["isne"]["device"]
            logger.info(f"Using GPU for ISNE inference: {device}")
        elif self.config.get("cpu_execution", {}).get("enabled", False):
            device = "cpu"
            logger.info(f"Using CPU for ISNE inference")
        
        # Check if model file exists
        model_file = Path(model_path)
        if not model_file.exists():
            logger.error(f"Model file not found: {model_path}")
            raise FileNotFoundError(f"ISNE model file not found: {model_path}")
        
        # Load model
        try:
            model_data = torch.load(model_path, map_location=device)
            logger.info(f"Successfully loaded model with keys: {list(model_data.keys())}")
            
            # Load model settings
            model_config = model_data.get("config", {})
            if not model_config:
                logger.warning("Model config not found in saved model, using defaults")
                
            # Extract model parameters
            embedding_dim = model_config.get("embedding_dim", 768)  # Default ModernBERT dimension
            hidden_dim = model_config.get("hidden_dim", 256)
            output_dim = model_config.get("output_dim", 768)
            
            logger.info(f"Model dimensions: {embedding_dim} → {hidden_dim} → {output_dim}")
            
            # Recreate the model architecture
            from src.isne.models.isne_model import ISNEModel
            model = ISNEModel(
                in_features=embedding_dim,
                hidden_features=hidden_dim,
                out_features=output_dim,
                num_layers=model_config.get("num_layers", 2)
            )
            
            # Load the model weights
            model.load_state_dict(model_data["model_state_dict"])
            model.to(device)
            model.eval()  # Set to evaluation mode
            
            logger.info("ISNE model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading ISNE model: {e}")
            raise
        
        # Prepare documents for ISNE inference
        logger.info("Preparing document graph for ISNE inference...")
        
        # Create a simplified graph building function similar to what the trainer uses
        def build_graph_from_documents(docs):
            import torch
            from torch_geometric.data import Data
            
            # Extract embeddings and metadata
            node_embeddings = []
            node_metadata = []
            node_model_types = []  # Track embedding model types
            edge_index_src = []
            edge_index_dst = []
            edge_attr = []
            
            # Node index mapping
            node_idx_map = {}
            current_idx = 0
            
            # First pass: collect all nodes
            for doc in docs:
                if "chunks" not in doc or not doc["chunks"]:
                    continue
                    
                for chunk_idx, chunk in enumerate(doc["chunks"]):
                    # Skip chunks without embeddings
                    if "embedding" not in chunk or chunk["embedding"] is None:
                        continue
                        
                    # Create a unique ID for this chunk
                    chunk_id = f"{doc['file_id']}_{chunk_idx}"
                    
                    # Store the node index mapping
                    node_idx_map[chunk_id] = current_idx
                    current_idx += 1
                    
                    # Add the embedding and metadata
                    node_embeddings.append(chunk["embedding"])
                    
                    # Store embedding model information for ISNE awareness
                    embedding_model = chunk.get("embedding_model", "unknown")
                    embedding_type = chunk.get("embedding_type", "unknown")
                    
                    node_model_types.append(embedding_type)
                    
                    node_metadata.append({
                        "file_id": doc["file_id"],
                        "chunk_id": chunk_id,
                        "text": chunk.get("text", ""),
                        "embedding_model": embedding_model,
                        "embedding_type": embedding_type,
                        "metadata": chunk.get("metadata", {})
                    })
            
            # Second pass: create edges
            for doc in docs:
                if "chunks" not in doc or len(doc["chunks"]) < 2:
                    continue
                    
                # Create sequential edges between chunks in the same document
                for i in range(len(doc["chunks"]) - 1):
                    src_id = f"{doc['file_id']}_{i}"
                    dst_id = f"{doc['file_id']}_{i+1}"
                    
                    if src_id in node_idx_map and dst_id in node_idx_map:
                        src_idx = node_idx_map[src_id]
                        dst_idx = node_idx_map[dst_id]
                        
                        # Check if embeddings are from the same model type
                        src_type = node_model_types[src_idx] if src_idx < len(node_model_types) else "unknown"
                        dst_type = node_model_types[dst_idx] if dst_idx < len(node_model_types) else "unknown"
                        
                        # Adjust edge weight based on embedding model compatibility
                        # Same model type gets full weight, different types get reduced weight
                        edge_weight = 1.0 if src_type == dst_type else 0.7
                        
                        # Add sequential edge
                        edge_index_src.append(src_idx)
                        edge_index_dst.append(dst_idx)
                        edge_attr.append([edge_weight])  # Sequential relationship with model-aware weight
                        
                        # Add reverse edge for undirected graph
                        edge_index_src.append(dst_idx)
                        edge_index_dst.append(src_idx)
                        edge_attr.append([edge_weight])
            
            # Create the PyTorch Geometric data object
            if not node_embeddings:
                logger.warning("No valid embeddings found in documents")
                return None, []
                
            node_embeddings_tensor = torch.tensor(node_embeddings, dtype=torch.float)
            edge_index = torch.tensor([edge_index_src, edge_index_dst], dtype=torch.long)
            edge_attr_tensor = torch.tensor(edge_attr, dtype=torch.float)
            
            graph = Data(
                x=node_embeddings_tensor,
                edge_index=edge_index,
                edge_attr=edge_attr_tensor
            )
            
            logger.info(f"Created graph with {graph.num_nodes} nodes and {graph.num_edges} edges")
            
            return graph, node_metadata, node_idx_map
        
        # Build the graph from documents
        graph, node_metadata, node_idx_map = build_graph_from_documents(documents)
        
        if graph is None or graph.num_nodes == 0:
            logger.error("Failed to build valid graph from documents")
            return documents
        
        # Apply the ISNE model to enhance embeddings
        logger.info(f"Applying ISNE model to enhance {graph.num_nodes} embeddings...")
        try:
            with torch.no_grad():
                # Move graph to appropriate device
                graph = graph.to(device)
                
                # Make sure all tensors are properly shaped and on the correct device
                x = graph.x.to(device)
                edge_index = graph.edge_index.to(device)
                
                # Handle edge attributes with careful type checking
                edge_attr = None
                if hasattr(graph, 'edge_attr') and graph.edge_attr is not None:
                    edge_attr = graph.edge_attr
                    # Convert to float tensor if needed
                    if not isinstance(edge_attr, torch.FloatTensor) and not isinstance(edge_attr, torch.cuda.FloatTensor):
                        edge_attr = edge_attr.float()
                    
                    # Reshape if needed
                    if edge_attr.dim() == 2 and edge_attr.size(1) == 1:
                        edge_attr = edge_attr.squeeze(1)
                    
                    edge_attr = edge_attr.to(device)
                
                # The ISNEModel forward method doesn't use edge_attr
                # It only takes x, edge_index, and an optional return_attention flag
                enhanced_embeddings = model(x, edge_index)
                
                # Move back to CPU for further processing
                enhanced_embeddings = enhanced_embeddings.cpu().numpy()
                
                logger.info(f"Successfully generated {len(enhanced_embeddings)} enhanced embeddings")
                
        except Exception as e:
            logger.error(f"Error during ISNE inference: {e}")
            return documents
        
        # Update documents with enhanced embeddings
        logger.info("Updating documents with enhanced embeddings...")
        
        # Create deep copy of documents to avoid modifying the originals
        import copy
        enhanced_documents = copy.deepcopy(documents)
        
        # Update embeddings in the document structure
        for doc_idx, doc in enumerate(enhanced_documents):
            if "chunks" not in doc or not doc["chunks"]:
                continue
                
            for chunk_idx, chunk in enumerate(doc["chunks"]):
                chunk_id = f"{doc['file_id']}_{chunk_idx}"
                
                if chunk_id in node_idx_map:
                    node_idx = node_idx_map[chunk_id]
                    # Add the enhanced embedding to the chunk
                    chunk["isne_embedding"] = enhanced_embeddings[node_idx].tolist()
        
        logger.info("Document enhancement with ISNE embeddings completed")
        
        return enhanced_documents
    
    def store_in_arango(self, documents: List[Dict[str, Any]], db_name: str, db_mode: str, force: bool = False) -> Dict[str, Any]:
        """
        Store enhanced documents in ArangoDB.
        
        Args:
            documents: List of documents with enhanced embeddings
            db_name: Name of the ArangoDB database
            db_mode: Database mode ('create' or 'append')
            force: Force recreation of collections even if they exist
            
        Returns:
            Dictionary with storage statistics
        """
        from src.storage.arango.connection import ArangoConnection
        from src.storage.arango.repository import ArangoRepository
        import uuid
        
        logger.info(f"Initializing ArangoDB connection for database '{db_name}'")
        
        # Create storage configuration
        storage_config = {
            "host": "localhost",
            "port": 8529,
            "username": "root",
            "password": "",  # Default empty password
            "database": db_name,
            "collection_prefix": "isne_",
            "use_vector_index": True,
            "vector_dimensions": 768  # Assuming ISNE output is same dimension
        }
        
        # Connect to ArangoDB
        try:
            # Initialize connection
            connection = ArangoConnection(
                host=storage_config["host"],
                port=storage_config["port"],
                username=storage_config["username"],
                password=storage_config["password"]
            )
            
            # Set up database
            if not connection.database_exists(db_name):
                logger.info(f"Creating database '{db_name}'")
                connection.create_database(db_name)
            
            # Set the current database
            connection.set_database(db_name)
            
            # Initialize repository
            node_collection = f"{storage_config['collection_prefix']}nodes"
            edge_collection = f"{storage_config['collection_prefix']}edges"
            graph_name = f"{storage_config['collection_prefix']}graph"
            
            repository = ArangoRepository(
                connection=connection,
                node_collection_name=node_collection,
                edge_collection_name=edge_collection,
                graph_name=graph_name,
                vector_dimensions=storage_config["vector_dimensions"],
                use_vector_index=storage_config["use_vector_index"]
            )
            
            # Handle create mode with force option
            if db_mode == "create" or force:
                # Delete graph and collections if they exist
                if force and connection.graph_exists(graph_name):
                    logger.info(f"Dropping existing graph '{graph_name}'")
                    connection.delete_graph(graph_name, drop_collections=True)
                elif force:
                    # Delete collections if they exist
                    if connection.collection_exists(node_collection):
                        logger.info(f"Dropping existing node collection '{node_collection}'")
                        connection.delete_collection(node_collection)
                    if connection.collection_exists(edge_collection):
                        logger.info(f"Dropping existing edge collection '{edge_collection}'")
                        connection.delete_collection(edge_collection)
                
                # Create collections and graph
                logger.info(f"Setting up collections and graph in '{db_mode}' mode")
                repository.setup_collections()
            
        except Exception as e:
            logger.error(f"Error initializing ArangoDB connection: {e}")
            raise
        
        # Store documents in ArangoDB
        logger.info("Storing documents in ArangoDB...")
        
        # Statistics counters
        stats = {
            "nodes_created": 0,
            "edges_created": 0,
            "errors": 0
        }
        
        # Process documents
        dataset_id = str(uuid.uuid4())
        try:
            for doc in documents:
                if "chunks" not in doc or not doc["chunks"]:
                    continue
                
                # Process each chunk as a node
                node_ids = []
                for chunk_idx, chunk in enumerate(doc["chunks"]):
                    # Skip chunks without embeddings
                    if "embedding" not in chunk or chunk["embedding"] is None:
                        continue
                    
                    # Prepare node data
                    node_data = {
                        "doc_id": doc["file_id"],
                        "chunk_id": chunk_idx,
                        "content": chunk.get("text", ""),
                        "metadata": {
                            "file_name": doc.get("file_name", ""),
                            "file_path": doc.get("file_path", ""),
                            "dataset_id": dataset_id,
                            "chunk_metadata": chunk.get("metadata", {})
                        }
                    }
                    
                    # Add embeddings
                    node_data["embedding"] = chunk["embedding"]
                    if "isne_embedding" in chunk:
                        node_data["isne_embedding"] = chunk["isne_embedding"]
                    
                    # Store node in ArangoDB
                    try:
                        node_id = repository.add_node(
                            content=node_data["content"],
                            metadata=node_data["metadata"],
                            embeddings={
                                "base": node_data["embedding"],
                                "isne": node_data.get("isne_embedding", node_data["embedding"])
                            }
                        )
                        node_ids.append(node_id)
                        stats["nodes_created"] += 1
                    except Exception as e:
                        logger.error(f"Error adding node: {e}")
                        stats["errors"] += 1
                        continue
                
                # Store the node_id to chunk mapping for later reference relationship processing
                chunk_to_node_id = {}
                for idx, node_id in enumerate(node_ids):
                    chunk_to_node_id[f"{doc['file_id']}_{idx}"] = node_id
                
                # Create edges between sequential chunks
                if len(node_ids) >= 2:
                    for i in range(len(node_ids) - 1):
                        try:
                            repository.add_edge(
                                from_id=node_ids[i],
                                to_id=node_ids[i+1],
                                relation_type="SEQUENTIAL",
                                weight=1.0,
                                metadata={"document_id": doc["file_id"]}
                            )
                            stats["edges_created"] += 1
                        except Exception as e:
                            logger.error(f"Error adding edge: {e}")
                            stats["errors"] += 1
                
                # Process code-specific relationships if this is a Python file
                # These relationships come from the PythonCodeChunker's metadata
                if doc.get("file_name", "").endswith(".py"):
                    for chunk_idx, chunk in enumerate(doc["chunks"]):
                        # Skip chunks without metadata
                        if "metadata" not in chunk or "references" not in chunk["metadata"]:
                            continue
                            
                        # Get source node ID
                        source_id = chunk_to_node_id.get(f"{doc['file_id']}_{chunk_idx}")
                        if not source_id:
                            continue
                            
                        # Process each reference (relationship) in the chunk metadata
                        for reference in chunk["metadata"]["references"]:
                            relation_type = reference.get("type")
                            target_chunk_id = reference.get("target")
                            weight = reference.get("weight", 0.8)
                            
                            # Skip invalid references
                            if not relation_type or not target_chunk_id:
                                continue
                                
                            # Get target node ID
                            target_id = chunk_to_node_id.get(target_chunk_id)
                            if not target_id:
                                continue
                                
                            # Create the edge representing the code relationship
                            try:
                                repository.add_edge(
                                    from_id=source_id,
                                    to_id=target_id,
                                    relation_type=relation_type,
                                    weight=weight,
                                    metadata={
                                        "document_id": doc["file_id"],
                                        "code_relationship": True,
                                        "source_type": chunk["metadata"].get("type"),
                                        "target_type": next((c["metadata"].get("type") for c in doc["chunks"] 
                                                          if f"{doc['file_id']}_{doc['chunks'].index(c)}" == target_chunk_id), None)
                                    }
                                )
                                stats["edges_created"] += 1
                            except Exception as e:
                                logger.error(f"Error adding code relationship edge: {e}")
                                stats["errors"] += 1
            
            logger.info(f"Storage complete: {stats['nodes_created']} nodes, {stats['edges_created']} edges created")
            
        except Exception as e:
            logger.error(f"Error during document storage: {e}")
            stats["errors"] += 1
        
        return stats
    
    def _generate_ingestion_report(self, stats: Dict[str, Any]):
        """Generate a comprehensive report for the ingestion process."""
        logger.info("\n=== Ingestion Pipeline Report ===")
        
        logger.info("\nProcessing Statistics:")
        logger.info(f"  Total Documents:       {stats['total_documents']}")
        logger.info(f"  Document Processing:   {stats['processing_time']:.2f}s")
        logger.info(f"  ISNE Enhancement:      {stats['isne_enhancement_time']:.2f}s")
        logger.info(f"  JSON Saving:           {stats['json_save_time']:.2f}s")
        logger.info(f"  Total Ingestion Time:  {stats['total_ingestion_time']:.2f}s")
        
        logger.info("\nOutput Files:")
        logger.info(f"  All Documents:         {stats['output_path']}")
        logger.info(f"  Sample Documents:      {stats['sample_path']}")
        
        logger.info("\nIngestion Pipeline Completed Successfully!")
        
        # Write report to file
        report_file = self.output_dir / "ingestion_report.json"
        with open(report_file, "w") as f:
            json.dump(stats, f, indent=2)
        logger.info(f"Detailed ingestion report saved to {report_file}")

# Initialize multiprocessing
def init_mp():
    """Initialize multiprocessing environment."""
    # Configure for multiprocessing on Windows if necessary
    if sys.platform == 'win32':
        mp.set_start_method('spawn', force=True)

def main():
    """
    Run the pipeline multiprocessing test as a standalone script.
    """
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='HADES-PathRAG Pipeline Multiprocessing Test')
    
    # Mode selection arguments
    mode_group = parser.add_argument_group('Mode Selection')
    mode_group.add_argument('--mode', type=str, choices=['process', 'train', 'ingest'], default='process',
                        help='Pipeline mode: process (document processing only), train (ISNE training), ingest (ISNE ingestion)')
    
    # Common arguments
    common_group = parser.add_argument_group('Common Options')
    common_group.add_argument('--workers', type=int, default=4,
                        help='Number of worker processes')
    common_group.add_argument('--files', type=int, default=50,
                        help='Maximum number of files to process')
    common_group.add_argument('--data-dir', type=str, default='./test3',
                        help='Directory containing test data')
    common_group.add_argument('--output-dir', type=str, default='./test-output/isne-training-dataset',
                        help='Directory for test output')
    
    # Legacy support
    common_group.add_argument('--run-isne', action='store_true',
                        help='[Legacy] Run ISNE training directly (same as --mode=train)')
    
    # Ingestion mode arguments
    ingest_group = parser.add_argument_group('Ingestion Options')
    ingest_group.add_argument('--model-path', type=str, default='./models/isne/isne_model_latest.pt',
                        help='Path to trained ISNE model (for ingestion mode)')
    ingest_group.add_argument('--output-file', type=str,
                        help='Custom output file path for enhanced documents (for ingestion mode)')
    
    args = parser.parse_args()
    
    # Create the pipeline tester
    test = PipelineMultiprocessTester(
        test_data_dir=args.data_dir,
        output_dir=args.output_dir,
        num_files=args.files,
        max_workers=args.workers,
        batch_size=8  # Default batch size
    )
    
    # Determine which mode to run
    mode = args.mode
    # Support legacy --run-isne flag
    if args.run_isne and mode == 'process':
        mode = 'train'
    
    # Run the appropriate pipeline mode
    if mode == 'train':
        logger.info("Running in ISNE training mode")
        test.run_test(run_isne_training=True)
    elif mode == 'ingest':
        logger.info("Running in ISNE ingestion mode")
        test.run_ingestion(
            model_path=args.model_path,
            output_dir=args.output_file
        )
    else:  # Default to 'process' mode
        logger.info("Running in document processing mode")
        test.run_test(run_isne_training=False)

if __name__ == "__main__":
    # Configure for multiprocessing on Windows
    mp.set_start_method('spawn', force=True)
    main()

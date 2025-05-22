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
from src.embedding.adapters.modernbert_adapter import ModernBERTEmbeddingAdapter
from src.docproc.adapters.docling_adapter import EXTENSION_TO_FORMAT

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
def init_docproc(config: Dict[str, Any], worker_id: int) -> Dict[str, Any]:
    """Initialize document processor with appropriate configuration."""
    logger.info(f"[Worker {worker_id}] Initializing document processor...")
    
    # Configure document processor based on execution mode
    docproc_options = {}
    
    if config.get("gpu_execution", {}).get("enabled", False):
        device = config["gpu_execution"]["docproc"]["device"]
        logger.info(f"[Worker {worker_id}] Using GPU configuration with device: {device}")
        docproc_options = {
            "device": device,
            "use_gpu": True
        }
    else:
        num_threads = config["cpu_execution"]["docproc"]["num_threads"]
        logger.info(f"[Worker {worker_id}] Using CPU configuration with {num_threads} threads")
        docproc_options = {
            "device": "cpu",
            "use_gpu": False,
            "num_threads": num_threads
        }
    
    # Initialize document processor with the appropriate options
    manager = DocumentProcessorManager(options=docproc_options)
    
    return {"processor": manager, "worker_id": worker_id}

def init_chunker(config: Dict[str, Any], worker_id: int) -> Dict[str, Any]:
    """Initialize chunker with appropriate device."""
    logger.info(f"[Worker {worker_id}] Initializing chunker...")
    
    # Configure chunking based on execution mode
    if config.get("gpu_execution", {}).get("enabled", False):
        device = config["gpu_execution"]["chunking"]["device"]
        batch_size = config["gpu_execution"]["chunking"]["batch_size"]
        logger.info(f"[Worker {worker_id}] Using GPU configuration for chunking with device: {device}")
    else:
        device = config["cpu_execution"]["chunking"]["device"]
        num_threads = config["cpu_execution"]["chunking"]["num_threads"]
        logger.info(f"[Worker {worker_id}] Using CPU configuration for chunking with {num_threads} threads")
    
    # No specific initialization needed for chunker as it's configured through config files
    return {"worker_id": worker_id}

def init_embedding(config: Dict[str, Any], worker_id: int) -> Dict[str, Any]:
    """Initialize embedding adapter with appropriate device."""
    logger.info(f"[Worker {worker_id}] Initializing embedding adapter...")
    
    # Configure embedding based on execution mode
    if config.get("gpu_execution", {}).get("enabled", False):
        device = config["gpu_execution"]["embedding"]["device"]
        batch_size = config["gpu_execution"]["embedding"]["batch_size"]
        precision = config["gpu_execution"]["embedding"]["model_precision"]
        logger.info(f"[Worker {worker_id}] Using GPU configuration for embedding with device: {device}, precision: {precision}")
        adapter = ModernBERTEmbeddingAdapter(device=device)
    else:
        device = config["cpu_execution"]["embedding"]["device"]
        num_threads = config["cpu_execution"]["embedding"]["num_threads"]
        logger.info(f"[Worker {worker_id}] Using CPU configuration for embedding with {num_threads} threads")
        adapter = ModernBERTEmbeddingAdapter(device=device)
    
    return {"adapter": adapter, "worker_id": worker_id}

# Process a document using multiprocessing
def process_document(args: Tuple[Path, Dict[str, Any], int]) -> Dict[str, Any]:
    """Process a single document using the pipeline components."""
    file_path, config, worker_id = args
    start_time = time.time()
    
    # Generate a consistent ID for the file based on its path
    file_id = f"pdf_{abs(hash(str(file_path))) % 10000:04d}_{file_path.stem}"
    
    logger.info(f"[Worker {worker_id}] Starting processing of {file_path.name}")
    
    result = {
        "file_id": file_id,
        "file_name": file_path.name,
        "file_size": file_path.stat().st_size,
        "worker_id": worker_id,
        "chunks": [],
        "embeddings": [],
        "timing": {
            "document_processing": 0,
            "chunking": 0,
            "embedding": 0,
            "total": 0
        }
    }
    
    try:
        # STEP 1: Document Processing
        doc_start = time.time()
        processor_config = init_docproc(config, worker_id)
        processor = processor_config["processor"]
        
        logger.info(f"[Worker {worker_id}] Starting document processing: {file_path.name} ({file_path.stat().st_size/1024:.1f} KB)")
        processed_doc = processor.process_document(path=str(file_path), doc_type="pdf")
        
        # Extract metadata
        content_length = len(str(processed_doc.get("content", "")))
        metadata_count = len(processed_doc.get("metadata", {}))
        logger.info(f"[Worker {worker_id}] Document processed: {file_path.name} - Content size: {content_length/1024:.1f} KB, Metadata fields: {metadata_count}")
        
        doc_time = time.time() - doc_start
        result["timing"]["document_processing"] = doc_time
        
        # STEP 2: Chunking
        chunk_start = time.time()
        
        # Get document properties
        doc_id = processed_doc.get('id', file_id)
        doc_path = processed_doc.get('source', str(file_path))
        
        logger.info(f"[Worker {worker_id}] Starting document chunking: {file_path.name}")
        doc_content = processed_doc.get("content", "")
        content_size = len(str(doc_content))
        logger.info(f"[Worker {worker_id}] Extracted content for chunking: {content_size/1024:.1f} KB")
        
        # Process through chunking
        chunks_result = chunk_text(
            content=doc_content,
            doc_id=doc_id,
            path=doc_path,
            doc_type="academic_pdf",
            max_tokens=1024,
            output_format="json"
        )
        
        # Extract chunks
        chunks = chunks_result.get("chunks", [])
        result["chunks"] = chunks
        
        # Calculate stats on chunks
        chunk_sizes = [len(str(chunk.get("content", ""))) for chunk in chunks]
        avg_chunk_size = sum(chunk_sizes) / len(chunk_sizes) if chunk_sizes else 0
        max_chunk_size = max(chunk_sizes) if chunk_sizes else 0
        min_chunk_size = min(chunk_sizes) if chunk_sizes else 0
        
        logger.info(f"[Worker {worker_id}] Created {len(chunks)} chunks for {file_path.name} - "
                  f"Avg size: {avg_chunk_size/1024:.1f} KB, Range: {min_chunk_size/1024:.1f}-{max_chunk_size/1024:.1f} KB")
        
        chunk_time = time.time() - chunk_start
        result["timing"]["chunking"] = chunk_time
        
        # STEP 3: Embedding generation
        embed_start = time.time()
        
        # Initialize embedding adapter
        embedding_config = init_embedding(config, worker_id)
        adapter = embedding_config["adapter"]
        
        # Extract text content from each chunk for embedding
        texts = [chunk.get("content", "") for chunk in result["chunks"]]
        
        if not texts:
            logger.warning(f"[Worker {worker_id}] No text content to embed for {file_path.name}")
            result["timing"]["embedding"] = time.time() - embed_start
            return result
        
        # Calculate total text size for embedding
        total_text_size = sum(len(text) for text in texts)
        avg_text_size = total_text_size / len(texts) if texts else 0
        
        logger.info(f"[Worker {worker_id}] Starting embedding generation for {len(texts)} chunks from {file_path.name} "
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
            
            logger.info(f"[Worker {worker_id}] Processing embedding batch {i//batch_size + 1}/"
                      f"{(len(texts) + batch_size - 1)//batch_size} "
                      f"({len(batch_texts)} chunks, {batch_size_bytes/1024:.1f} KB)")
            
            # Create a new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                # Run the async embed method in a synchronous context
                batch_embeddings = loop.run_until_complete(adapter.embed(batch_texts))
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
            if i < len(result["chunks"]):
                result["chunks"][i]["embedding"] = emb_list
        
        result["embeddings"] = embeddings_list
        
        embed_time = time.time() - embed_start
        result["timing"]["embedding"] = embed_time
        
        # Total time
        result["timing"]["total"] = time.time() - start_time
        
        # Add timestamp to show when this task completed
        result["completed_at"] = datetime.now().isoformat()
        
        logger.info(f"[Worker {worker_id}] Completed {file_path.name}: {len(result['chunks'])} chunks, {len(embeddings)} embeddings in {result['timing']['total']:.2f}s")
        
    except Exception as e:
        logger.error(f"[Worker {worker_id}] Error processing document {file_path.name}: {str(e)}", exc_info=True)
    
    return result

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
        num_files: int = 20,
        max_workers: int = 4,
        batch_size: int = 8
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
        
        # Use EXTENSION_TO_FORMAT to find all supported file types
        for extension in EXTENSION_TO_FORMAT.keys():
            # Remove the leading dot for the glob pattern, use ** for recursive search
            pattern = f"**/*{extension}"
            files = list(self.test_data_dir.glob(pattern))
            supported_files.extend(files)
            
            # Log the file types found
            if files:
                logger.info(f"Found {len(files)} {extension} files")
                
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
    
    def direct_train_isne(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Demonstrate direct in-memory handoff of document data to ISNE training.
        
        This method shows how to pass document objects directly to the ISNE training
        orchestrator without writing them to disk first, which is more efficient for
        production environments.
        
        Args:
            results: List of processed document results from the pipeline
        
        Returns:
            Training results from the ISNE training orchestrator
        """
        from src.isne.trainer.training_orchestrator import ISNETrainingOrchestrator
        from src.isne.training.random_walk_sampler import RandomWalkSampler
        
        logger.info("Demonstrating direct in-memory handoff to ISNE training...")
        
        # Prepare documents for ISNE training (keep full embeddings)
        documents = []
        for result in results:
            if "file_id" in result:
                # Use the complete document with embeddings
                documents.append(result)
        
        logger.info(f"Passing {len(documents)} documents directly to ISNE training orchestrator")
        
        # Determine device from configuration
        device = "cpu"  # Default fallback
        if self.config.get("gpu_execution", {}).get("enabled", False):
            device = self.config["gpu_execution"]["isne"]["device"]
            logger.info(f"Using GPU for ISNE training: {device}")
        elif self.config.get("cpu_execution", {}).get("enabled", False):
            device = "cpu"
            logger.info(f"Using CPU for ISNE training")
        
        # Create advanced sampler configuration
        sampler_config = {
            "sampler_class": RandomWalkSampler,
            "sampler_params": {
                "walk_length": 6,        # Length of random walks for positive pair sampling
                "context_size": 4,        # Context window size for positive pairs
                "walks_per_node": 10,     # Number of walks per starting node
                "p": 1.0,                # Return parameter in Node2Vec terminology
                "q": 0.7,                # In-out parameter in Node2Vec terminology
                "num_negative_samples": 1, # Ratio of negative to positive samples
                # Enable batch-aware sampling by setting flag (will be detected by trainer)
                "use_batch_aware_sampling": True  # This will dramatically reduce filtering
            }
        }
        
        logger.info(f"Using RandomWalkSampler with batch-aware sampling for reduced filtering rate")
            
        # Create the orchestrator with in-memory documents and improved sampler
        orchestrator = ISNETrainingOrchestrator(
            documents=documents,  # Pass documents directly in memory
            output_dir=self.output_dir / "isne-training",
            model_output_dir=Path("./models/isne"),
            device=device,  # Use device from configuration
            sampler_config=sampler_config  # Use our improved sampler implementation
        )
        
        # Run training with timing
        logger.info("Starting ISNE training with in-memory documents...")
        isne_start_time = time.time()
        training_results = orchestrator.train()
        isne_end_time = time.time()
        self.isne_training_time = isne_end_time - isne_start_time
        logger.info(f"ISNE training completed in {self.isne_training_time:.2f} seconds")
        
        # Store and return training results
        self.isne_results = training_results
        
        return training_results
        
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
                    node_metadata.append({
                        "file_id": doc["file_id"],
                        "chunk_id": chunk_id,
                        "text": chunk.get("text", ""),
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
                        # Add sequential edge
                        edge_index_src.append(node_idx_map[src_id])
                        edge_index_dst.append(node_idx_map[dst_id])
                        edge_attr.append([1.0])  # Sequential relationship
                        
                        # Add reverse edge for undirected graph
                        edge_index_src.append(node_idx_map[dst_id])
                        edge_index_dst.append(node_idx_map[src_id])
                        edge_attr.append([1.0])
            
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
    common_group.add_argument('--files', type=int, default=10,
                        help='Maximum number of files to process')
    common_group.add_argument('--data-dir', type=str, default='./test-data',
                        help='Directory containing test data')
    common_group.add_argument('--output-dir', type=str, default='./test-output/pipeline-mp-test',
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

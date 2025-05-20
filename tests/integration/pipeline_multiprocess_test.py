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
        """Find all supported file types in the test directory."""
        supported_files = []
        
        # Use EXTENSION_TO_FORMAT to find all supported file types
        for extension in EXTENSION_TO_FORMAT.keys():
            # Remove the leading dot for the glob pattern
            pattern = f"*{extension}"
            files = list(self.test_data_dir.glob(pattern))
            supported_files.extend(files)
            
            # Log the file types found
            if files:
                logger.info(f"Found {len(files)} {extension} files")
        
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
        save_start = time.time()
        output_file = self.output_dir / "pipeline_stats.json"
        with open(output_file, "w") as f:
            json.dump(self.stats, f, indent=2)
        
        # Save all processed documents for inspection
        sample_output = self.output_dir / "isne_input_sample.json"
        with open(sample_output, "w") as f:
            # Include all processed documents
            sample_data = []
            for result in results:
                if "file_id" in result:
                    # Create a simplified sample without large embedding vectors
                    sample_item = {
                        "file_id": result["file_id"],
                        "file_name": result["file_name"],
                        "chunk_count": len(result.get("chunks", [])),
                        "embedding_count": len(result.get("embeddings", [])),
                        "chunks": result.get("chunks", [])
                    }
                    sample_data.append(sample_item)
            json.dump(sample_data, f, indent=2)
        
        logger.info(f"Saved statistics to {output_file}")
        logger.info(f"Saved sample ISNE input to {sample_output}")
        save_time = time.time() - save_start
        
        # Calculate average processing times
        if self.stats["total_files"] > 0:
            avg_doc_time = self.stats["processing_time"]["document_processing"] / self.stats["total_files"]
            avg_chunk_time = self.stats["processing_time"]["chunking"] / self.stats["total_files"]
            avg_embed_time = self.stats["processing_time"]["embedding"] / self.stats["total_files"]
        else:
            avg_doc_time = avg_chunk_time = avg_embed_time = 0
        
        # Calculate throughput
        total_runtime = time.time() - test_start_time
        files_per_second = self.stats["total_files"] / total_runtime if total_runtime > 0 else 0
        chunks_per_second = self.stats["total_chunks"] / total_runtime if total_runtime > 0 else 0
        
        # Calculate parallelization efficiency
        theoretical_speedup = self.max_workers
        sequential_runtime = sum(r["timing"].get("total", 0) for r in results if "timing" in r)
        actual_speedup = sequential_runtime / parallel_time if parallel_time > 0 else 0
        efficiency = actual_speedup / theoretical_speedup if theoretical_speedup > 0 else 0
        
        # Performance report
        logger.info("\n=== Performance Report ===")
        logger.info("\nTiming Breakdown:")
        logger.info(f"  Setup:                0.00s")
        logger.info(f"  File Discovery:       {file_discovery_time:.2f}s")
        logger.info(f"  Parallel Processing:  {parallel_time:.2f}s")
        logger.info(f"  Statistics Update:    {stats_time:.2f}s")
        logger.info(f"  Results Saving:       {save_time:.2f}s")
        logger.info(f"  Total Runtime:        {total_runtime:.2f}s")
        
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
        logger.info(f"  - Generated {self.stats['total_embeddings']} embeddings with dimension {self.config.get('embedding', {}).get('dimension', 'unknown')}")
        logger.info(f"  - Saved results to {self.output_dir}")
        
        # Optional: Run ISNE training directly with in-memory data if requested
        if run_isne_training:
            logger.info("\n=== Starting In-Memory ISNE Training ===")
            try:
                # Run the direct ISNE training with the documents we already have in memory
                isne_results = self.direct_train_isne(results)
                logger.info(f"ISNE Training Completed Successfully!")
                logger.info(f"  - Trained model saved to {Path('./models/isne')}")
                # Add ISNE results to our stats
                self.stats["isne_training"] = isne_results
            except Exception as e:
                logger.error(f"Error during in-memory ISNE training: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
        
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
        
        logger.info("Demonstrating direct in-memory handoff to ISNE training...")
        
        # Prepare documents for ISNE training (keep full embeddings)
        documents = []
        for result in results:
            if "file_id" in result:
                # Use the complete document with embeddings
                documents.append(result)
        
        logger.info(f"Passing {len(documents)} documents directly to ISNE training orchestrator")
        
        # Create the orchestrator with in-memory documents
        orchestrator = ISNETrainingOrchestrator(
            documents=documents,  # Pass documents directly in memory
            output_dir=self.output_dir / "isne-training",
            model_output_dir=Path("./models/isne"),
            device="cpu"  # Use CPU as per user preference for now
        )
        
        # Run training
        logger.info("Starting ISNE training with in-memory documents...")
        training_results = orchestrator.train()
        
        # Return training results
        return training_results

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
    parser = argparse.ArgumentParser(description='Pipeline Multiprocessing Test')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of worker processes')
    parser.add_argument('--files', type=int, default=10,
                        help='Maximum number of files to process')
    parser.add_argument('--data-dir', type=str, default='./test-data',
                        help='Directory containing test data')
    parser.add_argument('--output-dir', type=str, default='./test-output/pipeline-mp-test',
                        help='Directory for test output')
    parser.add_argument('--run-isne', action='store_true',
                        help='Run ISNE training directly with in-memory data after pipeline processing')
    
    args = parser.parse_args()
    
    # Run the test
    test = PipelineMultiprocessTester(
        test_data_dir=args.data_dir,
        output_dir=args.output_dir,
        num_files=args.files,
        max_workers=args.workers,
        batch_size=8  # Default batch size
    )
    
    test.run_test(run_isne_training=args.run_isne)

if __name__ == "__main__":
    # Configure for multiprocessing on Windows
    mp.set_start_method('spawn', force=True)
    main()

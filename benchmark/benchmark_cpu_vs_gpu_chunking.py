#!/usr/bin/env python3
"""
Benchmark script for comparing CPU vs GPU performance in document chunking.

This script provides a direct comparison between CPU and GPU performance
for the chunking pipeline, specifically for the Chonky chunker with the
mirth/chonky_modernbert_large_1 model.
"""

import os
import sys
import time
import json
import logging
import argparse
import tempfile
import subprocess
import re
import hashlib
from pathlib import Path
from multiprocessing.pool import ThreadPool

# Add the project root to the Python path
script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, script_dir)

from src.docproc.adapters.docling_adapter import DoclingAdapter
from src.chunking.text_chunkers.chonky_chunker import chunk_text, get_chunker_config, _get_splitter_with_engine
from src.model_engine.engines.haystack import HaystackModelEngine
from src.model_engine.engines.haystack.runtime import ModelClient
from src.model_engine.engines.haystack.runtime.server import run_server

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("benchmark_chunking")

# Constants
DEFAULT_TEST_DATA = os.path.join(script_dir, "test-data")
DEFAULT_OUTPUT_DIR = os.path.join(script_dir, "benchmark-results")
DEFAULT_MODEL_ID = "mirth/chonky_modernbert_large_1"
DEFAULT_SOCKET_PATH = "/tmp/hades_model_mgr_benchmark.sock"
DEFAULT_DEVICE = "cuda:0"
DEFAULT_CPU_DEVICE = "cpu"

def start_model_server(socket_path, device):
    """Start the Haystack model server in a separate process.
    
    Args:
        socket_path: Path to the Unix domain socket for the server
        device: Device to use for model loading (cpu or cuda)
        
    Returns:
        subprocess.Popen: The process running the server
    """
    logger.info(f"Starting Haystack model server at {socket_path} on {device}")
    
    # Remove existing socket if present
    if os.path.exists(socket_path):
        os.unlink(socket_path)
    
    # Set environment variables for the server
    env = os.environ.copy()
    env["HADES_MODEL_MGR_SOCKET"] = socket_path
    env["HADES_DEFAULT_DEVICE"] = device
    
    # Start the server process
    server_script = os.path.join(
        script_dir, "src", "model_engine", "engines", "haystack", 
        "runtime", "server.py"
    )
    
    process = subprocess.Popen(
        [sys.executable, server_script, socket_path],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Wait for the server socket to appear (up to 10 seconds)
    for _ in range(20):
        if os.path.exists(socket_path):
            logger.info(f"Server socket created at {socket_path}")
            break
        time.sleep(0.5)
    else:
        logger.error(f"Timed out waiting for server socket at {socket_path}")
        process.terminate()
        stdout, stderr = process.communicate()
        logger.error(f"Server stdout: {stdout}")
        logger.error(f"Server stderr: {stderr}")
        raise RuntimeError("Failed to start model server")
    
    return process

def setup_model_engine(socket_path, model_id, device):
    """Set up the Haystack model engine and load the model.
    
    Args:
        socket_path: Path to the server socket
        model_id: ID of the model to load
        device: Device to use for model loading
        
    Returns:
        HaystackModelEngine: The initialized model engine
    """
    logger.info(f"Setting up Haystack model engine with socket {socket_path}")
    
    # Initialize the model engine
    engine = HaystackModelEngine(socket_path=socket_path)
    
    # Start the engine
    if not engine.start():
        logger.error("Failed to start Haystack model engine")
        raise RuntimeError("Failed to start model engine")
    
    # Test the connection
    if not engine.client or not engine.running:
        logger.error("Model engine is not running")
        raise RuntimeError("Model engine is not running")
    
    # Load the model
    logger.info(f"Loading model {model_id} on {device}")
    try:
        result = engine.load_model(model_id, device=device)
        logger.info(f"Model load result: {result}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise RuntimeError(f"Failed to load model: {e}")
    
    # Get loaded models
    models = engine.get_loaded_models()
    logger.info(f"Loaded models: {models}")
    
    return engine

def process_pdf(adapter, pdf_path):
    """Process a PDF file using the Docling adapter.
    
    Args:
        adapter: DoclingAdapter instance
        pdf_path: Path to the PDF file
        
    Returns:
        dict: The processed document
    """
    logger.info(f"Processing PDF: {pdf_path}")
    start_time = time.time()
    
    # Process the document
    document = adapter.process(Path(pdf_path))
    
    processing_time = time.time() - start_time
    logger.info(f"Processed {pdf_path} in {processing_time:.2f} seconds")
    logger.info(f"Document ID: {document['id']}")
    logger.info(f"Content length: {len(document['content'])} characters")
    
    return document, processing_time

def chunk_document_cpu(document, model_id, num_workers=8):
    """Chunk a document using CPU-only semantic processing with optimized processing.
    
    Args:
        document: The document to chunk
        model_id: ID of the model to use
        num_workers: Number of available CPU cores (used for logging only)
        
    Returns:
        tuple: (chunked document, processing time)
    """
    logger.info(f"CPU Chunking document {document['id']} with model {model_id} using optimized CPU approach")

    # Store original environment settings
    original_cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    original_default_device = os.environ.get("HADES_DEFAULT_DEVICE", "")

    # Ensure we're using CPU by setting environment variables
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["HADES_DEFAULT_DEVICE"] = "cpu"

    start_time = time.time()
    
    try:
        # Get document details
        content = document['content']
        doc_id = document['id']
        path = document.get('path', 'unknown')
        doc_type = document.get('type', 'text')
        
        # Initialize the ParagraphSplitter with CPU device
        from chonky import ParagraphSplitter
        logger.info(f"Creating CPU ParagraphSplitter with model {model_id}")
        splitter = ParagraphSplitter(model_id, device="cpu")
        
        # Determine if we should segment the document for improved performance
        # Large documents benefit from being processed in chunks
        content_length = len(content)
        min_segment_length = 10000  # Minimum segment size to consider splitting
        
        if content_length > min_segment_length:
            logger.info(f"Processing large document ({content_length} chars) in segments")
            
            # Find natural break points (paragraph boundaries)
            paragraph_breaks = [m.start() for m in re.finditer(r'\n\s*\n', content)]
            paragraph_breaks = [0] + paragraph_breaks + [len(content)]
            
            # Calculate reasonable number of segments based on document size
            optimal_segment_count = min(num_workers, max(2, content_length // 10000))
            segment_boundaries = []
            
            # Try to create segment_count evenly sized segments at paragraph boundaries
            if len(paragraph_breaks) <= optimal_segment_count:
                # Not enough paragraph breaks, use what we have
                segment_boundaries = paragraph_breaks
            else:
                # Select boundaries to create approximately equal segments
                target_size = content_length / optimal_segment_count
                current_size = 0
                last_boundary = 0
                segment_boundaries = [0]  # Always start at the beginning
                
                for boundary in paragraph_breaks[1:-1]:  # Skip first (0) and last (len)
                    current_size += (boundary - last_boundary)
                    last_boundary = boundary
                    
                    if current_size >= target_size:
                        segment_boundaries.append(boundary)
                        current_size = 0
                
                segment_boundaries.append(len(content))  # Always include the end
            
            # Process each segment and collect results
            all_paragraphs = []
            for i in range(len(segment_boundaries) - 1):
                start = segment_boundaries[i]
                end = segment_boundaries[i+1]
                segment_text = content[start:end]
                
                logger.info(f"Processing segment {i+1}/{len(segment_boundaries)-1} ({len(segment_text)} chars)")
                segment_paragraphs = list(splitter(segment_text))
                all_paragraphs.extend(segment_paragraphs)
            
            logger.info(f"Split document into {len(all_paragraphs)} semantic paragraphs across {len(segment_boundaries)-1} segments")
            
            # Create chunks from all paragraphs
            chunks = []
            for i, para_text in enumerate(all_paragraphs):
                chunk_id = f"{doc_id}_p{i}"
                
                # Create chunk dictionary with metadata
                chunk = {
                    "id": chunk_id,
                    "parent": doc_id,
                    "parent_id": doc_id,
                    "path": path,
                    "type": doc_type,
                    "content": para_text,
                    "overlap_context": {
                        "pre": "",
                        "post": "",
                        "position": i,
                        "total": len(all_paragraphs)
                    },
                    "symbol_type": "paragraph",
                    "name": f"paragraph_{i}",
                    "chunk_index": i,
                    "line_start": 0,
                    "line_end": 0,
                    "token_count": len(para_text.split()),
                    "content_hash": hashlib.md5(para_text.encode()).hexdigest(),
                    "embedding": None
                }
                chunks.append(chunk)
        else:
            # For smaller documents, process the entire content at once
            logger.info(f"Processing small document ({content_length} chars) as a single unit")
            paragraphs = list(splitter(content))
            logger.info(f"Split document into {len(paragraphs)} semantic paragraphs")
            
            # Create chunks from paragraphs
            chunks = []
            for i, para_text in enumerate(paragraphs):
                chunk_id = f"{doc_id}_p{i}"
                
                # Create chunk dictionary with metadata
                chunk = {
                    "id": chunk_id,
                    "parent": doc_id,
                    "parent_id": doc_id,
                    "path": path,
                    "type": doc_type,
                    "content": para_text,
                    "overlap_context": {
                        "pre": "",
                        "post": "",
                        "position": i,
                        "total": len(paragraphs)
                    },
                    "symbol_type": "paragraph",
                    "name": f"paragraph_{i}",
                    "chunk_index": i,
                    "line_start": 0,
                    "line_end": 0,
                    "token_count": len(para_text.split()),
                    "content_hash": hashlib.md5(para_text.encode()).hexdigest(),
                    "embedding": None
                }
                chunks.append(chunk)
        
        chunking_time = time.time() - start_time
        return chunks, chunking_time
    finally:
        # Restore original environment variables
        os.environ["CUDA_VISIBLE_DEVICES"] = original_cuda_devices
        os.environ["HADES_DEFAULT_DEVICE"] = original_default_device

def chunk_document_gpu(document, model_id, socket_path, device="cuda:0"):
    """Chunk a document using GPU-accelerated processing.
    
    Args:
        document: The document to chunk
        model_id: ID of the model to use
        socket_path: Path to the Haystack model server socket
        device: GPU device to use
        
    Returns:
        tuple: (chunked document, processing time)
    """
    logger.info(f"GPU Chunking document {document['id']} with model {model_id}")
    
    # Enable GPU by resetting the environment variable and setting device
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        del os.environ["CUDA_VISIBLE_DEVICES"]
    os.environ["HADES_DEFAULT_DEVICE"] = device
    
    # Configure environment for chunking with our socket path
    os.environ["HADES_MODEL_MGR_SOCKET"] = socket_path
    os.environ["HADES_RUNTIME_AUTOSTART"] = "1"
    
    # Setup the chonky module to use our model engine
    import src.chunking.text_chunkers.chonky_chunker as chonky_module
    original_device = None
    original_engine = None
    original_engine_available = None
    
    # Store original values to restore later
    if hasattr(chonky_module, "DEFAULT_DEVICE"):
        original_device = chonky_module.DEFAULT_DEVICE
        chonky_module.DEFAULT_DEVICE = device
    
    if hasattr(chonky_module, "_MODEL_ENGINE"):
        original_engine = chonky_module._MODEL_ENGINE
    if hasattr(chonky_module, "_ENGINE_AVAILABLE"):
        original_engine_available = chonky_module._ENGINE_AVAILABLE
    
    # Set up engine
    engine = HaystackModelEngine(socket_path=socket_path)
    chonky_module._MODEL_ENGINE = engine
    chonky_module._ENGINE_AVAILABLE = True
    
    # Set the config to use GPU
    config = get_chunker_config('chonky')
    config['device'] = device
    config['model_engine'] = 'haystack'
    
    start_time = time.time()
    
    try:
        # Ensure engine is started
        engine.start()
        
        # Chunk the document with GPU settings
        chunks = chunk_text(
            content=document['content'],
            doc_id=document['id'],
            path=document.get('path', 'unknown'),
            doc_type=document.get('type', 'text'),
            model_id=model_id,
            output_format="json"
        )
        
        chunking_time = time.time() - start_time
        return chunks, chunking_time
    finally:
        # Restore original settings
        if original_device is not None:
            chonky_module.DEFAULT_DEVICE = original_device
        
        # Restore original engine settings
        if original_engine is not None:
            chonky_module._MODEL_ENGINE = original_engine
        if original_engine_available is not None:
            chonky_module._ENGINE_AVAILABLE = original_engine_available

def count_and_analyze_chunks(chunks):
    """Count and analyze chunks for statistical comparison.
    
    Args:
        chunks: List of chunks returned by the chunker
        
    Returns:
        dict: Analysis results including count and average size
    """
    if not chunks:
        return {
            "count": 0,
            "avg_size": 0,
            "samples": []
        }
    
    # Handle different types of chunks - they could be strings or dictionaries
    chunk_count = len(chunks)
    samples = []
    total_size = 0
    token_counts = []
    
    if isinstance(chunks[0], str):
        # String chunks
        for c in chunks:
            total_size += len(c)
            token_counts.append(len(c.split()))
        
        avg_size = total_size / chunk_count if chunk_count > 0 else 0
        avg_tokens = sum(token_counts) / chunk_count if chunk_count > 0 else 0
        samples = [c[:100] + "..." if len(c) > 100 else c for c in chunks[:3]]
    else:
        # Dictionary chunks - extract content field for calculations
        for c in chunks:
            if not isinstance(c, dict):
                continue
                
            content = c.get('content', '')
            total_size += len(content)
            token_counts.append(len(content.split()))
            
        avg_size = total_size / chunk_count if chunk_count > 0 else 0
        avg_tokens = sum(token_counts) / chunk_count if chunk_count > 0 else 0
        
        samples = [
            {
                "id": c.get('id', 'unknown'),
                "type": c.get('symbol_type', 'unknown'),
                "length": len(c.get('content', '')),
                "tokens": len(c.get('content', '').split()),
                "preview": c.get('content', '')[:100] + "..." if len(c.get('content', '')) > 100 else c.get('content', '')
            }
            for c in chunks[:3] if isinstance(c, dict)
        ]
    
    return {
        "count": chunk_count,
        "avg_size": avg_size,
        "avg_tokens": avg_tokens,
        "samples": samples
    }

def run_benchmark(model_server_process, model_engine, adapter, test_data_dir, output_dir):
    """Run the CPU vs GPU chunking benchmark.
    
    Args:
        model_server_process: The running model server process
        model_engine: The initialized model engine
        adapter: DoclingAdapter instance
        test_data_dir: Directory containing test PDFs
        output_dir: Directory to write output files
        
    Returns:
        dict: Benchmark results summary
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Find PDF files
    pdf_files = list(Path(test_data_dir).glob("*.pdf"))
    if not pdf_files:
        logger.error(f"No PDF files found in {test_data_dir}")
        return {}
    
    # Sort by name for consistent ordering
    pdf_files.sort()
    
    # Results structure
    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "system_info": {
            "device": DEFAULT_DEVICE,
            "model_id": DEFAULT_MODEL_ID
        },
        "documents": [],
        "summary": {
            "total_documents": len(pdf_files),
            "total_cpu_chunking_time": 0.0,
            "total_gpu_chunking_time": 0.0,
            "speedup": 0.0
        }
    }
    
    # Process each document
    for pdf_file in pdf_files:
        logger.info("=" * 80)
        logger.info(f"Benchmarking document: {pdf_file.name}")
        logger.info("=" * 80)
        
        # Process the document
        document, processing_time = process_pdf(adapter, str(pdf_file))
        
        # Run CPU chunking
        logger.info("-" * 40)
        logger.info("Running CPU-only chunking...")
        # Use 12 workers by default for parallel processing
        num_cpu_workers = 12
        cpu_chunks, cpu_time = chunk_document_cpu(document, DEFAULT_MODEL_ID, num_workers=num_cpu_workers)
        cpu_analysis = count_and_analyze_chunks(cpu_chunks)
        logger.info(f"CPU chunking completed in {cpu_time:.2f} seconds")
        logger.info(f"Generated {cpu_analysis['count']} chunks with average size of {cpu_analysis['avg_size']:.0f} characters and {cpu_analysis['avg_tokens']:.2f} tokens")
        
        # Run GPU chunking
        logger.info("-" * 40)
        logger.info("Running GPU-accelerated chunking...")
        gpu_chunks, gpu_time = chunk_document_gpu(
            document, 
            DEFAULT_MODEL_ID, 
            DEFAULT_SOCKET_PATH,
            DEFAULT_DEVICE
        )
        gpu_analysis = count_and_analyze_chunks(gpu_chunks)
        logger.info(f"GPU chunking completed in {gpu_time:.2f} seconds")
        logger.info(f"Generated {gpu_analysis['count']} chunks with average size of {gpu_analysis['avg_size']:.0f} characters and {gpu_analysis['avg_tokens']:.2f} tokens")
        
        # Calculate speedup
        speedup = cpu_time / gpu_time if gpu_time > 0 else 0.0
        logger.info("-" * 40)
        logger.info(f"Performance comparison:")
        logger.info(f"CPU time: {cpu_time:.2f}s, GPU time: {gpu_time:.2f}s")
        logger.info(f"Speedup: {speedup:.2f}x")
        
        # Check chunk count consistency
        chunk_count_diff = abs(cpu_analysis['count'] - gpu_analysis['count'])
        logger.info(f"Chunk count difference: {chunk_count_diff}")
        
        # Add to results
        document_result = {
            "filename": pdf_file.name,
            "id": document['id'],
            "content_length": len(document['content']),
            "processing_time": processing_time,
            "cpu_chunking": {
                "time": cpu_time,
                "chunk_count": cpu_analysis['count'],
                "avg_characters": cpu_analysis['avg_size'],
                "avg_tokens": cpu_analysis['avg_tokens']
            },
            "gpu_chunking": {
                "time": gpu_time,
                "chunk_count": gpu_analysis['count'],
                "avg_characters": gpu_analysis['avg_size'],
                "avg_tokens": gpu_analysis['avg_tokens']
            },
            "comparison": {
                "speedup": speedup,
                "chunk_count_diff": chunk_count_diff
            }
        }
        
        results["documents"].append(document_result)
        results["summary"]["total_cpu_chunking_time"] += cpu_time
        results["summary"]["total_gpu_chunking_time"] += gpu_time
    
    # Calculate overall speedup
    total_cpu_time = results["summary"]["total_cpu_chunking_time"]
    total_gpu_time = results["summary"]["total_gpu_chunking_time"]
    results["summary"]["speedup"] = total_cpu_time / total_gpu_time if total_gpu_time > 0 else 0.0
    
    # Log final summary
    logger.info("=" * 80)
    logger.info("Benchmark Summary")
    logger.info("=" * 80)
    logger.info(f"Total documents processed: {len(pdf_files)}")
    logger.info(f"Total CPU chunking time: {total_cpu_time:.2f} seconds")
    logger.info(f"Total GPU chunking time: {total_gpu_time:.2f} seconds")
    logger.info(f"Overall speedup: {results['summary']['speedup']:.2f}x")
    
    # Save results to file
    results_file = os.path.join(output_dir, f"chunking_benchmark_{time.strftime('%Y%m%d_%H%M%S')}.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Benchmark results saved to {results_file}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Benchmark CPU vs GPU chunking performance")
    parser.add_argument("--data-dir", default=DEFAULT_TEST_DATA, help="Directory containing test PDF files")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory to write output files")
    parser.add_argument("--socket-path", default=DEFAULT_SOCKET_PATH, help="Path for the model server socket")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="Model ID to use for chunking")
    parser.add_argument("--device", default=DEFAULT_DEVICE, help="GPU device to use")
    args = parser.parse_args()
    
    # Ensure directories exist
    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Start the model server
    model_server_process = start_model_server(args.socket_path, args.device)
    
    try:
        # Set up the model engine
        model_engine = setup_model_engine(args.socket_path, args.model_id, args.device)
        
        # Create the document adapter
        adapter = DoclingAdapter()
        
        # Run the benchmark
        results = run_benchmark(model_server_process, model_engine, adapter, args.data_dir, args.output_dir)
        
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
    finally:
        # Clean up
        logger.info("Stopping model server...")
        model_server_process.terminate()
        model_server_process.wait(timeout=5)
        
        # Remove socket
        if os.path.exists(args.socket_path):
            os.unlink(args.socket_path)

if __name__ == "__main__":
    main()

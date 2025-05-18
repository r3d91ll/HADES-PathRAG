#!/usr/bin/env python3
"""
Enhanced PDF processing test script with GPU-accelerated chunking.

This script tests the document processing and chunking pipeline with the
Haystack model engine properly initialized for GPU acceleration.
"""

import os
import sys
import time
import json
import logging
import argparse
import subprocess
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from src.docproc.adapters.docling_adapter import DoclingAdapter
from src.chunking.text_chunkers.chonky_chunker import chunk_text, chunk_document
from src.model_engine.engines.haystack import HaystackModelEngine
from src.model_engine.engines.haystack.runtime import ModelClient
from src.model_engine.engines.haystack.runtime.server import run_server

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("test_gpu_chunking")

# Constants
DEFAULT_TEST_DATA = "test-data"
DEFAULT_OUTPUT_DIR = "test-output"
DEFAULT_SOCKET_PATH = "/tmp/hades_model_mgr_test.sock"
DEFAULT_MODEL_ID = "mirth/chonky_modernbert_large_1"
DEFAULT_DEVICE = "cuda:0"

def start_model_server(socket_path):
    """Start the Haystack model server in a separate process.
    
    Returns:
        subprocess.Popen: The process running the server
    """
    logger.info(f"Starting Haystack model server at {socket_path}")
    
    # Remove existing socket if present
    if os.path.exists(socket_path):
        os.unlink(socket_path)
    
    # Set environment variables for the server
    env = os.environ.copy()
    env["HADES_MODEL_MGR_SOCKET"] = socket_path
    env["HADES_DEFAULT_DEVICE"] = DEFAULT_DEVICE
    
    # Start the server process
    server_script = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "src", "model_engine", "engines", "haystack", "runtime", "server.py"
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

def setup_model_engine(socket_path, model_id):
    """Set up the Haystack model engine and load the model.
    
    Args:
        socket_path: Path to the server socket
        model_id: ID of the model to load
        
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
    logger.info(f"Loading model {model_id}")
    try:
        result = engine.load_model(model_id, device=DEFAULT_DEVICE)
        logger.info(f"Model load result: {result}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise RuntimeError(f"Failed to load model: {e}")
    
    # Get loaded models
    models = engine.get_loaded_models()
    logger.info(f"Loaded models: {models}")
    
    return engine

def process_pdf(adapter, pdf_path, output_dir):
    """Process a PDF file using the Docling adapter.
    
    Args:
        adapter: DoclingAdapter instance
        pdf_path: Path to the PDF file
        output_dir: Directory to write output files
        
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
    
    # Log some metadata
    if 'metadata' in document:
        logger.info(f"Metadata: {json.dumps(document['metadata'], indent=2)}")
    
    return document

def chunk_document_with_gpu(document, model_id, output_dir):
    """Chunk a document using GPU-accelerated Chonky.
    
    Args:
        document: The document to chunk
        model_id: ID of the model to use
        output_dir: Directory to write output files
        
    Returns:
        dict: Chunked document
    """
    logger.info(f"Chunking document {document['id']} with model {model_id}")
    start_time = time.time()
    
    # Configure environment for chunking
    os.environ["HADES_RUNTIME_AUTOSTART"] = "1"
    
    # Chunk the document
    try:
        # First try the direct chunk_text method which returns chunks directly
        chunks = chunk_text(
            content=document['content'],
            doc_id=document['id'],
            path=document.get('path', 'unknown'),
            doc_type=document.get('type', 'text'),
            model_id=model_id,
            output_format="json"  # Use json output format for consistent structure
        )
        
        # Create a chunked document with the same structure as expected
        chunked_doc = {
            'id': document['id'],
            'content': document['content'],
            'path': document.get('path', 'unknown'),
            'type': document.get('type', 'text'),
            'chunks': chunks if isinstance(chunks, list) else []
        }
    except Exception as e:
        logger.warning(f"Error using chunk_text directly: {e}")
        # Try alternate method
        chunked_doc = chunk_document(
            document=document,
            max_tokens=2048,
            return_pydantic=False
        )
        chunks = chunked_doc.get('chunks', [])
    
    chunking_time = time.time() - start_time
    logger.info(f"Chunked document in {chunking_time:.2f} seconds")
    logger.info(f"Number of chunks: {len(chunks)}")
    
    if chunks:
        # Handle different types of chunks - they could be strings or dictionaries
        if isinstance(chunks, list) and len(chunks) > 0:
            if isinstance(chunks[0], str):
                # Handle string chunks
                avg_size = sum(len(c) for c in chunks) / len(chunks)
                logger.info(f"Average chunk size: {avg_size:.0f} characters")
                
                # Sample the first few chunks
                logger.info("\nSample chunks:\n")
                for i, chunk in enumerate(chunks[:3]):
                    logger.info(f"Chunk {i+1}:")
                    logger.info(f"  Content length: {len(chunk)} characters")
                    preview = chunk[:100] + "..." if len(chunk) > 100 else chunk
                    logger.info(f"  Content preview: {preview}")
            else:
                # Handle dictionary chunks
                avg_size = sum(len(c.get('content', '')) for c in chunks if isinstance(c, dict)) / len(chunks)
                logger.info(f"Average chunk size: {avg_size:.0f} characters")
                
                # Sample the first few chunks
                logger.info("\nSample chunks:\n")
                for i, chunk in enumerate(chunks[:3]):
                    if isinstance(chunk, dict):
                        logger.info(f"Chunk {i+1}:")
                        logger.info(f"  ID: {chunk.get('id', 'unknown')}")
                        logger.info(f"  Type: {chunk.get('symbol_type', 'unknown')}")
                        logger.info(f"  Content length: {len(chunk.get('content', ''))} characters")
                        preview = chunk.get('content', '')[:100] + "..." if len(chunk.get('content', '')) > 100 else chunk.get('content', '')
                        logger.info(f"  Content preview: {preview}")
                    else:
                        logger.info(f"Chunk {i+1}: {str(chunk)[:100]}...")
        else:
            logger.warning(f"Unexpected chunks format: {type(chunks)}")
            logger.info(f"First 500 characters of chunks: {str(chunks)[:500]}")
    
    # Save to file
    output_file = os.path.join(output_dir, f"gpu_{document['id']}_chunked.json")
    with open(output_file, 'w') as f:
        json.dump(chunked_doc, f, indent=2)
    logger.info(f"Wrote output to {output_file}")
    
    return chunked_doc

def run_test(model_server_process, model_engine, adapter, test_data_dir, output_dir):
    """Run the test with both single and batch processing.
    
    Args:
        model_server_process: The running model server process
        model_engine: The initialized model engine
        adapter: DoclingAdapter instance
        test_data_dir: Directory containing test PDFs
        output_dir: Directory to write output files
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Find PDF files
    pdf_files = list(Path(test_data_dir).glob("*.pdf"))
    if not pdf_files:
        logger.error(f"No PDF files found in {test_data_dir}")
        return
    
    # Sort by name for consistent ordering
    pdf_files.sort()
    
    # Test with first PDF file
    logger.info("=" * 80)
    logger.info(f"Testing single PDF processing with GPU chunking: {pdf_files[0].name}")
    logger.info("=" * 80)
    
    # Process the PDF
    document = process_pdf(adapter, str(pdf_files[0]), output_dir)
    
    # Chunk the document with GPU
    chunked_doc = chunk_document_with_gpu(document, DEFAULT_MODEL_ID, output_dir)
    
    # If there are more PDFs, test batch processing
    if len(pdf_files) > 1:
        logger.info("\n" + "=" * 80)
        logger.info(f"Testing batch PDF processing with GPU chunking: {len(pdf_files)} files")
        logger.info("=" * 80 + "\n")
        
        # Process each PDF
        documents = []
        batch_start_time = time.time()
        
        for pdf_file in pdf_files:
            logger.info(f"Processing {pdf_file.name}...")
            document = process_pdf(adapter, str(pdf_file), output_dir)
            documents.append(document)
        
        batch_processing_time = time.time() - batch_start_time
        logger.info(f"Processed {len(documents)} PDFs in {batch_processing_time:.2f} seconds")
        
        # Batch chunk the documents
        batch_chunk_start_time = time.time()
        chunked_docs = []
        for doc in documents:
            chunked_doc = chunk_document(
                document=doc,
                max_tokens=2048,
                return_pydantic=False
            )
            chunked_docs.append(chunked_doc)
        batch_chunking_time = time.time() - batch_chunk_start_time
        
        logger.info(f"Batch chunked {len(chunked_docs)} documents in {batch_chunking_time:.2f} seconds")
        
        # Get overall stats
        total_chunks = sum(len(doc.get('chunks', [])) for doc in chunked_docs)
        logger.info(f"Total chunks: {total_chunks}")
        logger.info(f"Average chunks per document: {total_chunks / len(chunked_docs):.1f}")
        
        # Write batch results
        batch_summary = {
            "processing_time": batch_processing_time,
            "chunking_time": batch_chunking_time,
            "total_documents": len(chunked_docs),
            "total_chunks": total_chunks,
            "documents": []
        }
        
        # Log stats for each document
        for i, doc in enumerate(chunked_docs):
            doc_id = doc.get('id', f"doc_{i}")
            chunks = doc.get('chunks', [])
            logger.info(f"\nDocument {i+1} ({pdf_files[i].name}):")
            logger.info(f"  Number of chunks: {len(chunks)}")
            
            if chunks:
                avg_size = sum(len(c.get('content', '')) for c in chunks) / len(chunks)
                logger.info(f"  Average chunk size: {avg_size:.0f} characters")
            
            # Save individual document
            output_file = os.path.join(output_dir, f"gpu_batch_{doc_id}_chunked.json")
            with open(output_file, 'w') as f:
                json.dump(doc, f, indent=2)
            logger.info(f"Wrote output to {output_file}")
            
            # Add to summary
            batch_summary["documents"].append({
                "id": doc_id,
                "file": str(pdf_files[i]),
                "chunk_count": len(chunks),
                "avg_chunk_size": avg_size if chunks else 0
            })
        
        # Save batch summary
        summary_file = os.path.join(output_dir, "gpu_batch_processing_summary.json")
        with open(summary_file, 'w') as f:
            json.dump(batch_summary, f, indent=2)
        logger.info(f"Wrote output to {summary_file}")
    
    logger.info("\n" + "=" * 80)
    logger.info("Summary")
    logger.info("=" * 80)
    logger.info(f"Successfully processed {len(pdf_files)} PDF files with GPU chunking")
    logger.info("Test completed successfully!")

def main():
    parser = argparse.ArgumentParser(description="Test PDF processing with GPU-accelerated chunking")
    parser.add_argument("--data-dir", default=DEFAULT_TEST_DATA, help="Directory containing test PDF files")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory to write output files")
    parser.add_argument("--socket-path", default=DEFAULT_SOCKET_PATH, help="Path for the model server socket")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="Model ID to use for chunking")
    args = parser.parse_args()
    
    # Ensure directories exist
    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Start the model server
    model_server_process = start_model_server(args.socket_path)
    
    try:
        # Set up the model engine
        model_engine = setup_model_engine(args.socket_path, args.model_id)
        
        # Create the document adapter
        adapter = DoclingAdapter()
        
        # Run the test
        run_test(model_server_process, model_engine, adapter, args.data_dir, args.output_dir)
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
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

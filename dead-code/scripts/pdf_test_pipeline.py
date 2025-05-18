#!/usr/bin/env python
"""
Simplified PDF Processing Pipeline for Testing

This script provides a simplified version of the PDF processing pipeline
that bypasses some of the issues in the current implementation. It's 
designed for testing and demonstration purposes only.
"""

# Fix import paths - add project root to Python path
import sys
from pathlib import Path

# Add the project root directory to Python path
project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))

import json
import logging
import os
import time
from typing import Dict, Any, List, Optional, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import chunking functions - these should work fine
from src.chunking import chunk_text

# Try to import embedding module if available
try:
    from src.embedding.processors import add_embeddings_to_document
    from src.embedding.adapters.cpu_adapter import CPUEmbeddingAdapter
    EMBEDDING_AVAILABLE = True
    logger.info("Embedding module available")
except ImportError:
    logger.warning("Embedding module not available - embeddings will be skipped")
    EMBEDDING_AVAILABLE = False

# PDF processing library
import pdfplumber


class SimplePDFPipeline:
    """
    A simplified pipeline for processing PDFs directly with pdfplumber.
    This bypasses the issues in the docproc module for testing purposes.
    """
    
    def __init__(
        self,
        output_dir: Optional[Union[str, Path]] = None,
        chunking_options: Optional[Dict[str, Any]] = None,
        embedding_options: Optional[Dict[str, Any]] = None,
        save_intermediate_results: bool = False
    ):
        """Initialize the pipeline."""
        # Set output directory
        if output_dir:
            self.output_dir = Path(output_dir) if isinstance(output_dir, str) else output_dir
            self.output_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.output_dir = None
            
        # Default chunking options
        self.chunking_options = chunking_options or {
            "max_tokens": 1024,
            "output_format": "json",
            "doc_type": "academic_pdf"
        }
        
        # Default embedding options
        self.embedding_options = embedding_options or {
            "model_name": "all-MiniLM-L6-v2"
        }
        
        # Whether to save intermediate results
        self.save_intermediate_results = save_intermediate_results
        
        # Setup embedding adapter if available
        self.embedding_adapter = None
        if EMBEDDING_AVAILABLE:
            try:
                logger.info(f"Initializing CPU embedding adapter with model: {self.embedding_options['model_name']}")
                self.embedding_adapter = CPUEmbeddingAdapter(model_name=self.embedding_options['model_name'])
                logger.info("CPU embedding adapter initialized")
            except Exception as e:
                logger.error(f"Failed to initialize embedding adapter: {e}")
    
    def process_document(self, pdf_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
        """Process a PDF document directly using pdfplumber.
        
        This bypasses the docproc module for testing.
        """
        path_obj = Path(pdf_path) if isinstance(pdf_path, str) else pdf_path
        
        if not path_obj.exists():
            logger.error(f"PDF file not found: {path_obj}")
            return None
        
        logger.info(f"STAGE 1: Document Processing - {path_obj.name}")
        start_time = time.time()
        
        try:
            # Extract text directly from PDF
            all_text = ""
            with pdfplumber.open(path_obj) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    text = page.extract_text() or ""
                    all_text += f"\n--- PAGE {page_num + 1} ---\n{text}\n"
            
            # Create a simple document structure
            doc_id = f"pdf_{path_obj.stem}_{os.urandom(4).hex()}"
            result = {
                "id": doc_id,
                "content": all_text,
                "path": str(path_obj.absolute()),
                "type": "academic_pdf",
                "metadata": {
                    "filename": path_obj.name,
                    "size_bytes": path_obj.stat().st_size,
                    "processing_time": time.time() - start_time
                }
            }
            
            processing_time = time.time() - start_time
            logger.info(f"Document processing completed in {processing_time:.2f} seconds")
            
            # Save intermediate result if requested
            if self.save_intermediate_results and self.output_dir:
                output_path = self.output_dir / f"docproc_{path_obj.stem}_output.json"
                with open(output_path, 'w') as f:
                    json.dump(result, f, indent=2)
                logger.info(f"Document processing output saved to: {output_path}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing {path_obj.name}: {e}")
            return None
    
    def chunk_document(self, doc_result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Chunk a processed document."""
        if not doc_result:
            logger.error("No document provided for chunking")
            return None
        
        logger.info(f"STAGE 2: Document Chunking - {doc_result.get('id', 'Unknown')}")
        start_time = time.time()
        
        try:
            # Get document properties
            doc_id = doc_result.get('id', f"pdf_{os.urandom(4).hex()}")
            doc_path = doc_result.get('path', 'unknown')
            
            # Prepare document for chunking
            document = {
                "id": doc_id,
                "path": doc_path,
                "content": doc_result.get("content", ""),
                "type": "academic_pdf"
            }
            
            # Process through chunking
            chunking_options = self.chunking_options.copy()
            chunking_options["doc_id"] = doc_id
            chunking_options["path"] = doc_path
            
            chunks = chunk_text(document, **chunking_options)
            
            # Wrap into result structure
            result = {
                "document": doc_result,
                "chunks": chunks,
                "chunk_count": len(chunks),
                "options": chunking_options
            }
            
            chunking_time = time.time() - start_time
            logger.info(f"Document chunking completed in {chunking_time:.2f} seconds, generated {len(chunks)} chunks")
            
            # Save intermediate result if requested
            if self.save_intermediate_results and self.output_dir:
                # Save chunks in separate file for easier analysis
                chunks_output_path = self.output_dir / f"chunks_{Path(doc_path).stem}_output.json"
                with open(chunks_output_path, 'w') as f:
                    json.dump({
                        "doc_id": doc_id,
                        "chunk_count": len(chunks),
                        "chunks": chunks
                    }, f, indent=2)
                logger.info(f"Chunking output saved to: {chunks_output_path}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error chunking document: {e}")
            return None
    
    async def add_embeddings(self, pipeline_result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Add embeddings to document chunks."""
        if not pipeline_result or "chunks" not in pipeline_result:
            logger.error("No valid chunks provided for embedding")
            return pipeline_result
        
        if not EMBEDDING_AVAILABLE:
            logger.warning("Embedding module not available - skipping embedding generation")
            return pipeline_result
        
        if not self.embedding_adapter:
            logger.warning("No embedding adapter configured - skipping embedding generation")
            return pipeline_result
        
        logger.info(f"STAGE 3: Adding Embeddings to {len(pipeline_result['chunks'])} chunks")
        start_time = time.time()
        
        try:
            # Get chunks from pipeline result
            chunks = pipeline_result["chunks"]
            
            # Extract text content from chunks for embedding
            texts = [chunk.get("text", "") for chunk in chunks]
            logger.info(f"Generating embeddings for {len(texts)} text chunks")
            
            # Generate embeddings directly using the adapter
            embeddings = self.embedding_adapter.embed(texts)
            logger.info(f"Generated {len(embeddings)} embeddings")
            
            # Add embeddings back to chunks
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                chunk["embedding"] = embedding
                chunk["_embedding_info"] = {
                    "model": self.embedding_options.get("model_name", "unknown"),
                    "dimensions": len(embedding) if isinstance(embedding, list) else 0,
                    "chunk_index": i
                }
            
            # Update the pipeline result with embedded chunks
            pipeline_result["chunks"] = chunks
            
            embedding_time = time.time() - start_time
            logger.info(f"Embedding completed in {embedding_time:.2f} seconds")
            
            return pipeline_result
            
        except Exception as e:
            logger.error(f"Error adding embeddings: {e}")
            # Return original pipeline result without embeddings
            return pipeline_result
    
    async def process_pdf(self, pdf_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
        """Process a PDF file through the complete pipeline."""
        logger.info(f"Starting complete pipeline for {pdf_path}")
        
        # Stage 1: Document Processing
        doc_result = self.process_document(pdf_path)
        if not doc_result:
            logger.error(f"Document processing failed for {pdf_path}")
            return None
        
        # Stage 2: Chunking
        pipeline_result = self.chunk_document(doc_result)
        if not pipeline_result:
            logger.error(f"Chunking failed for {pdf_path}")
            return None
        
        # Stage 3: Embedding (if available)
        if EMBEDDING_AVAILABLE and self.embedding_adapter:
            pipeline_result = await self.add_embeddings(pipeline_result)
        
        # Save complete pipeline result
        if self.output_dir:
            path_obj = Path(pdf_path) if isinstance(pdf_path, str) else pdf_path
            output_path = self.output_dir / f"complete_{path_obj.stem}_output.json"
            with open(output_path, 'w') as f:
                json.dump(pipeline_result, f, indent=2)
            logger.info(f"Complete pipeline output saved to: {output_path}")
        
        logger.info(f"Pipeline processing complete for {pdf_path}")
        return pipeline_result


async def run_pipeline(pdf_paths: List[Union[str, Path]], output_dir: Union[str, Path] = "./test-output"):
    """Run the pipeline on multiple PDF files."""
    # Initialize pipeline
    pipeline = SimplePDFPipeline(
        output_dir=output_dir,
        save_intermediate_results=True,
        chunking_options={
            "max_tokens": 1024,
            "output_format": "json",
            "doc_type": "academic_pdf"
        },
        embedding_options={
            "model_name": "all-MiniLM-L6-v2"
        }
    )
    
    results = []
    for pdf_path in pdf_paths:
        logger.info(f"Processing {pdf_path}...")
        result = await pipeline.process_pdf(pdf_path)
        if result:
            results.append({
                "path": str(pdf_path),
                "doc_id": result["document"].get("id", "unknown"),
                "chunks": len(result["chunks"]),
                "has_embeddings": any("embedding" in chunk for chunk in result["chunks"]),
                "success": True
            })
        else:
            results.append({
                "path": str(pdf_path),
                "success": False
            })
    
    # Save overall results
    summary_path = Path(output_dir) / "pipeline_summary.json"
    with open(summary_path, 'w') as f:
        json.dump({
            "processed": len(pdf_paths),
            "successful": sum(1 for r in results if r["success"]),
            "results": results
        }, f, indent=2)
    
    logger.info(f"Pipeline complete. Processed {len(pdf_paths)} files, {sum(1 for r in results if r['success'])} successful")
    logger.info(f"Results saved to {output_dir}")
    
    return results


if __name__ == "__main__":
    import argparse
    import asyncio
    
    parser = argparse.ArgumentParser(description="Simplified PDF Processing Pipeline for HADES-PathRAG")
    parser.add_argument("--files", "-f", nargs="+", help="Input PDF files to process")
    parser.add_argument("--output", "-o", default="./test-output", help="Output directory")
    args = parser.parse_args()
    
    # Use provided files or default to test-data PDFs
    if args.files:
        paths = [Path(f) for f in args.files]
    else:
        # Use the PDF files from test-data
        test_data_dir = Path(__file__).parent.parent / "test-data"
        paths = list(test_data_dir.glob("*.pdf"))
    
    if not paths:
        logger.error("No PDF files found to process")
        sys.exit(1)
    
    logger.info(f"Found {len(paths)} PDF files to process")
    asyncio.run(run_pipeline(paths, args.output))

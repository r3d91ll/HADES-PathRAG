"""Text Processing Transformation Pipeline Prototype.

This module provides a prototype implementation of the document transformation pipeline,
focusing on the integration of document processing, chunking, and embedding components.
It serves as a reference implementation for future refactoring of the main ingestor.

The pipeline follows these transformation stages:
1. Document Processing: Transform source document into normalized markdown text
2. Chunking: Transform normalized text into semantic chunks
3. Embedding: Enrich chunks with vector embeddings
4. Storage: Persist transformed representation to the database (future)

Usage:
    from src.pipelines.ingest.orchestrator.pdf_pipeline_prototype import PDFPipeline
    processor = PDFPipeline()
    result = await processor.process_pdf('/path/to/document.pdf')
"""

import json
import logging
import sys
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple, cast

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import document processing functions
from src.docproc.core import process_document

# Import chunking functions
from src.chunking.text_chunkers.chonky_chunker import chunk_text

# Try to import embedding module if available
try:
    from src.embedding.processors import add_embeddings_to_document
    from src.embedding.base import get_adapter, EmbeddingAdapter
    from src.config.embedding_config import get_adapter_config, load_config
    EMBEDDING_AVAILABLE = True
except ImportError:
    logger.warning("Embedding module not available or incomplete - embeddings will be skipped")
    EMBEDDING_AVAILABLE = False


class PDFPipeline:
    """Pipeline for processing documents through the HADES-PathRAG system.
    
    This class implements a complete transformation pipeline that:
    1. Processes documents using the docproc module (normalizes to markdown)
    2. Transforms document into semantic chunks using the chunking module
    3. Enriches chunks with vector embeddings (if embedding module available)
    4. Can save the results at each stage for analysis
    
    The pipeline is designed as a true transformation pipeline, where each stage
    transforms the document representation rather than simply adding to it.
    """
    
    def __init__(
            self,
            output_dir: Optional[Union[str, Path]] = None,
            chunking_options: Optional[Dict[str, Any]] = None,
            embedding_options: Optional[Dict[str, Any]] = None,
            save_intermediate_results: bool = False
        ):
        """Initialize the transformation pipeline.
        
        Args:
            output_dir: Directory to save output files (None = don't save)
            chunking_options: Options for the chunking stage
            embedding_options: Options for the embedding stage
            save_intermediate_results: Whether to save results from each stage
        """
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
        
        # Load embedding configuration from config system
        if EMBEDDING_AVAILABLE:
            try:
                embedding_config = load_config()
                default_adapter = embedding_config.get("default_adapter", "modernbert")
                # Get default adapter configuration
                default_options = get_adapter_config(default_adapter)
                
                # If user provided options, merge them with defaults
                if embedding_options:
                    self.embedding_options = dict(default_options)
                    self.embedding_options.update(embedding_options)
                else:
                    self.embedding_options = default_options
                    
                logger.info(f"Using embedding adapter: {self.embedding_options.get('adapter_name', default_adapter)} "
                           f"on device: {self.embedding_options.get('device', 'cpu')}")
            except Exception as e:
                logger.warning(f"Error loading embedding configuration: {e}")
                # Fallback to defaults if config loading fails
                self.embedding_options = embedding_options or {
                    "adapter_name": "modernbert",
                    "model_name": "answerdotai/ModernBERT-base",
                    "max_length": 8192,
                    "pooling_strategy": "cls",
                    "batch_size": 8,
                    "device": "cpu",
                    "normalize_embeddings": True
                }
        else:
            # Fallback if embedding module isn't available
            self.embedding_options = embedding_options or {}
            logger.warning("Embedding module not available - using default options")
        
        
        # Whether to save intermediate results
        self.save_intermediate_results = save_intermediate_results
        
        # Setup embedding adapter if available
        self.embedding_adapter = None
        if EMBEDDING_AVAILABLE:
            try:
                adapter_name = self.embedding_options.get("adapter_name", "cpu")
                adapter_options = {
                    k: v for k, v in self.embedding_options.items()
                    if k not in ["adapter_name"]
                }
                self.embedding_adapter = get_adapter(adapter_name, **adapter_options)
                logger.info(f"Initialized embedding adapter: {adapter_name}")
            except Exception as e:
                logger.error(f"Failed to initialize embedding adapter: {e}")
    
    def process_document(self, pdf_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
        """Process a document through the document processing module.
        
        This stage transforms the source document (PDF, etc.) into a normalized
        text format (markdown) with extracted metadata.
        
        Args:
            pdf_path: Path to the document file
            
        Returns:
            Normalized document dictionary or None if processing failed
        """
        path_obj = Path(pdf_path) if isinstance(pdf_path, str) else pdf_path
        
        logger.info(f"STAGE 1: Document Processing - {path_obj.name}")
        start_time = time.time()
        
        try:
            # Process document using docproc
            doc_result = process_document(path_obj)
            
            # Save intermediate result if requested
            if self.save_intermediate_results and self.output_dir:
                output_path = self.output_dir / f"docproc_{path_obj.stem}_output.json"
                with open(output_path, 'w') as f:
                    json.dump(doc_result, f, indent=2)
                logger.info(f"Document processing output saved to: {output_path}")
            
            processing_time = time.time() - start_time
            logger.info(f"Document processing completed in {processing_time:.2f} seconds")
            
            return doc_result
            
        except Exception as e:
            logger.error(f"Error processing {path_obj.name}: {e}")
            return None
    
    def chunk_document(self, doc_result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Transform a processed document into chunks.
        
        This method transforms the document representation by splitting the content
        into semantic chunks and reorganizing the data structure to be chunk-centric.
        The original document content is removed to eliminate redundancy, while preserving
        all document metadata.
        
        Args:
            doc_result: Document processing result from process_document()
            
        Returns:
            Transformed document with chunks, or None if chunking failed
        """
        if not doc_result:
            logger.error("No document provided for chunking")
            return None
        
        logger.info(f"STAGE 2: Document Chunking - {doc_result.get('id', 'Unknown')}")
        start_time = time.time()
        
        try:
            # Get document properties
            doc_id = doc_result.get('id', f"doc_{os.urandom(4).hex()}")
            doc_path = doc_result.get('source', 'unknown')
            
            # Setup chunking options
            options = self.chunking_options.copy()
            
            # Set document-specific options
            options["doc_id"] = doc_id
            options["path"] = doc_path
            
            # Prepare for chunking - extract content string
            doc_content = doc_result.get("content", "")
            
            # Process through chunking - note that chunk_text expects content as string, not dict
            chunks_result = chunk_text(
                content=doc_content,
                doc_id=doc_id,
                path=doc_path,
                doc_type=options.get("doc_type", "academic_pdf"),
                max_tokens=options.get("max_tokens", 1024),
                output_format=options.get("output_format", "json")
            )
            
            # Extract chunks from the chunking result
            if isinstance(chunks_result, dict) and "chunks" in chunks_result:
                chunks = chunks_result.get("chunks", [])
            else:
                # Direct list of chunks
                chunks = chunks_result
                
            # Transfer metadata from the original document
            metadata = doc_result.get("metadata", {})
            
            # Create transformed result structure (removing redundant document content)
            result = {
                "id": doc_id,
                "metadata": metadata,
                "source": doc_path,
                "format": doc_result.get("format", ""),
                "content_type": doc_result.get("content_type", "text"),
                "chunks": chunks,
                "chunk_count": len(chunks),
                "processing_metadata": {
                    "chunking_options": options,
                    "processing_time": time.time() - start_time,
                    "timestamp": datetime.now().isoformat()
                }
            }
            
            chunking_time = time.time() - start_time
            logger.info(f"Document chunking completed in {chunking_time:.2f} seconds, generated {len(chunks)} chunks")
            
            # Save intermediate result if requested
            if self.save_intermediate_results and self.output_dir:
                # Save full pipeline output
                pipeline_output_path = self.output_dir / f"pipeline_{Path(doc_path).stem}_output.json"
                with open(pipeline_output_path, 'w') as f:
                    json.dump(result, f, indent=2)
                logger.info(f"Pipeline output saved to: {pipeline_output_path}")
                
                # Save chunks in separate file for easier analysis
                chunks_output_path = self.output_dir / f"chunks_{Path(doc_path).stem}_output.json"
                with open(chunks_output_path, 'w') as f:
                    json.dump({
                        "doc_id": doc_id,
                        "chunk_count": len(chunks),
                        "chunks": chunks,
                        "chunking_options": result["processing_metadata"]["chunking_options"]
                    }, f, indent=2)
                logger.info(f"Chunking output saved to: {chunks_output_path}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error chunking document: {e}")
            return None
    
    async def add_embeddings(self, pipeline_result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Add embeddings to document chunks.
        
        This continues the document transformation by adding embeddings to each chunk,
        further enriching the document representation for downstream tasks.
        
        ModernBERT adapter is used by default to generate embeddings with support for longer
        contexts (up to 8K tokens), making it suitable for academic papers with complex
        semantic structure. 
        
        Args:
            pipeline_result: Transformed document from chunk_document()
            
        Returns:
            Further transformed document with embeddings, or original if embedding failed
        """
        if not pipeline_result or "chunks" not in pipeline_result:
            logger.error("No valid chunks provided for embedding")
            return pipeline_result
        
        if not EMBEDDING_AVAILABLE:
            logger.warning("Embedding module not available - skipping embedding generation")
            return pipeline_result
        
        if not self.embedding_adapter:
            logger.warning("No embedding adapter configured - skipping embedding generation")
            return pipeline_result
        
        doc_id = pipeline_result.get('id', 'Unknown')
        logger.info(f"STAGE 3: Embedding Generation - {doc_id}")
        start_time = time.time()
        
        try:
            # Extract chunks from the pipeline result
            chunks = pipeline_result.get("chunks", [])
            num_chunks = len(chunks)
            
            if num_chunks == 0:
                logger.warning(f"Document {doc_id} has no chunks to embed")
                return pipeline_result
                
            # Extract text content for embedding
            chunk_texts: List[str] = [chunk.get("content", "") for chunk in chunks]
            
            # Log stats about the chunks being processed
            total_tokens = sum(len(text.split()) for text in chunk_texts)
            avg_tokens = total_tokens / num_chunks if num_chunks > 0 else 0
            logger.info(f"Generating embeddings for {num_chunks} chunks with avg {avg_tokens:.1f} tokens per chunk")
            
            # Generate embeddings using the configured adapter (ModernBERT by default)
            adapter_name = self.embedding_options.get("adapter_name", "modernbert")
            pooling = self.embedding_options.get("pooling_strategy", "cls")
            logger.info(f"Using {adapter_name} adapter with {pooling} pooling strategy")
            
            # Use embed method directly from our adapter for type safety
            embeddings = await self.embedding_adapter.embed(chunk_texts)
            
            # Validate that we received the expected number of embeddings
            if len(embeddings) != num_chunks:
                logger.warning(f"Expected {num_chunks} embeddings but received {len(embeddings)}")
            
            # Update chunks with embeddings, ensuring proper conversion
            embedding_dim = None
            for i, embedding in enumerate(embeddings):
                if i < len(chunks):
                    # Handle different embedding types (numpy arrays, lists, etc.)
                    if hasattr(embedding, "tolist"):
                        embedding_list = embedding.tolist()
                    else:
                        embedding_list = list(embedding)
                    
                    # Store the first dimension for metadata
                    if embedding_dim is None and embedding_list:
                        embedding_dim = len(embedding_list)
                    
                    # Add embedding to the chunk
                    chunks[i]["embedding"] = embedding_list
            
            # Update the pipeline result with the enhanced chunks
            pipeline_result["chunks"] = chunks
            
            # Add embedding metadata to the processing metadata
            if "processing_metadata" not in pipeline_result:
                pipeline_result["processing_metadata"] = {}
                
            # Add detailed embedding metadata
            pipeline_result["processing_metadata"]["embedding"] = {
                "adapter": adapter_name,
                "model": self.embedding_options.get("model_name", "answerdotai/ModernBERT-base"),
                "pooling_strategy": pooling,
                "embedding_dimensions": embedding_dim,
                "chunks_embedded": len(embeddings),
                "timestamp": time.time(),
                "processing_time_sec": time.time() - start_time
            }
            
            embedding_time = time.time() - start_time
            logger.info(f"Embedding generation completed in {embedding_time:.2f} seconds")
            
            # Save intermediate result if requested
            if self.save_intermediate_results and self.output_dir:
                doc_id = pipeline_result.get('id', 'unknown')
                doc_path = pipeline_result.get('source', 'unknown')
                
                # Determine filename from document path
                if isinstance(doc_path, str):
                    filename = Path(doc_path).stem
                else:  # Handle Path objects
                    filename = Path(str(doc_path)).stem
                
                complete_output_path = self.output_dir / f"complete_{filename}_output.json"
                with open(complete_output_path, 'w') as f:
                    json.dump(pipeline_result, f, indent=2)
                logger.info(f"Complete pipeline output saved to: {complete_output_path}")
            
            return pipeline_result
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            return pipeline_result
    
    async def process_pdf(self, pdf_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
        """Process a document through the complete transformation pipeline.
        
        This method orchestrates the complete document transformation pipeline:
        1. Document Processing: Converts the document to normalized text
        2. Chunking: Transforms the document into chunks
        3. Embedding: Enriches chunks with vector embeddings
        
        Args:
            pdf_path: Path to the document file
            
        Returns:
            Transformed document with chunks and embeddings
        """
        path_obj = Path(pdf_path) if isinstance(pdf_path, str) else pdf_path
        logger.info(f"Starting complete pipeline for {path_obj}")
        
        # Stage 1: Document Processing - Convert to normalized text format
        doc_result = self.process_document(path_obj)
        if doc_result is None:
            logger.error(f"Document processing failed for {path_obj}")
            return None
        
        # Stage 2: Chunking - Transform into chunk-centric representation
        transformed_doc = self.chunk_document(doc_result)
        if transformed_doc is None:
            logger.error(f"Chunking failed for {path_obj}")
            return None
        
        # Stage 3: Embedding - Enrich chunks with vector embeddings
        final_doc = await self.add_embeddings(transformed_doc)
        
        # Save final output if requested
        if self.save_intermediate_results and self.output_dir:
            doc_id = final_doc.get('id', 'unknown')
            doc_path = final_doc.get('source', 'unknown')
            
            # Determine filename from document path
            if isinstance(doc_path, str):
                filename = Path(doc_path).stem
            else:  # Handle Path objects
                filename = Path(str(doc_path)).stem
            
            complete_output_path = self.output_dir / f"complete_{filename}_output.json"
            with open(complete_output_path, 'w') as f:
                json.dump(final_doc, f, indent=2)
            logger.info(f"Complete pipeline output saved to: {complete_output_path}")
        
        logger.info(f"Pipeline processing complete for {path_obj}")
        return final_doc


# Add a main function to run the pipeline
async def run_pipeline(pdf_paths: List[Union[str, Path]], output_dir: Union[str, Path] = "./test-output"):
    """Run the pipeline on multiple PDF files.
    
    Args:
        pdf_paths: List of paths to PDF files
        output_dir: Directory to save output files
    """
    # Initialize pipeline
    pipeline = PDFPipeline(
        output_dir=output_dir,
        save_intermediate_results=True
    )
    
    # Initialize result counters
    processed_count = 0
    success_count = 0
    
    # Process each PDF
    results = []
    for path in pdf_paths:
        try:
            result = await pipeline.process_pdf(path)
            results.append({
                "path": str(path),
                "success": result is not None,
                "chunks": len(result["chunks"]) if result and "chunks" in result else 0,
                "has_embeddings": any("embedding" in chunk for chunk in result.get("chunks", []))
                if result and "chunks" in result else False
            })
            
            processed_count += 1
            if result is not None:
                success_count += 1
                
        except Exception as e:
            logger.error(f"Error processing {path}: {e}")
            results.append({
                "path": str(path),
                "success": False,
                "error": str(e)
            })
            
    # Save summary
    summary = {
        "processed": processed_count,
        "successful": success_count,
        "timestamp": datetime.now().isoformat(),
        "results": results
    }
    
    # Save summary to file
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    with open(output_dir_path / "pipeline_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
        
    logger.info(f"Pipeline complete. Processed {processed_count} files, {success_count} successful")
    logger.info(f"Results saved to {output_dir}")
    
    return results


# Command-line entry point
if __name__ == "__main__":
    import argparse
    import asyncio
    
    parser = argparse.ArgumentParser(description="Document Processing Pipeline for HADES-PathRAG")
    parser.add_argument("--files", "-f", nargs="+", help="Input PDF files to process")
    parser.add_argument("--output", "-o", default="./test-output", help="Output directory")
    args = parser.parse_args()
    
    # Use provided files or default to test-data PDFs
    if args.files:
        paths = [Path(f) for f in args.files]
    else:
        # Use the PDF files from test-data
        test_data_dir = Path(__file__).parent.parent.parent.parent.parent / "test-data"
        paths = list(test_data_dir.glob("*.pdf"))
    
    if not paths:
        logger.error("No PDF files found to process")
        sys.exit(1)
    
    logger.info(f"Found {len(paths)} PDF files to process")
    asyncio.run(run_pipeline(paths, args.output))

#!/usr/bin/env python3
"""
Mini-pipeline test for document processing and chunking.

This script processes documents from the test-data directory through the
document processing and chunking modules, then outputs the JSON objects
that would be passed to the embedding module.
"""

import argparse
import json
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

# Add project root to path so we can import from src
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import necessary modules
from src.schema.document_schema import DocumentSchema
from src.schema.validation import validate_document, ValidationStage, ValidationResult
from src.chunking import chunk_text

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Mock the HaystackModelEngine for chunking tests
class MockHaystackModelEngine:
    """Mock for HaystackModelEngine to avoid actual model loading during tests."""
    
    def __init__(self) -> None:
        self.started: bool = False
        self.loaded_models: Dict[str, bool] = {}
    
    def start(self) -> Dict[str, str]:
        self.started = True
        return {"status": "ok"}
    
    def status(self) -> Dict[str, bool]:
        return {"running": self.started}
    
    def load_model(self, model_id: str) -> Dict[str, Any]:
        """Mock implementation of load_model method."""
        self.loaded_models[model_id] = True
        return {"success": True, "model_id": model_id}

# Setup mock
import src.chunking.text_chunkers.chonky_chunker as chonky_chunker
chonky_chunker._MODEL_ENGINE = MockHaystackModelEngine()

# Mock splitter for chunking
class MockParagraphSplitter:
    """Mock ParagraphSplitter for testing."""
    
    def split_into_paragraphs(self, text):
        """Split text into mock paragraphs."""
        # Simple split by double newlines for testing
        paragraphs = text.split("\n\n")
        
        # Create mock ParagraphInfo objects
        result = []
        current_pos = 0
        for i, para in enumerate(paragraphs):
            if para.strip():  # Skip empty paragraphs
                start_idx = text.find(para, current_pos)
                end_idx = start_idx + len(para)
                current_pos = end_idx
                
                # Create a dict matching expected structure
                result.append({
                    "text": para,
                    "start_idx": start_idx,
                    "end_idx": end_idx,
                    "type": "paragraph",
                    "score": 0.95,
                    "metadata": {"paragraph_id": i}
                })
        
        return result

# Patch the get_splitter function
original_get_splitter = chonky_chunker._get_splitter_with_engine

def mock_get_splitter(*args, **kwargs):
    """Mock implementation that returns our MockParagraphSplitter."""
    return MockParagraphSplitter()

chonky_chunker._get_splitter_with_engine = mock_get_splitter


def process_pdf_file(pdf_file: Path, output_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Process a PDF file through document processing and chunking.
    
    Args:
        pdf_file: Path to PDF file
        output_dir: Optional directory to save output
        
    Returns:
        Processed and chunked document
    """
    logger.info(f"Processing PDF file: {pdf_file}")
    
    # Skip if file doesn't exist
    if not pdf_file.exists():
        logger.error(f"PDF file not found: {pdf_file}")
        return {}
    
    try:
        # For PDFs, we'll read the content and create a document directly
        # since DoclingAdapter may have issues with binary PDFs
        with open(pdf_file, 'rb') as f:
            # Try to extract text directly (this is a simplified approach)
            try:
                content = f.read().decode('utf-8', errors='ignore')
                logger.info(f"Read {len(content)} characters from {pdf_file}")
            except UnicodeDecodeError:
                logger.warning(f"Could not decode {pdf_file} as text. Using placeholder content.")
                # Use filename as placeholder content
                content = f"Content of {pdf_file.name}\n\nThis is placeholder content for testing."
        
        # Create a document for chunking
        doc_id = f"pdf_{pdf_file.stem}"
        document = {
            "id": doc_id,
            "path": str(pdf_file),
            "content": content,
            "type": "pdf"
        }
        
        # Process the document with the text chunking module
        chunks = chunk_text(document, max_tokens=1024, output_format="python")
        logger.info(f"Generated {len(chunks)} chunks from {pdf_file}")
        
        # Create a document schema with chunks
        schema_doc = {
            "id": doc_id,
            "title": pdf_file.stem,
            "content": content,
            "source": str(pdf_file),
            "document_type": "pdf",
            "metadata": {
                "format": "pdf",
                "language": "en",
                "creation_date": "2025-05-10",
                "author": "HADES-PathRAG Team",
            },
            "chunks": []
        }
        
        # Convert chunker output to ChunkMetadata format
        for idx, chunk in enumerate(chunks):
            chunk_content = chunk.get("content", "")
            chunk_metadata = {
                "start_offset": 0,  # PDF doesn't have reliable offsets
                "end_offset": len(chunk_content),
                "chunk_type": "text",
                "chunk_index": idx,
                "parent_id": doc_id,
                "metadata": {
                    "content": chunk_content,
                    "symbol_type": chunk.get("symbol_type", "paragraph"),
                    "name": chunk.get("name", f"chunk_{idx}"),
                    "token_count": chunk.get("token_count", 0),
                    "page": chunk.get("page", 0)  # Add page number if available
                }
            }
            schema_doc["chunks"].append(chunk_metadata)
        
        # Validate using schema validation
        validation_result = validate_document(schema_doc, ValidationStage.INGESTION)
        
        if not validation_result.is_valid:
            logger.warning(f"Document validation failed: {validation_result.errors}")
        else:
            logger.info("Document validation successful")
        
        # Save schema document with chunks if output_dir is provided
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"{pdf_file.stem}_chunked.json"
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(schema_doc, f, indent=2)
            logger.info(f"Saved chunked document to {output_path}")
        
        return schema_doc
    
    except Exception as e:
        logger.error(f"Error processing {pdf_file}: {e}")
        return {}


def main():
    """Run the mini-pipeline test."""
    parser = argparse.ArgumentParser(description="Run document processing and chunking pipeline")
    parser.add_argument("--output", "-o", default="./test-output", help="Output directory for chunked documents")
    parser.add_argument("--files", "-f", nargs="+", help="Specific files to process (optional)")
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Use provided files or search for PDFs in test-data
    if args.files:
        pdf_files = [Path(file) for file in args.files]
    else:
        data_dir = Path(__file__).parent.parent / "test-data"
        pdf_files = list(data_dir.glob("*.pdf"))
    
    logger.info(f"Found {len(pdf_files)} PDF files to process")
    
    # Process each PDF file
    for pdf_file in pdf_files:
        schema_doc = process_pdf_file(pdf_file, output_dir)
        
        if schema_doc:
            print(f"\nSuccessfully processed {pdf_file.name}")
            print(f"Document ID: {schema_doc.get('id')}")
            print(f"Number of chunks: {len(schema_doc.get('chunks', []))}")
            
            # Print sample of first chunk if available
            if schema_doc.get("chunks"):
                first_chunk = schema_doc["chunks"][0]
                content = first_chunk["metadata"]["content"]
                print("\nSample first chunk:")
                print(f"Chunk index: {first_chunk['chunk_index']}")
                preview = content[:100] + "..." if len(content) > 100 else content
                print(f"Content: {preview}")
    
    print(f"\nAll processing complete. Results saved to {output_dir}")


if __name__ == "__main__":
    main()

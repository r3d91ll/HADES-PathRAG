#!/usr/bin/env python3
"""
Test driver for the document processing and chunking pipeline.

This script processes PDF files through the document processing and chunking
modules to generate the exact JSON object that will be passed to the embedding module.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, List

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import the necessary modules
from src.docproc.core import process_document
from src.chunking import chunk_text


def process_and_chunk_document(file_path: str) -> Dict[str, Any]:
    """Process a document through docproc and chunking modules.
    
    Args:
        file_path: Path to document
        
    Returns:
        Chunked document ready for embedding
    """
    print(f"\n{'='*80}\nProcessing: {file_path}\n{'='*80}")
    
    # Step 1: Process through docproc
    print("\nStep 1: Document Processing")
    processed_doc = process_document(file_path)
    print(f"Processed document with ID: {processed_doc.get('id', 'unknown')}")
    
    # Step 2: Prepare document for chunking
    doc_id = processed_doc.get("id", f"doc_{Path(file_path).stem}")
    document = {
        "id": doc_id,
        "path": file_path,
        "content": processed_doc.get("content", ""),
        "type": processed_doc.get("type", Path(file_path).suffix[1:])
    }
    
    # Step 3: Chunk the document
    print("\nStep 2: Document Chunking")
    chunked_doc = chunk_text(document, max_tokens=1024, output_format="python")
    
    print(f"Generated {len(chunked_doc)} chunks")
    
    # Step 4: Create document with schema structure
    schema_doc = {
        "id": doc_id,
        "title": processed_doc.get("title", Path(file_path).stem),
        "content": processed_doc.get("content", ""),
        "source": file_path,
        "document_type": processed_doc.get("type", Path(file_path).suffix[1:]),
        "metadata": processed_doc.get("metadata", {}),
        "chunks": []
    }
    
    # Convert chunker output to ChunkMetadata format
    for idx, chunk in enumerate(chunked_doc):
        chunk_content = chunk.get("content", "")
        chunk_metadata = {
            "start_offset": chunk.get("start_offset", 0),
            "end_offset": chunk.get("end_offset", len(chunk_content)),
            "chunk_type": chunk.get("chunk_type", "text"),
            "chunk_index": idx,
            "parent_id": doc_id,
            "metadata": {
                "content": chunk_content,
                "symbol_type": chunk.get("symbol_type", "paragraph"),
                "name": chunk.get("name", f"chunk_{idx}"),
                "token_count": chunk.get("token_count", 0)
            }
        }
        schema_doc["chunks"].append(chunk_metadata)
    
    print("\nStep 3: Ready for Embedding")
    print(f"Document with {len(schema_doc['chunks'])} chunks ready for embedding")
    
    return schema_doc


def main():
    """Main function to run the test pipeline."""
    parser = argparse.ArgumentParser(description="Test document processing pipeline")
    parser.add_argument("files", nargs="+", help="Files to process")
    parser.add_argument("--output", "-o", help="Output directory for JSON files")
    args = parser.parse_args()
    
    # Process each file
    for file_path in args.files:
        try:
            # Process and chunk the document
            schema_doc = process_and_chunk_document(file_path)
            
            # Save the output if requested
            if args.output:
                output_dir = Path(args.output)
                output_dir.mkdir(parents=True, exist_ok=True)
                
                output_file = output_dir / f"{Path(file_path).stem}_chunked.json"
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(schema_doc, f, indent=2)
                
                print(f"\nSaved output to: {output_file}")
            
            # Print sample of the document (first chunk)
            if schema_doc["chunks"]:
                print("\nSample chunk:")
                first_chunk = schema_doc["chunks"][0]
                print(f"  Index: {first_chunk['chunk_index']}")
                print(f"  Type: {first_chunk['chunk_type']}")
                content = first_chunk["metadata"]["content"]
                print(f"  Content (first 100 chars): {content[:100]}...")
                
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()

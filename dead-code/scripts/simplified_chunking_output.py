#!/usr/bin/env python3
"""
Simplified script to generate a sample chunked document for embedding module development.

This script processes a text file through the chunking module and outputs the
exact JSON structure that would be passed to the embedding module.
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, List

# Add project root to path so we can import from src
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import necessary modules
from src.chunking import chunk_text
from src.schema.validation import validate_document, ValidationStage

# Setup output directory
OUTPUT_DIR = Path("./test-output/embedding-input")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def create_sample_document(content: str, doc_id: str = "sample_doc") -> Dict[str, Any]:
    """Create a sample document for chunking.
    
    Args:
        content: Document content
        doc_id: Document ID
        
    Returns:
        Document dictionary ready for chunking
    """
    return {
        "id": doc_id,
        "path": f"{doc_id}.txt",
        "content": content,
        "type": "text"
    }

def process_sample_document():
    """Process a sample document and output the chunked result."""
    # Use the sample text file from test-data
    sample_file = Path("test-data/sample_text.txt")
    if not sample_file.exists():
        print(f"Sample text file not found: {sample_file}")
        return
    
    # Read the content
    with open(sample_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Create a document for chunking
    doc_id = "sample_text_doc"
    document = create_sample_document(content, doc_id)
    
    # Process with chunking module
    print(f"Processing sample document: {doc_id}")
    chunks = chunk_text(document, max_tokens=1024, output_format="python")
    print(f"Generated {len(chunks)} chunks")
    
    # Create a document schema with chunks
    schema_doc = {
        "id": doc_id,
        "title": "Sample Text Document",
        "content": content,
        "source": str(sample_file),
        "document_type": "text",
        "metadata": {
            "format": "text",
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
            "start_offset": chunk.get("start_offset", 0),
            "end_offset": chunk.get("end_offset", len(chunk_content)),
            "chunk_type": "text",
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
    
    # Validate using schema validation
    validation_result = validate_document(schema_doc, ValidationStage.INGESTION)
    if not validation_result.is_valid:
        print(f"Document validation failed: {validation_result.errors}")
    else:
        print("Document validation successful")
    
    # Save schema document with chunks
    output_path = OUTPUT_DIR / "sample_chunked_document.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(schema_doc, f, indent=2)
    
    print(f"Saved chunked document to {output_path}")
    
    # Also create a sample Python file output
    output_py_path = OUTPUT_DIR / "sample_chunked_document.py"
    with open(output_py_path, 'w', encoding='utf-8') as f:
        f.write("#!/usr/bin/env python3\n")
        f.write("\"\"\"Sample chunked document for embedding module development.\"\"\"\n\n")
        f.write("# This is the exact JSON structure passed from chunking to embedding\n")
        f.write("SAMPLE_CHUNKED_DOCUMENT = ")
        f.write(json.dumps(schema_doc, indent=4))
        f.write("\n")
    
    print(f"Saved Python sample to {output_py_path}")
    
    # Print sample chunk
    if schema_doc["chunks"]:
        first_chunk = schema_doc["chunks"][0]
        print("\nSample first chunk:")
        print(f"Chunk index: {first_chunk['chunk_index']}")
        content = first_chunk["metadata"]["content"]
        preview = content[:100] + "..." if len(content) > 100 else content
        print(f"Content: {preview}")

if __name__ == "__main__":
    process_sample_document()

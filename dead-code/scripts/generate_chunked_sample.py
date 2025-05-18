#!/usr/bin/env python3
"""
Generate a sample chunked document for embedding module development.

This script processes a text file into chunks using the chunking module
and outputs a JSON file with the exact structure that would be passed to
the embedding module.
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import from chunking module - using the correct API
from src.chunking.text_chunkers.chonky_chunker import chunk_document


def process_sample_document(output_dir: Optional[Path] = None):
    """Process a sample document and output the chunked result.
    
    Args:
        output_dir: Directory to save the output JSON file
    """
    # Setup output directory
    if output_dir is None:
        output_dir = Path("./test-output/embedding-input")
    output_dir.mkdir(parents=True, exist_ok=True)
    
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
    document = {
        "id": doc_id,
        "path": str(sample_file),
        "content": content,
        "type": "text",
        "metadata": {
            "format": "text",
            "language": "en",
            "creation_date": "2025-05-10",
            "author": "HADES-PathRAG Team"
        }
    }
    
    print(f"Processing sample document: {doc_id}")
    print(f"Document content length: {len(content)} characters")
    
    # Process with chunk_document function
    chunked_doc = chunk_document(
        document=document,
        max_tokens=1024,
        return_pydantic=False,  # Return as dictionary
        save_to_disk=False
    )
    
    # Count chunks
    chunks = chunked_doc.get("chunks", [])
    print(f"Generated {len(chunks)} chunks")
    
    # Save the chunked document as JSON
    output_path = output_dir / "sample_chunked_document.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(chunked_doc, f, indent=2)
    
    print(f"Saved chunked document to {output_path}")
    
    # Also create a sample Python file
    output_py_path = output_dir / "sample_chunked_document.py"
    with open(output_py_path, 'w', encoding='utf-8') as f:
        f.write("#!/usr/bin/env python3\n")
        f.write("\"\"\"Sample chunked document for embedding module development.\"\"\"\n\n")
        f.write("# This is the exact JSON structure passed from chunking to embedding\n")
        f.write("SAMPLE_CHUNKED_DOCUMENT = ")
        f.write(json.dumps(chunked_doc, indent=4))
        f.write("\n")
    
    print(f"Saved Python sample to {output_py_path}")
    
    # Print sample chunk
    if chunks:
        first_chunk = chunks[0]
        print("\nSample first chunk:")
        if "metadata" in first_chunk and "content" in first_chunk["metadata"]:
            content = first_chunk["metadata"]["content"]
            preview = content[:100] + "..." if len(content) > 100 else content
            print(f"Chunk index: {first_chunk.get('chunk_index', 0)}")
            print(f"Content: {preview}")


if __name__ == "__main__":
    process_sample_document()

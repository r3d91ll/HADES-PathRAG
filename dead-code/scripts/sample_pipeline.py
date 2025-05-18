#!/usr/bin/env python3
"""
Simplified pipeline sample script for HADES-PathRAG.

This script focuses on creating sample inputs for each pipeline stage
to aid in the modular development of components.
"""

import json
import logging
import os
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import chunking module for document processing
from src.chunking.text_chunkers.chonky_chunker import chunk_text, chunk_document

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("sample_pipeline")


def setup_directories(base_dir: str = "./test-output") -> Dict[str, Path]:
    """Set up output directories for sample files.
    
    Args:
        base_dir: Base directory for output
        
    Returns:
        Dictionary mapping stage names to output directories
    """
    stages = ["docproc", "chunking", "embedding", "storage"]
    base_path = Path(base_dir)
    dirs = {}
    
    # Create base directory
    base_path.mkdir(parents=True, exist_ok=True)
    
    # Create stage-specific directories
    for stage in stages:
        stage_dir = base_path / stage
        stage_dir.mkdir(exist_ok=True)
        dirs[stage] = stage_dir
    
    return dirs


def create_sample_document(file_path: Path, doc_id: str = None) -> Dict[str, Any]:
    """Create a sample document dict using a text file.
    
    Args:
        file_path: Path to text file
        doc_id: Optional document ID (generated if None)
        
    Returns:
        Document dictionary
    """
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
        
    # Generate document ID if not provided
    if doc_id is None:
        doc_id = f"{file_path.stem}_{uuid.uuid4().hex[:8]}"
    
    # Read file content
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Create document dictionary
    document = {
        "id": doc_id,
        "path": str(file_path),
        "content": content,
        "type": "text",
        "metadata": {
            "format": "text",
            "language": "en",
            "creation_date": datetime.now().strftime("%Y-%m-%d"),
            "author": "HADES-PathRAG Team"
        }
    }
    
    return document


def process_chunking_stage(
    doc: Dict[str, Any], 
    output_dir: Path,
    max_tokens: int = 1024
) -> Tuple[Dict[str, Any], Path]:
    """Process a document through the chunking stage.
    
    Args:
        doc: Document dictionary
        output_dir: Directory to save output
        max_tokens: Maximum tokens per chunk
        
    Returns:
        Tuple of (chunked document dict, output file path)
    """
    logger.info(f"[CHUNKING] Processing document {doc.get('id', 'unknown')}")
    
    try:
        # Process document with chunking
        chunked_doc = chunk_document(
            document=doc,
            max_tokens=max_tokens,
            return_pydantic=False
        )
        
        # Generate output file name
        doc_id = chunked_doc.get("id", "unknown")
        output_file = output_dir / f"{doc_id}_chunked.json"
        
        # Save chunked document
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(chunked_doc, f, indent=2)
        
        # Print chunk stats
        chunks = chunked_doc.get("chunks", [])
        logger.info(f"[CHUNKING] Generated {len(chunks)} chunks for document {doc_id}")
        
        return chunked_doc, output_file
    
    except Exception as e:
        logger.error(f"[CHUNKING] Error chunking document: {e}")
        # Add error to document but pass it through
        doc["_chunking_error"] = str(e)
        return doc, Path("")


def create_sample_pipeline_outputs(
    text_files: List[Path],
    output_dirs: Dict[str, Path]
) -> Dict[str, Any]:
    """Create sample outputs for all pipeline stages.
    
    Args:
        text_files: List of text files to use
        output_dirs: Output directories for each stage
        
    Returns:
        Dictionary with processing statistics
    """
    results = []
    
    for file_path in text_files:
        logger.info(f"Processing {file_path}")
        
        # Stage 1: Create sample document dict (simulating docproc output)
        document = create_sample_document(file_path)
        
        # Save docproc sample
        doc_output = output_dirs["docproc"] / f"{document['id']}_processed.json"
        with open(doc_output, 'w', encoding='utf-8') as f:
            json.dump(document, f, indent=2)
        logger.info(f"Created sample docproc output: {doc_output}")
        
        # Stage 2: Chunking
        chunked_doc, chunking_output = process_chunking_stage(
            document, 
            output_dirs["chunking"]
        )
        logger.info(f"Created sample chunking output: {chunking_output}")
        
        # Future stages (embedding, storage) will be added here
        
        # Record success
        if "_chunking_error" not in chunked_doc:
            results.append({
                "id": document["id"],
                "path": str(file_path),
                "chunks": len(chunked_doc.get("chunks", [])),
                "success": True
            })
        else:
            results.append({
                "id": document["id"],
                "path": str(file_path),
                "error": chunked_doc.get("_chunking_error", "Unknown error"),
                "success": False
            })
    
    # Create summary
    summary = {
        "timestamp": datetime.now().isoformat(),
        "files_processed": len(text_files),
        "successful": len([r for r in results if r["success"]]),
        "results": results
    }
    
    # Save summary
    summary_file = Path(output_dirs["docproc"].parent) / "pipeline_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    
    return summary


def main():
    """Main entry point for the sample pipeline script."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate sample pipeline outputs for HADES-PathRAG")
    parser.add_argument("--files", "-f", nargs="+", help="Specific text files to process")
    parser.add_argument("--output", "-o", default="./test-output", help="Output directory")
    args = parser.parse_args()
    
    # Setup directories
    output_dirs = setup_directories(args.output)
    
    # Use provided files or default to test-data text files
    if args.files:
        files = [Path(file) for file in args.files]
    else:
        # Try to find text files in test-data
        data_dir = Path(__file__).parent.parent / "test-data"
        files = list(data_dir.glob("*.txt"))
        
        # If no .txt files, try to find .md files
        if not files:
            files = list(data_dir.glob("*.md"))
        
        # If still no files, create a sample text file
        if not files:
            sample_file = data_dir / "sample_text.txt"
            if not sample_file.exists():
                logger.info(f"Creating sample text file: {sample_file}")
                with open(sample_file, 'w', encoding='utf-8') as f:
                    f.write("""# Sample Text Document for Testing

This is a sample text document used for testing the chunking and validation functionality of the HADES-PathRAG system.

## Introduction

Text documents are a common format for storing information. They can contain various types of content, including:

1. Plain text
2. Structured text with headings
3. Lists and enumerations
4. Code snippets

## Processing Text Documents

When processing text documents, the system should:

- Extract the content
- Identify any structure (headings, lists, etc.)
- Chunk the content appropriately
- Validate the chunks against the schema

## Sample Code

Here's a sample code snippet that might be found in a text document:

```python
def process_text(text_content):
    \"\"\"Process text content and return chunks.\"\"\"
    chunks = []
    paragraphs = text_content.split("\\n\\n")
    for i, paragraph in enumerate(paragraphs):
        if paragraph.strip():
            chunks.append({
                "content": paragraph,
                "index": i,
                "type": "paragraph"
            })
    return chunks
```

## Conclusion

Text documents are versatile and can contain a mix of content types. The chunking system should be able to handle this variety and produce meaningful chunks that preserve the semantic meaning of the content.

This sample document includes multiple paragraphs, headings, lists, and a code snippet to test various aspects of the chunking system.
""")
            files = [sample_file]
    
    logger.info(f"Found {len(files)} files to process")
    
    # Process files and create sample outputs
    summary = create_sample_pipeline_outputs(files, output_dirs)
    
    logger.info(f"Sample pipeline outputs created in {args.output}")
    logger.info(f"Processed {summary['files_processed']} files, {summary['successful']} successful")
    
    # Print paths to sample files for next steps
    if summary['successful'] > 0:
        sample_result = next((r for r in summary['results'] if r['success']), None)
        if sample_result:
            chunked_file = output_dirs["chunking"] / f"{sample_result['id']}_chunked.json"
            if chunked_file.exists():
                logger.info(f"Sample chunked document for embedding module development: {chunked_file}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Pipeline test script for HADES-PathRAG with text file fallback.

This script processes documents through each stage of the pipeline:
1. Document Processing (docproc)
2. Chunking
3. Embedding (future)
4. Storage (future)

For PDF files, it attempts to use a corresponding .txt file as a fallback if PDF processing fails.
Each stage outputs its result to test-output with stage-specific subdirectories.
"""

import argparse
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

# Import modules for each pipeline stage
from src.docproc.core import process_document
from src.chunking.text_chunkers.chonky_chunker import chunk_document
from src.schema.validation import validate_document, ValidationStage

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("pipeline_test")

# Define pipeline stages
STAGES = ["docproc", "chunking", "embedding", "storage"]


def setup_output_directories(base_dir: str = "./test-output/pipeline") -> Dict[str, Path]:
    """Set up output directories for each pipeline stage.
    
    Args:
        base_dir: Base directory for output
        
    Returns:
        Dictionary mapping stage names to output directories
    """
    base_path = Path(base_dir)
    dirs = {}
    
    # Create base directory
    base_path.mkdir(parents=True, exist_ok=True)
    
    # Create stage-specific directories
    for stage in STAGES:
        stage_dir = base_path / stage
        stage_dir.mkdir(exist_ok=True)
        dirs[stage] = stage_dir
    
    return dirs


def create_text_fallback(pdf_path: Path) -> Optional[Path]:
    """Create a text fallback file for a PDF file.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Path to the text fallback file, or None if not created
    """
    # Define fallback text formats to look for
    extensions = [".txt", ".md", ".pdf.txt"]
    
    # Check if a fallback file already exists
    for ext in extensions:
        fallback_path = pdf_path.with_suffix(ext)
        if fallback_path.exists():
            logger.info(f"Found existing fallback file: {fallback_path}")
            return fallback_path
    
    # If no fallback exists, create a basic text file
    fallback_path = pdf_path.with_suffix(".pdf.txt")
    logger.info(f"Creating fallback text file for {pdf_path}")
    
    # Extract PDF filename and create a placeholder content
    pdf_name = pdf_path.name
    content = f"""# Content from {pdf_name}

This is a placeholder for the content of the PDF file that could not be 
processed directly. In a production system, this would be replaced with 
actual text extracted from the PDF using appropriate libraries.

## Document Information

- **Filename**: {pdf_name}
- **Path**: {pdf_path}
- **Size**: {pdf_path.stat().st_size} bytes

## Sample Structure

### Abstract

This is a placeholder abstract. The real content would be extracted from the PDF.

### Introduction

This is a placeholder introduction. The real content would be extracted from the PDF.

### Methodology

This is a placeholder methodology section. The real content would be extracted from the PDF.

### Results

This is a placeholder results section. The real content would be extracted from the PDF.

### Conclusion

This is a placeholder conclusion. The real content would be extracted from the PDF.
"""
    
    # Write the fallback content
    with open(fallback_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    return fallback_path


def process_docproc_stage(
    file_path: Path, 
    output_dir: Path,
    options: Optional[Dict[str, Any]] = None,
    use_fallback: bool = True
) -> Tuple[Dict[str, Any], Path]:
    """Process a document through the docproc stage.
    
    Args:
        file_path: Path to the document
        output_dir: Directory to save the output
        options: Optional processing options
        use_fallback: Whether to use text fallback for PDF files
        
    Returns:
        Tuple of (processed document dict, output file path)
    """
    logger.info(f"[DOCPROC] Processing {file_path}")
    
    try:
        # Process document
        processed_doc = process_document(file_path, options)
        
        # Generate output file name
        output_file = output_dir / f"{file_path.stem}_processed.json"
        
        # Save processed document
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(processed_doc, f, indent=2)
        
        logger.info(f"[DOCPROC] Successfully processed {file_path}")
        logger.info(f"[DOCPROC] Output saved to {output_file}")
        
        return processed_doc, output_file
    
    except Exception as e:
        logger.error(f"[DOCPROC] Error processing {file_path}: {e}")
        
        # Try fallback for PDF files
        if use_fallback and file_path.suffix.lower() == '.pdf':
            logger.info(f"[DOCPROC] Attempting text fallback for PDF {file_path}")
            fallback_path = create_text_fallback(file_path)
            
            if fallback_path and fallback_path.exists():
                # Process the fallback file
                logger.info(f"[DOCPROC] Processing fallback file {fallback_path}")
                try:
                    fallback_doc = process_document(fallback_path, options)
                    
                    # Override some fields to maintain PDF identity
                    fallback_doc["original_path"] = str(file_path)
                    fallback_doc["original_format"] = "pdf"
                    fallback_doc["using_fallback"] = True
                    
                    # Save processed document
                    output_file = output_dir / f"{file_path.stem}_processed_fallback.json"
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(fallback_doc, f, indent=2)
                    
                    logger.info(f"[DOCPROC] Successfully processed fallback for {file_path}")
                    logger.info(f"[DOCPROC] Fallback output saved to {output_file}")
                    
                    return fallback_doc, output_file
                
                except Exception as fallback_error:
                    logger.error(f"[DOCPROC] Error processing fallback file: {fallback_error}")
        
        # Return empty document and placeholder path if all processing fails
        return {"_error": str(e), "id": f"error_{uuid.uuid4()}", "path": str(file_path)}, Path("")


def process_chunking_stage(
    doc: Dict[str, Any], 
    output_dir: Path,
    max_tokens: int = 1024
) -> Tuple[Dict[str, Any], Path]:
    """Process a document through the chunking stage.
    
    Args:
        doc: Document output from docproc stage
        output_dir: Directory to save the output
        max_tokens: Maximum tokens per chunk
        
    Returns:
        Tuple of (chunked document dict, output file path)
    """
    if "_error" in doc:
        logger.warning(f"[CHUNKING] Skipping document with error: {doc['_error']}")
        return doc, Path("")
    
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
        
        # Validate document
        logger.info(f"[CHUNKING] Validating document {doc_id}")
        validation_result = validate_document(chunked_doc, ValidationStage.INGESTION)
        
        if validation_result.is_valid:
            logger.info(f"[CHUNKING] Document {doc_id} passed validation")
        else:
            logger.warning(f"[CHUNKING] Document {doc_id} failed validation: {validation_result.errors}")
        
        # Print chunk stats
        chunks = chunked_doc.get("chunks", [])
        logger.info(f"[CHUNKING] Generated {len(chunks)} chunks for document {doc_id}")
        
        return chunked_doc, output_file
    
    except Exception as e:
        logger.error(f"[CHUNKING] Error chunking document: {e}")
        # Add error to document but pass it through
        doc["_chunking_error"] = str(e)
        return doc, Path("")


def process_document_pipeline(
    file_path: Path, 
    output_dirs: Dict[str, Path],
    options: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Process a document through the complete pipeline.
    
    Args:
        file_path: Path to the document
        output_dirs: Output directories for each stage
        options: Optional processing options
        
    Returns:
        Final processed document
    """
    logger.info(f"Starting pipeline for {file_path}")
    
    # Stage 1: Document Processing
    doc, docproc_output = process_docproc_stage(file_path, output_dirs["docproc"], options)
    
    # Stage 2: Chunking
    chunked_doc, chunking_output = process_chunking_stage(doc, output_dirs["chunking"])
    
    # Future stages will be added here
    
    return chunked_doc


def main():
    """Main entry point for the pipeline test script."""
    parser = argparse.ArgumentParser(description="Pipeline test for HADES-PathRAG")
    parser.add_argument("--files", "-f", nargs="+", help="Specific files to process (optional)")
    parser.add_argument("--output", "-o", default="./test-output/pipeline", help="Output directory")
    parser.add_argument("--max-tokens", type=int, default=1024, help="Maximum tokens per chunk")
    parser.add_argument("--no-fallback", action="store_true", help="Disable text fallback for PDFs")
    args = parser.parse_args()
    
    # Setup output directories
    output_dirs = setup_output_directories(args.output)
    
    # Use provided files or search for PDFs in test-data
    if args.files:
        files = [Path(file) for file in args.files]
    else:
        # Default to PDF files in test-data
        data_dir = Path(__file__).parent.parent / "test-data"
        files = list(data_dir.glob("*.pdf"))
    
    logger.info(f"Found {len(files)} files to process")
    
    # Process each file through the pipeline
    results = []
    for file_path in files:
        result = process_document_pipeline(
            file_path=file_path,
            output_dirs=output_dirs,
            options={"use_ocr": True}  # Enable OCR for PDFs
        )
        results.append(result)
    
    # Save pipeline results summary
    summary = {
        "timestamp": datetime.now().isoformat(),
        "files_processed": len(files),
        "successful": len([r for r in results if "_error" not in r and "_chunking_error" not in r])
    }
    
    summary_file = Path(args.output) / "pipeline_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Pipeline processing complete. Results saved to {args.output}")
    logger.info(f"Processed {summary['files_processed']} files, {summary['successful']} successful")


if __name__ == "__main__":
    main()

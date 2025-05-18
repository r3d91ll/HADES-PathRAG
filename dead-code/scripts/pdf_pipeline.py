#!/usr/bin/env python3
"""
PDF processing pipeline for HADES-PathRAG.

This script processes PDF documents through the document processing and chunking
stages of the pipeline, preparing them for embedding generation. It leverages
the same approach as the docproc-chunking integration tests but is adapted
for processing real PDFs in a production-like environment.
"""

import os
import json
import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, cast

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import document processing modules
from src.docproc.core import process_document

# Import chunking modules
from src.chunking.text_chunkers.chonky_chunker import chunk_document

# Import schema validation
from src.schema.validation import validate_document, ValidationStage

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("pdf_pipeline")


def process_pdf(
    pdf_path: Path,
    output_dir: Path,
    max_tokens: int = 1024,
    doc_type: str = "academic_pdf"
) -> Optional[Path]:
    """
    Process a PDF document through the document processing and chunking pipeline.
    
    Args:
        pdf_path: Path to the PDF file
        output_dir: Directory to save the output
        max_tokens: Maximum tokens per chunk
        doc_type: Document type to use in metadata
        
    Returns:
        Path to the output JSON file, or None if processing failed
    """
    try:
        logger.info(f"Processing PDF: {pdf_path}")
        
        # Step 1: Process the PDF document using docproc
        try:
            processed_doc = process_document(pdf_path)
            logger.info(f"Document processing successful for {pdf_path}")
        except Exception as e:
            logger.error(f"Document processing failed for {pdf_path}: {e}")
            
            # Try using a text version if it exists
            text_path = pdf_path.with_suffix(".pdf.txt")
            if text_path.exists():
                logger.info(f"Attempting to use text version: {text_path}")
                with open(text_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Create a simple processed document
                processed_doc = {
                    "id": f"pdf_{os.urandom(4).hex()}_{pdf_path.stem}",
                    "path": str(pdf_path),
                    "content": content,
                    "type": doc_type,
                    "metadata": {
                        "format": "pdf",
                        "language": "en",
                        "title": pdf_path.stem,
                        "source": str(pdf_path)
                    }
                }
                logger.info(f"Created processed document from text version")
            else:
                # No fallback available
                logger.error(f"No text version found for {pdf_path}")
                return None
        
        # Save the processed document for inspection
        processed_path = output_dir / f"{processed_doc['id']}_processed.json"
        with open(processed_path, 'w', encoding='utf-8') as f:
            json.dump(processed_doc, f, indent=2)
        logger.info(f"Saved processed document to {processed_path}")
        
        # Step 2: Process the document with the chunking module
        try:
            # Ensure document has the required fields
            document = {
                "id": processed_doc.get("id", f"pdf_{os.urandom(4).hex()}_{pdf_path.stem}"),
                "path": str(pdf_path),
                "content": processed_doc.get("content", ""),
                "type": doc_type,
                "metadata": processed_doc.get("metadata", {})
            }
            
            # Chunk the document
            chunked_doc = chunk_document(
                document=document,
                max_tokens=max_tokens,
                return_pydantic=False
            )
            logger.info(f"Chunking successful, generated {len(chunked_doc.get('chunks', []))} chunks")
        
        except Exception as e:
            logger.error(f"Chunking failed for {pdf_path}: {e}")
            return None
        
        # Step 3: Validate the chunked document
        try:
            validation_result = validate_document(chunked_doc, ValidationStage.INGESTION)
            if validation_result.is_valid:
                logger.info("Document validation passed")
            else:
                logger.warning(f"Document validation failed: {validation_result.errors}")
        except Exception as e:
            logger.warning(f"Validation failed: {e}")
        
        # Save the chunked document
        output_path = output_dir / f"{document['id']}_chunked.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(chunked_doc, f, indent=2)
        logger.info(f"Saved chunked document to {output_path}")
        
        return output_path
    
    except Exception as e:
        logger.error(f"Failed to process {pdf_path}: {e}")
        return None


def main():
    """Main entry point for the PDF pipeline script."""
    parser = argparse.ArgumentParser(description="PDF Processing Pipeline for HADES-PathRAG")
    parser.add_argument("--input", "-i", type=str, nargs="+", required=True, 
                        help="Input PDF files to process")
    parser.add_argument("--output", "-o", type=str, default="./test-output/chunking_output",
                        help="Output directory for chunked documents")
    parser.add_argument("--max-tokens", type=int, default=1024,
                        help="Maximum tokens per chunk")
    args = parser.parse_args()
    
    # Prepare output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each input PDF
    results = []
    for input_path in args.input:
        file_path = Path(input_path)
        if not file_path.exists():
            logger.error(f"Input file not found: {file_path}")
            continue
        
        if file_path.suffix.lower() != '.pdf':
            logger.warning(f"Input file is not a PDF: {file_path}")
        
        output_path = process_pdf(
            pdf_path=file_path,
            output_dir=output_dir,
            max_tokens=args.max_tokens
        )
        
        results.append({
            "input": str(file_path),
            "output": str(output_path) if output_path else None,
            "success": output_path is not None
        })
    
    # Report results
    logger.info(f"Processed {len(results)} files, "
                f"{sum(1 for r in results if r['success'])} successful")
    
    # Save processing summary
    summary = {
        "processed": len(results),
        "successful": sum(1 for r in results if r['success']),
        "results": results
    }
    summary_path = output_dir / "processing_summary.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved processing summary to {summary_path}")


if __name__ == "__main__":
    main()

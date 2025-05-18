#!/usr/bin/env python
"""Document processing and chunking pipeline evaluation script.

This script creates a mini ingestion pipeline that:
1. Processes the ISNE and PathRAG PDF papers using the docproc module
2. Passes the processed documents to the chunking module
3. Displays the structure of the outputs at each stage
4. Compares results with existing test output files
5. Analyzes the complete pipeline flow
"""

import json
import sys
import os
import time
from pathlib import Path
from pprint import pprint
import difflib

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import document processing functions
from src.docproc.core import process_document

# Import chunking functions
from src.chunking import chunk_text, chunk_text_batch


def print_json_structure(data, prefix="", is_last=True, max_str_length=80, key=None):
    """Print the structure of a JSON object with types."""
    if key:
        key_str = f"{key} "
    else:
        key_str = ""
        
    if isinstance(data, dict):
        print(f"{prefix}{'└── ' if is_last else '├── '}{key_str}dict ({len(data)} keys)")
        prefix = prefix + ('    ' if is_last else '│   ')
        items = list(data.items())
        for i, (k, v) in enumerate(items):
            is_last_item = i == len(items) - 1
            print(f"{prefix}{'└── ' if is_last_item else '├── '}{k} ({type(v).__name__}):", end="")
            
            if isinstance(v, (dict, list)):
                print()  # Newline for nested structures
                print_json_structure(v, prefix + ('    ' if is_last_item else '│   '), True, max_str_length)
            else:
                str_value = str(v)
                if len(str_value) > max_str_length:
                    str_value = str_value[:max_str_length] + "..."
                print(f" {str_value}")
    
    elif isinstance(data, list):
        print(f"{prefix}{'└── ' if is_last else '├── '}{key_str}list ({len(data)} items)")
        if data and len(data) > 0:
            prefix = prefix + ('    ' if is_last else '│   ')
            # Just show the first item's structure if list is not empty
            print(f"{prefix}└── [0] ({type(data[0]).__name__}):", end="")
            if isinstance(data[0], (dict, list)):
                print()
                print_json_structure(data[0], prefix + '    ', True, max_str_length)
            else:
                str_value = str(data[0])
                if len(str_value) > max_str_length:
                    str_value = str_value[:max_str_length] + "..."
                print(f" {str_value}")
    else:
        print(f"{prefix}{'└── ' if is_last else '├── '}{key_str}{type(data).__name__}: {data}")


def process_pdf_file(pdf_path):
    """Process a PDF file and return the result."""
    print(f"\n=== STAGE 1: Document Processing - {pdf_path.name} ===\n")
    start_time = time.time()
    try:
        result = process_document(pdf_path, options={"validation_level": "warn"})
        processing_time = time.time() - start_time
        print(f"Document processing completed in {processing_time:.2f} seconds")
        return result
    except Exception as e:
        print(f"Error processing {pdf_path.name}: {e}")
        return None


def chunk_document(doc_result, chunk_options=None):
    """Chunk a processed document using the chunking module."""
    if not doc_result:
        return None
    
    print(f"\n=== STAGE 2: Document Chunking - {doc_result.get('id', 'Unknown')} ===\n")
    start_time = time.time()
    
    # Default chunking options
    options = {
        "max_tokens": 1024,        # Maximum tokens per chunk
        "doc_id": None,           # Will use document ID from processing
        "path": "unknown",        # Path to the document (will override)
        "doc_type": "text",       # Type of document
        "output_format": "json"   # Return format, one of "document", "json", "dict"
    }
    
    # Update with any provided options
    if chunk_options:
        options.update(chunk_options)
    
    try:
        # Extract the text content from the document
        content = doc_result.get('content', '')
        metadata = doc_result.get('metadata', {})
        doc_id = doc_result.get('id', 'unknown_doc')
        source_path = doc_result.get('source', 'unknown')
        
        # Set document-specific parameters
        options["doc_id"] = doc_id
        options["path"] = source_path
        
        # Run the chunking with correct parameters
        # Note: chunk_text expects specific parameters, not **options
        chunking_result = chunk_text(
            content=content,
            doc_id=options["doc_id"],
            path=options["path"],
            doc_type=options["doc_type"],
            max_tokens=options["max_tokens"],
            output_format=options["output_format"]
        )
        
        # The chunks are in chunking_result["chunks"] for the "json" output format
        chunks = chunking_result.get("chunks", [])
        
        # Create a structured result with both doc info and chunks
        result = {
            "document": doc_result,
            "chunks": chunks,
            "chunk_count": len(chunks),
            "options": options,
            "chunking_result": chunking_result,
            "processing_time": time.time() - start_time
        }
        
        chunking_time = time.time() - start_time
        print(f"Document chunking completed in {chunking_time:.2f} seconds")
        print(f"Created {len(chunks)} chunks from document")
        
        return result
    except Exception as e:
        print(f"Error chunking document: {e}")
        return None


def run_document_pipeline(pdf_path, chunking_options=None):
    """Run the complete document processing and chunking pipeline."""
    print(f"\n=== PIPELINE START: {pdf_path.name} ===\n")
    pipeline_start = time.time()
    
    # Stage 1: Document Processing
    doc_result = process_pdf_file(pdf_path)
    if not doc_result:
        print("Pipeline stopped: Document processing failed")
        return None
    
    # Stage 2: Document Chunking
    pipeline_result = chunk_document(doc_result, chunking_options)
    
    pipeline_time = time.time() - pipeline_start
    print(f"\n=== PIPELINE COMPLETE: {pdf_path.name} ===")
    print(f"Total pipeline execution time: {pipeline_time:.2f} seconds")
    
    return pipeline_result


def find_existing_output(pdf_name, output_dir):
    """Find existing output files for comparison."""
    # Look for files that might match our PDF name pattern
    base_name = pdf_name.split('.')[0].lower()
    matching_files = []
    
    for file in output_dir.glob("*.json"):
        if base_name in file.name.lower():
            matching_files.append(file)
    
    return matching_files


def compare_with_existing(current_result, pdf_name, output_dir):
    """Compare current results with existing output files."""
    if not current_result:
        print(f"No current result to compare for {pdf_name}")
        return
    
    existing_files = find_existing_output(pdf_name, output_dir)
    if not existing_files:
        print(f"No existing output files found for {pdf_name}")
        return
    
    print(f"\n=== Comparing with existing output files ===")
    for file in existing_files:
        print(f"\nComparing with {file.name}:")
        try:
            with open(file, 'r') as f:
                previous_result = json.load(f)
            
            # Compare keys at the top level
            current_keys = set(current_result.keys())
            previous_keys = set(previous_result.keys())
            
            print("Top-level keys comparison:")
            print(f"  Current keys ({len(current_keys)}): {', '.join(sorted(current_keys))}")
            print(f"  Previous keys ({len(previous_keys)}): {', '.join(sorted(previous_keys))}")
            
            # Show differences
            added = current_keys - previous_keys
            removed = previous_keys - current_keys
            if added:
                print(f"  Added keys: {', '.join(sorted(added))}")
            if removed:
                print(f"  Removed keys: {', '.join(sorted(removed))}")
            
            # Detailed metadata comparison if available
            if 'metadata' in current_result and 'metadata' in previous_result:
                print("\nMetadata comparison:")
                current_meta = current_result['metadata']
                previous_meta = previous_result['metadata']
                
                meta_keys = set(current_meta.keys()).union(set(previous_meta.keys()))
                for key in sorted(meta_keys):
                    current_val = current_meta.get(key, "N/A")
                    previous_val = previous_meta.get(key, "N/A")
                    
                    if current_val != previous_val:
                        print(f"  {key}:")
                        print(f"    Current: {str(current_val)[:80]}")
                        print(f"    Previous: {str(previous_val)[:80]}")
            
            # Content size comparison
            if 'content' in current_result and 'content' in previous_result:
                current_len = len(current_result['content'])
                previous_len = len(previous_result['content'])
                print(f"\nContent size comparison:")
                print(f"  Current: {current_len} chars")
                print(f"  Previous: {previous_len} chars")
                print(f"  Difference: {current_len - previous_len} chars ({(current_len/previous_len*100)-100:.1f}%)")
            
            # Entities count comparison
            if 'entities' in current_result and 'entities' in previous_result:
                current_count = len(current_result['entities'])
                previous_count = len(previous_result['entities'])
                print(f"\nEntities comparison:")
                print(f"  Current: {current_count} entities")
                print(f"  Previous: {previous_count} entities")
                print(f"  Difference: {current_count - previous_count} entities")
                
                # Show entity types distribution
                if current_count > 0:
                    entity_types = {}
                    for entity in current_result['entities']:
                        entity_type = entity.get('type', 'unknown')
                        entity_types[entity_type] = entity_types.get(entity_type, 0) + 1
                    print("  Entity types distribution:")
                    for entity_type, count in sorted(entity_types.items()):
                        print(f"    {entity_type}: {count}")
                
        except Exception as e:
            print(f"Error comparing with {file.name}: {e}")


def save_pipeline_output(pipeline_result, pdf_name, output_dir):
    """Save pipeline output for future comparison."""
    if not pipeline_result:
        return
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Save document processing output
    doc_output_path = output_dir / f"docproc_{pdf_name.split('.')[0]}_output.json"
    with open(doc_output_path, 'w') as f:
        json.dump(pipeline_result["document"], f, indent=2)
    print(f"Document processing output saved to: {doc_output_path}")
    
    # Save full pipeline output
    pipeline_output_path = output_dir / f"pipeline_{pdf_name.split('.')[0]}_output.json"
    with open(pipeline_output_path, 'w') as f:
        json.dump(pipeline_result, f, indent=2)
    print(f"Complete pipeline output saved to: {pipeline_output_path}")
    
    # Save chunks in separate file for easier analysis
    chunks_output_path = output_dir / f"chunks_{pdf_name.split('.')[0]}_output.json"
    with open(chunks_output_path, 'w') as f:
        json.dump({
            "doc_id": pipeline_result["document"].get("id", ""),
            "chunk_count": pipeline_result["chunk_count"],
            "chunks": pipeline_result["chunks"],
            "chunking_options": pipeline_result["options"]
        }, f, indent=2)
    print(f"Chunking output saved to: {chunks_output_path}")
    
    return {
        "document": doc_output_path,
        "pipeline": pipeline_output_path,
        "chunks": chunks_output_path
    }


def analyze_pipeline_results(pipeline_result, pdf_name):
    """Analyze the results of the complete pipeline."""
    if not pipeline_result:
        print(f"No pipeline results available for {pdf_name}")
        return
    
    print(f"\n=== PIPELINE ANALYSIS: {pdf_name} ===\n")
    
    # Document processing analysis
    doc_result = pipeline_result.get("document", {})
    content = doc_result.get("content", "")
    print(f"Document Processing Results:")
    print(f"  - Content length: {len(content)} characters")
    print(f"  - Entities extracted: {len(doc_result.get('entities', []))}")
    if 'metadata' in doc_result:
        metadata = doc_result['metadata']
        print(f"  - Title: {metadata.get('title', 'Unknown')}")
        print(f"  - Authors: {len(metadata.get('authors', []))} detected")
        print(f"  - Pages: {metadata.get('page_count', 'Unknown')}")
    
    # Validation issues
    if '_validation_error' in doc_result:
        print("\nValidation Issues:")
        print(f"  - {doc_result['_validation_error'][:80]}...")
    
    # Chunking analysis
    chunks = pipeline_result.get("chunks", [])
    chunk_count = pipeline_result.get("chunk_count", 0)
    options = pipeline_result.get("options", {})
    
    print(f"\nChunking Results:")
    print(f"  - Total chunks: {chunk_count}")
    print(f"  - Target chunk size: {options.get('chunk_size', 'Unknown')} characters")
    print(f"  - Overlap: {options.get('chunk_overlap', 'Unknown')} characters")
    
    # Chunk size distribution
    if chunks:
        chunk_sizes = [len(chunk.get('content', '')) for chunk in chunks]
        avg_size = sum(chunk_sizes) / len(chunk_sizes)
        max_size = max(chunk_sizes)
        min_size = min(chunk_sizes)
        print(f"  - Average chunk size: {avg_size:.1f} characters")
        print(f"  - Min chunk size: {min_size} characters")
        print(f"  - Max chunk size: {max_size} characters")
        print(f"  - Content distribution: {sum(chunk_sizes) / len(content) * 100:.1f}% of original content")
    
    # Performance metrics
    if "processing_time" in pipeline_result:
        print(f"\nPerformance Metrics:")
        print(f"  - Chunking time: {pipeline_result['processing_time']:.2f} seconds")
        print(f"  - Chunks per second: {chunk_count / pipeline_result['processing_time']:.1f}")
    
    print("\nPipeline Summary:")
    print(f"  - Document processing: {'✓ Success' if doc_result else '✗ Failed'}")
    print(f"  - Chunking: {'✓ Success' if chunks else '✗ Failed'}")
    print(f"  - Overall result: {'✓ Complete' if doc_result and chunks else '✗ Incomplete'}")


def main():
    """Main function to run the pipeline and analysis."""
    # Define file paths
    isne_pdf = project_root / "test-data" / "ISNE_paper.pdf"
    pathrag_pdf = project_root / "test-data" / "PathRAG_paper.pdf"
    output_dir = project_root / "test-output"
    
    # Ensure all paths exist
    if not isne_pdf.exists():
        print(f"ISNE paper not found at {isne_pdf}")
        return
    
    if not pathrag_pdf.exists():
        print(f"PathRAG paper not found at {pathrag_pdf}")
        return
    
    if not output_dir.exists():
        output_dir.mkdir(exist_ok=True, parents=True)
        print(f"Created output directory at {output_dir}")
    
    # Setup chunking options
    chunking_options = {
        "max_tokens": 1024,
        "output_format": "json",
        "doc_type": "academic_pdf"  # Set document type specifically for academic papers
    }
    
    # Process ISNE paper through complete pipeline
    print("\n" + "=" * 80)
    print(f"PROCESSING ISNE PAPER THROUGH COMPLETE PIPELINE")
    print("=" * 80)
    
    isne_pipeline_result = run_document_pipeline(isne_pdf, chunking_options)
    
    if isne_pipeline_result:
        # Show document structure
        print("\nDocument Structure:")
        print_json_structure(isne_pipeline_result["document"])
        
        # Show chunk sample
        if isne_pipeline_result.get("chunks"):
            first_chunk = isne_pipeline_result["chunks"][0]
            print("\nSample Chunk Structure:")
            print_json_structure(first_chunk)
            print(f"\nTotal chunks: {len(isne_pipeline_result['chunks'])}")
        
        # Compare with existing output
        compare_with_existing(isne_pipeline_result["document"], isne_pdf.name, output_dir)
        
        # Save output files
        save_pipeline_output(isne_pipeline_result, isne_pdf.name, output_dir)
        
        # Analyze results
        analyze_pipeline_results(isne_pipeline_result, isne_pdf.name)
    
    # Process PathRAG paper through complete pipeline
    print("\n" + "=" * 80)
    print(f"PROCESSING PATHRAG PAPER THROUGH COMPLETE PIPELINE")
    print("=" * 80)
    
    pathrag_pipeline_result = run_document_pipeline(pathrag_pdf, chunking_options)
    
    if pathrag_pipeline_result:
        # Show document structure
        print("\nDocument Structure:")
        print_json_structure(pathrag_pipeline_result["document"])
        
        # Show chunk sample
        if pathrag_pipeline_result.get("chunks"):
            first_chunk = pathrag_pipeline_result["chunks"][0]
            print("\nSample Chunk Structure:")
            print_json_structure(first_chunk)
            print(f"\nTotal chunks: {len(pathrag_pipeline_result['chunks'])}")
        
        # Compare with existing output
        compare_with_existing(pathrag_pipeline_result["document"], pathrag_pdf.name, output_dir)
        
        # Save output files
        save_pipeline_output(pathrag_pipeline_result, pathrag_pdf.name, output_dir)
        
        # Analyze results
        analyze_pipeline_results(pathrag_pipeline_result, pathrag_pdf.name)
    
    print("\n" + "=" * 80)
    print("PIPELINE TESTING COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()

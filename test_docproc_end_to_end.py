#!/usr/bin/env python
"""
End-to-end test script for the document processing module.

This script tests the document processing module with real files:
- PDF: data/PathRAG_paper.pdf
- HTML: data/langchain_docling.html
- Python: data/file_batcher.py

The processed results are saved to the test-output/ directory.
"""

import json
import os
import sys
import time
from pathlib import Path
import argparse

# Add the project root to the Python path
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

from src.docproc.adapters.docling_adapter import DoclingAdapter
from src.docproc.adapters.python_adapter import PythonAdapter
from src.docproc.utils.format_detector import detect_format_from_path
from src.docproc.serializers import save_to_json_file


def process_file(file_path: Path, output_dir: Path) -> dict:
    """
    Process a single file and save the results to the output directory.
    
    Args:
        file_path: Path to the input file
        output_dir: Directory to save results
        
    Returns:
        Processing result dictionary
    """
    print(f"\n{'-' * 80}")
    print(f"Processing file: {file_path}")
    
    # Detect format
    try:
        format_name = detect_format_from_path(file_path)
        print(f"Detected format: {format_name}")
    except ValueError as e:
        print(f"Error detecting format: {e}")
        return {"error": str(e), "file": str(file_path)}
    
    # Choose appropriate adapter
    if format_name == "python":
        adapter = PythonAdapter()
    else:
        # Use Docling for PDF, HTML, and other formats
        adapter = DoclingAdapter()
    
    # Process the file
    start_time = time.time()
    try:
        result = adapter.process(file_path)
        processing_time = time.time() - start_time
        print(f"Processing completed in {processing_time:.2f} seconds")
        
        # Extract basic stats
        entity_count = len(result.get("entities", []))
        content_length = len(result.get("content", ""))
        print(f"Content length: {content_length} characters")
        print(f"Entities found: {entity_count}")
        
        # Save the complete result in a single JSON file using the new serializer
        result_file = output_dir / f"{file_path.stem}_{format_name}.json"
        save_to_json_file(result, result_file)
        print(f"Complete results saved to: {result_file}")
        return result
    
    except Exception as e:
        error_message = f"Error processing {file_path}: {e}"
        print(f"ERROR: {error_message}")
        
        error_file = output_dir / f"{file_path.stem}_{format_name}_error.txt"
        with open(error_file, "w", encoding="utf-8") as f:
            f.write(error_message)
        
        return {"error": str(e), "file": str(file_path)}


def main():
    """Run the end-to-end test on all specified documents."""
    parser = argparse.ArgumentParser(description="Process documents with the docproc module")
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="test-output", 
        help="Directory to save processed output"
    )
    args = parser.parse_args()
    
    # Prepare paths
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    data_dir = Path("data")
    
    # Files to process
    files = [
        data_dir / "PathRAG_paper.pdf",
        data_dir / "langchain_docling.html",
        data_dir / "file_batcher.py"
    ]
    
    # Verify files exist
    missing_files = [f for f in files if not f.exists()]
    if missing_files:
        print(f"ERROR: The following files are missing: {', '.join(str(f) for f in missing_files)}")
        sys.exit(1)
    
    # Process each file
    results = {}
    for file_path in files:
        result = process_file(file_path, output_dir)
        results[file_path.name] = {
            "success": "error" not in result,
            "format": result.get("format", "unknown"),
            "content_length": len(result.get("content", "")),
            "entity_count": len(result.get("entities", [])) if isinstance(result.get("entities", []), list) else 0
        }
    
    # Create a summary report with file sizes
    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "files_processed": len(files),
        "results": results
    }
    
    summary_file = output_dir / "processing_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    
    # Calculate output file sizes for reporting
    output_sizes = {}
    for file_path in files:
        format_name = results[file_path.name]["format"]
        json_file = output_dir / f"{file_path.stem}_{format_name}.json"
        if json_file.exists():
            output_sizes[file_path.name] = json_file.stat().st_size
    
    print(f"\n{'-' * 80}")
    print(f"Processing summary saved to: {summary_file}")
    print(f"\nProcessed {len(files)} files:")
    for file_name, result in results.items():
        status = "✅ Success" if result["success"] else "❌ Failed"
        file_size = f"{output_sizes.get(file_name, 0) / (1024*1024):.2f} MB" if file_name in output_sizes else "N/A"
        print(f"  {status} - {file_name} ({result['format']}): {result['entity_count']} entities, {result['content_length']} chars, {file_size}")
        
    print(f"\nAll documents processed and saved to single JSON files in {output_dir}/")


if __name__ == "__main__":
    main()

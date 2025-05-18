#!/usr/bin/env python3
"""Isolated script to test docproc functionality.

This script directly imports the docproc modules without going through the main package.
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, List

# Add the src directory to the Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Force docling to be detected
try:
    import docling
    print(f"Found docling package")
    DOCLING_AVAILABLE = True
except ImportError:
    print("Docling not found, will use fallback processing")
    DOCLING_AVAILABLE = False

# Import docproc modules directly to avoid PathRAG imports
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Patch the docling_adapter module to use our DOCLING_AVAILABLE value
import src.docproc.adapters.docling_adapter
src.docproc.adapters.docling_adapter.DOCLING_AVAILABLE = DOCLING_AVAILABLE
from src.docproc.adapters.registry import get_adapter_for_format
from src.docproc.utils.format_detector import detect_format_from_path


def process_document(file_path: Path) -> Dict[str, Any]:
    """Process a document file, converting it to a standardized format."""
    # Check if file exists first
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Detect the document format
    format_type = detect_format_from_path(file_path)
    
    # Get the appropriate adapter
    adapter = get_adapter_for_format(format_type)
    
    # Process the document
    return adapter.process(file_path)


def parse_args():
    parser = argparse.ArgumentParser(description="Test docproc functionality.")
    parser.add_argument(
        "-d", "--directory", required=True, type=str, help="Directory to scan for documents."
    )
    parser.add_argument(
        "-o", "--output", type=str, default="data/test-output", 
        help="Directory to write JSON outputs."
    )
    return parser.parse_args()


def should_skip(path: Path, output_dir: Path) -> bool:
    """Return True if path should be skipped (directories/files)."""
    try:
        # py >=3.9 has Path.is_relative_to
        return path.resolve().is_relative_to(output_dir.resolve())
    except AttributeError:  # pragma: no cover
        try:
            path.resolve().relative_to(output_dir.resolve())
            return True
        except ValueError:
            return False


def process_directory(input_dir: Path, output_dir: Path) -> None:
    """Process all files in the input directory."""
    files_processed = 0
    files_skipped = 0
    errors: List[str] = []

    for file_path in input_dir.rglob("*"):
        if file_path.is_dir():
            continue
        if should_skip(file_path, output_dir):
            continue
            
        try:
            # Quick filter: detect_format_from_path will raise for unsupported
            detect_format_from_path(file_path)
        except Exception:
            files_skipped += 1
            continue

        try:
            result = process_document(file_path)
            
            # For JSON serialization, convert any non-serializable objects to strings
            def make_serializable(obj):
                if isinstance(obj, dict):
                    return {k: make_serializable(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [make_serializable(item) for item in obj]
                elif hasattr(obj, '__dict__'):
                    return str(obj)
                else:
                    return obj
                
            result = make_serializable(result)
            
            # Determine output path: preserve relative structure, add .json
            rel_path = file_path.relative_to(input_dir)
            out_path = output_dir / rel_path
            out_path = out_path.with_suffix(out_path.suffix + ".json")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
            files_processed += 1
            print(f"Processed: {file_path}")
        except Exception as exc:
            errors.append(f"{file_path}: {exc}")

    print(
        f"Processed {files_processed} files. Skipped {files_skipped}. "
        + (f"Errors: {len(errors)}" if errors else "")
    )
    if errors:
        for err in errors:
            print(f"ERROR: {err}", file=sys.stderr)


def main():
    args = parse_args()
    input_dir = Path(args.directory).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()

    if not input_dir.exists() or not input_dir.is_dir():
        print(f"Input directory not found: {input_dir}", file=sys.stderr)
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    process_directory(input_dir, output_dir)


if __name__ == "__main__":
    main()

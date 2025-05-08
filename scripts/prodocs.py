"""CLI utility to preprocess documents with *docproc*.

This utility processes documents into standardized JSON format for later pipeline stages.

Usage::

    python scripts/prodocs.py -d ./data/example_dir -o ./test-output

Scans the provided directory recursively (excluding the output directory itself),
processes each supported file and writes JSON files to the output directory.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Any

# Local imports - ensure the src directory is on PYTHONPATH when executed directly
try:
    from src.docproc.core import process_documents_batch
    from src.docproc.utils.format_detector import detect_format_from_path
except ModuleNotFoundError:  # pragma: no cover - executed from project root
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(PROJECT_ROOT))
    from src.docproc.core import process_documents_batch  # type: ignore  # noqa: E402
    from src.docproc.utils.format_detector import detect_format_from_path  # type: ignore  # noqa: E402


DEFAULT_OUTPUT_DIR = Path("test-output")


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:  # noqa: D401
    parser = argparse.ArgumentParser(description="Pre-process documents for RAG ingestion.")
    parser.add_argument(
        "-d", "--directory", required=True, type=str, help="Directory to scan for documents.",
    )
    parser.add_argument(
        "-o", "--output", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Directory to write JSON outputs.",
    )
    return parser.parse_args(argv)


def should_skip(path: Path, output_dir: Path) -> bool:  # noqa: D401
    """Return True if *path* should be skipped (directories/files)."""
    # Always skip JSON files - these are assumed to be generated outputs.
    if path.suffix.lower() == ".json":
        return True

    # Skip files inside the *current* output directory
    try:
        if path.resolve().is_relative_to(output_dir.resolve()):  # type: ignore[attr-defined]
            return True
    except AttributeError:  # pragma: no cover - for Python <3.9
        try:
            path.resolve().relative_to(output_dir.resolve())
            return True
        except ValueError:
            pass

    # Also skip any file contained in a directory named `test-output` (legacy runs)
    parts = {p.name for p in path.parents}
    if "test-output" in parts:
        return True

    return False


def on_document_processed(doc: Dict[str, Any], output_path: Path) -> None:
    """Callback when a document is successfully processed and saved."""
    print(f" Processed {doc['source']} -> {output_path}")


def on_document_error(path: str, error: Exception) -> None:
    """Callback when document processing fails."""
    print(f" Error processing {path}: {error}", file=sys.stderr)


def main(argv: List[str] | None = None) -> int:
    """Main entry point."""
    args = parse_args(argv)
    input_dir = Path(args.directory).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()
    
    if not input_dir.exists() or not input_dir.is_dir():
        print(f"Input directory not found: {input_dir}", file=sys.stderr)
        return 1
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect all files to process
    all_files = []
    for file_path in input_dir.rglob("*"):
        if file_path.is_dir():
            continue
        if should_skip(file_path, output_dir):
            continue
        try:
            # Quick filter: detect_format_from_path will raise for unsupported
            detect_format_from_path(file_path)
            all_files.append(file_path)
        except Exception:
            continue
    
    print(f"Found {len(all_files)} files to process")
    
    # Track processing time
    start_time = time.time()
    
    # Process all documents in batch
    stats = process_documents_batch(
        file_paths=all_files,
        output_dir=output_dir,
        on_success=on_document_processed,
        on_error=on_document_error
    )
    
    # Report statistics
    duration = time.time() - start_time
    print(f"\nProcessing complete in {duration:.2f} seconds:")
    print(f"  Processed: {stats['processed']}")
    print(f"  Errors: {stats['errors']}")
    print(f"  Skipped: {stats['skipped']}")
    
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

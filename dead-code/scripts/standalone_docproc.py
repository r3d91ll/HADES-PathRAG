#!/usr/bin/env python3
"""Standalone document processor that avoids importing the full PathRAG package.

This script contains a minimal implementation of the document processing functionality
needed to test the DoclingAdapter with proper fallbacks when Docling is not available.
"""
import argparse
import json
import re
import sys
import tempfile
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Set, cast

# Check if docling is available
try:
    import docling
    from docling.document_converter import DocumentConverter
    DOCLING_AVAILABLE = True
    print("Found docling package - will use it for document processing")
except ImportError:
    DOCLING_AVAILABLE = False
    print("Docling not available - will use fallback text processing")

# Extension to format mapping
EXTENSION_TO_FORMAT = {
    ".txt": "text",
    ".md": "markdown",
    ".py": "python",
    ".js": "javascript",
    ".html": "html",
    ".htm": "html",
    ".css": "css",
    ".json": "json",
    ".xml": "xml",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".csv": "csv",
    ".pdf": "pdf",
    ".doc": "doc",
    ".docx": "docx",
    ".ppt": "ppt",
    ".pptx": "pptx",
    ".xls": "xls",
    ".xlsx": "xlsx",
}

# Formats that might benefit from OCR
OCR_FORMATS = {"pdf", "doc", "docx", "ppt", "pptx", "xls", "xlsx", "image"}


class BaseAdapter:
    """Base adapter interface for document processing."""
    
    def process(self, file_path: Path, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process a document file."""
        raise NotImplementedError("Subclasses must implement process")
    
    def process_text(self, text: str, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process text content directly."""
        raise NotImplementedError("Subclasses must implement process_text")


class DoclingAdapter(BaseAdapter):
    """Adapter that leverages Docling to process rich document formats."""

    def __init__(self, options: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the DoclingAdapter with configuration options."""
        self.options: Dict[str, Any] = options or {}

        # Always attempt to instantiate `DocumentConverter` – this allows
        # unit-tests to supply a patched/mock implementation even when the real
        # dependency is missing.
        try:
            self.converter = DocumentConverter()  # type: ignore[call-arg]
            self._docling_available = True
        except Exception:
            self.converter = cast(Any, None)
            self._docling_available = False

    def process(
        self, file_path: Path, options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        if not file_path.exists():
            raise FileNotFoundError(file_path)

        opts = {**self.options, **(options or {})}
        format_name = detect_format_from_path(file_path)

        # Build deterministic ID – same pattern used across adapters
        doc_id = _build_doc_id(file_path, format_name)

        # ------------------------------------------------------------------
        # Fallback path when Docling converter is unavailable.
        # ------------------------------------------------------------------
        if not self._docling_available or self.converter is None:
            try:
                text_content = file_path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                # Binary files or unreadable encodings – just return empty string
                text_content = ""

            return {
                "id": doc_id,
                "source": str(file_path),
                "content": text_content,
                "content_type": "text",
                "format": format_name,
                "metadata": {
                    "format": format_name,
                    "file_path": str(file_path),
                },
                "entities": [],
            }

        # Build converter kwargs – currently we only surface `use_ocr`
        converter_kwargs: Dict[str, Any] = {}
        if format_name in OCR_FORMATS and opts.get("use_ocr", True):
            converter_kwargs["use_ocr"] = True

        try:
            result = self.converter.convert(file_path, **converter_kwargs)  # noqa: E501
            # Some test environments monkey-patch converter to return the document
            doc = getattr(result, "document", result)
        except Exception as exc:  # pragma: no cover – Docling failures
            # If failure due to unexpected kwarg, retry without kwargs once
            if (
                converter_kwargs
                and (
                    "unexpected keyword" in str(exc).lower()
                    or "unexpected_keyword_argument" in str(exc).lower()
                )
            ):
                try:
                    result = self.converter.convert(file_path)  # type: ignore[arg-type]
                    doc = getattr(result, "document", result)
                except Exception as exc2:  # pragma: no cover
                    raise ValueError(f"Docling failed to process {file_path}: {exc2}") from exc2
            else:
                raise ValueError(f"Docling failed to process {file_path}: {exc}") from exc

        # Extract content – prefer markdown for downstream chunkers
        markdown_content: str
        if hasattr(doc, "export_to_markdown"):
            markdown_content = str(doc.export_to_markdown())
        elif hasattr(doc, "export_to_text"):
            markdown_content = str(doc.export_to_text())
        elif isinstance(doc, dict) and "content" in doc:
            markdown_content = str(doc["content"])
        else:
            try:
                markdown_content = file_path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                markdown_content = ""

        metadata = self.extract_metadata(doc)
        metadata.update({"format": format_name, "file_path": str(file_path)})

        return {
            "id": doc_id,
            "source": str(file_path),
            "content": markdown_content,
            "content_type": "markdown",
            "format": format_name,
            "metadata": metadata,
            "entities": self.extract_entities(doc),
            **({"docling_document": str(doc)} if isinstance(doc, (str, dict)) else {}),
        }

    def process_text(
        self, text: str, options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Process text content directly."""
        opts = {**self.options, **(options or {})}
        format_name = opts.get("format", "text")
        suffix = f".{format_name}" if format_name != "text" else ".txt"

        with tempfile.NamedTemporaryFile(mode="w", suffix=suffix, delete=False) as tmp:
            tmp.write(text)
            tmp_path = Path(tmp.name)

        try:
            return self.process(tmp_path, options)
        finally:
            tmp_path.unlink(missing_ok=True)

    def extract_entities(self, content: Any) -> List[Dict[str, Any]]:
        """Extract entities from document content."""
        entities: List[Dict[str, Any]] = []
        
        # Extract headings if available
        if hasattr(content, "headings"):
            headings = getattr(content, "headings", [])
            for heading in headings:
                entities.append(
                    {
                        "type": "heading",
                        "text": getattr(heading, "text", str(heading)),
                        "level": getattr(heading, "level", 1),
                    }
                )
                
        return entities

    def extract_metadata(self, content: Any) -> Dict[str, Any]:
        """Extract metadata from document content."""
        metadata: Dict[str, Any] = {}
        raw_meta = getattr(content, "metadata", {})
        if isinstance(raw_meta, dict):
            for k, v in raw_meta.items():
                if isinstance(v, (str, int, float, bool)):
                    metadata[k.lower()] = v

        # Page count is handy
        pages = getattr(content, "pages", None)
        if pages is not None:
            metadata["page_count"] = len(pages)
        return metadata

    def convert_to_markdown(self, content: Any) -> str:
        """Convert content to markdown format."""
        if hasattr(content, "export_to_markdown"):
            return str(content.export_to_markdown())
        if isinstance(content, dict) and "content" in content:
            return str(content["content"])
        if isinstance(content, str):
            return content
        return self.convert_to_text(content)

    def convert_to_text(self, content: Any) -> str:
        """Convert content to plain text format."""
        if hasattr(content, "export_to_text"):
            return str(content.export_to_text())
        if isinstance(content, dict) and "content" in content:
            return str(content["content"])
        if isinstance(content, str):
            return content
        return ""


class PythonAdapter(BaseAdapter):
    """Simplified Python adapter that just reads the file as text."""
    
    def __init__(self, options: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the PythonAdapter."""
        self.options = options or {}
    
    def process(self, file_path: Path, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process a Python file."""
        if not file_path.exists():
            raise FileNotFoundError(file_path)
        
        opts = {**self.options, **(options or {})}
        format_name = "python"
        doc_id = _build_doc_id(file_path, format_name)
        
        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            content = ""
        
        return {
            "id": doc_id,
            "source": str(file_path),
            "content": content,
            "content_type": "code",
            "format": format_name,
            "metadata": {
                "format": format_name,
                "file_path": str(file_path),
            },
            "entities": [],
        }
    
    def process_text(self, text: str, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process Python text content directly."""
        opts = {**self.options, **(options or {})}
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tmp:
            tmp.write(text)
            tmp_path = Path(tmp.name)
        
        try:
            return self.process(tmp_path, options)
        finally:
            tmp_path.unlink(missing_ok=True)


# Registry for adapters
_ADAPTER_REGISTRY: Dict[str, type] = {}
_DEFAULT_ADAPTER: Optional[type] = None


def register_adapter(format_name: str, adapter_class: type) -> None:
    """Register an adapter for a specific format."""
    _ADAPTER_REGISTRY[format_name.lower()] = adapter_class


def get_adapter_for_format(format_name: str, options: Optional[Dict[str, Any]] = None) -> BaseAdapter:
    """Get an adapter instance for a specific format."""
    format_name = format_name.lower()
    adapter_class = _ADAPTER_REGISTRY.get(format_name)
    
    if adapter_class is None:
        if _DEFAULT_ADAPTER is None:
            raise ValueError(f"No adapter registered for format: {format_name}")
        adapter_class = _DEFAULT_ADAPTER
    
    return adapter_class(options)


def detect_format_from_path(file_path: Path) -> str:
    """Detect the format of a document from its file path."""
    suffix = file_path.suffix.lower()
    if not suffix:
        # Default to text for files without extension
        return "text"
    
    return EXTENSION_TO_FORMAT.get(suffix, "text")


def _build_doc_id(file_path: Path, format_name: str) -> str:
    """Build a deterministic document ID."""
    safe_name = re.sub(r"[^A-Za-z0-9_:\-@\.\(\)\+\,=;\$!\*'%]+", "_", file_path.name)
    return f"{format_name}_{hashlib.md5(str(file_path).encode()).hexdigest()[:8]}_{safe_name}"


# Register adapters
register_adapter("python", PythonAdapter)

# Register DoclingAdapter for all formats
for fmt in set(EXTENSION_TO_FORMAT.values()) | {"text", "document"}:
    register_adapter(fmt, DoclingAdapter)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Process documents for RAG ingestion.")
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
            # Process the document
            adapter = get_adapter_for_format(detect_format_from_path(file_path))
            result = adapter.process(file_path)
            
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
    """Main entry point."""
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

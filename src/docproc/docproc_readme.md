# HADES-PathRAG: Document Processing Module (`src/docproc`)

## Overview

The `src/docproc` module provides a unified, extensible interface for processing documents of various types—including PDFs, HTML, Python code, and structured data (JSON, XML, YAML)—into standardized, machine-readable formats. It is responsible for:

- Format detection and normalization
- Metadata and entity extraction
- Specialized adapters for different document types
- Standardized JSON serialization (for downstream RAG and analytics)

This module is a core part of the HADES-PathRAG pipeline, ensuring all documents are regularized and ready for chunking, embedding, or graph construction.

---

## Main Components

- **`core.py`**: Unified entry points for document processing (`process_document`, `process_text`, etc.).
- **`adapters/`**: Format-specific adapters (e.g., `docling_adapter.py` for general documents, `python_adapter.py` for Python code).
- **`schemas/`**: Pydantic models and validation logic for all document types.
- **`models/`**: Data models for code and general documents.
- **`serializers/`**: JSON serialization utilities for standardized output.
- **`utils/`**: Format detection and helper utilities.

---

## Usage Example

```python
from src.docproc import process_document
from src.docproc.serializers import save_to_json_file

result = process_document("myfile.pdf")
save_to_json_file(result, "output/myfile.json")
```

---

## JSON Output Schemas

### 1. Docling (General Document) JSON Schema

A typical output for a Docling-processed document:

```json
{
  "id": "<unique_id>",
  "source": "<file_path_or_uri>",
  "content": "<main_text_content>",
  "content_type": "text|markdown|html|...",
  "format": "pdf|docx|html|csv|...",
  "raw_content": "<original_raw_content>",
  "metadata": {
    "language": "en|...",
    "format": "pdf|docx|...",
    "content_type": "text|...",
    "file_size": 123456,
    "line_count": 123,
    "char_count": 4567,
    "has_errors": false,
    "...": "... (adapter-specific fields) ..."
  },
  "entities": [
    {
      "type": "entity_type",
      "value": "entity_value",
      "line": 42,
      "confidence": 0.98
    },
    "... more ..."
  ],
  "error": null,
  "version": "1.0.0",
  "timestamp": "2025-05-07T21:00:03-05:00"
}
```

- **Note**: `entities` are adapter-specific (e.g., named entities for text, symbols for code).
- **`version`** and **`timestamp`** are injected by the serialization module.

---

### 2. Python Code JSON Schema

A typical output for a processed Python file (see also `schemas/python_document.py`, `models/python_code.py`):

```json
{
  "format": "python",
  "metadata": {
    "language": "python",
    "format": "python",
    "content_type": "code",
    "file_size": 1234,
    "line_count": 56,
    "char_count": 7890,
    "has_errors": false,
    "function_count": 5,
    "class_count": 2,
    "import_count": 3,
    "method_count": 4,
    "has_module_docstring": true,
    "has_syntax_errors": false
  },
  "entities": [
    {
      "type": "function|class|import|...",
      "value": "<name>",
      "line": 10,
      "confidence": 1.0,
      "...": "... (entity-specific fields) ..."
    }
    // ...more entities...
  ],
  "relationships": [
    {
      "source": "<qualified_name>",
      "target": "<qualified_name>",
      "type": "CALLS|CONTAINS|EXTENDS|...",
      "weight": 1.0,
      "line": 23
    }
    // ...more relationships...
  ],
  "symbol_table": {
    "type": "module",
    "name": "<module_name>",
    "docstring": "...",
    "path": "<file_path>",
    "module_path": "<dotted.path>",
    "line_range": [1, 56],
    "elements": [
      // Nested code elements (functions, classes, etc.)
    ]
  },
  "error": null,
  "version": "1.0.0",
  "timestamp": "2025-05-07T21:00:03-05:00"
}
```

- See `schemas/python_document.py` and `models/python_code.py` for full field definitions and types.
- `entities` and `relationships` provide a code graph for downstream analysis.

---

## Integration Notes

- All outputs are regularized and ready for chunking, embedding, or graph construction.
- The JSON serialization module (`serializers/json_serializer.py`) ensures all outputs include versioning and timestamps.
- The module is fully type-checked with `mypy` and covered by unit tests (see `tests/docproc/`).

---

## Testing & Type Safety

- 100% unit test coverage for serialization logic.
- All public functions and adapters are type-checked with `mypy`.
- Run tests with `pytest tests/docproc/`.

---

## Extending the Module

- Add new adapters in `adapters/` for new formats.
- Extend schemas in `schemas/` to support new document types.
- Use the serialization utilities to ensure all outputs remain consistent.

---

## References

- See also: `src/docproc/adapters/README.md` for adapter-specific details.
- For schema details, review `schemas/base.py` and `schemas/python_document.py`.

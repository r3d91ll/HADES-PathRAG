# HADES-PathRAG Chunking Module

This module provides document chunking capabilities for the HADES-PathRAG system, supporting both code and text documents.

## Overview

The chunking module contains two main chunking systems:

1. **AST-based chunking** for code files (`code_chunkers/`)
   - Preserves code structure using Abstract Syntax Tree analysis
   - Language-aware chunking for Python, JavaScript, Java, and C++

2. **Chonky semantic chunking** for text files (`text_chunkers/`)
   - Neural-based semantic chunking that respects content boundaries
   - Preserves original text formatting and casing
   - Supports overlap context for better retrieval

## Key Features

- **Semantic Boundaries**: Identifies natural paragraph and section boundaries
- **Original Text Preservation**: Maintains original casing, formatting, and special characters
- **Overlap Context**: Stores context before and after each chunk for better retrieval
- **Content Hashing**: Efficiently stores and retrieves chunks using content hashing
- **Configurable**: Highly customizable through YAML configuration

## Usage

```python
from src.chunking.text_chunkers.chonky_chunker import chunk_text
from src.chunking.code_chunkers.ast_chunker import chunk_python_code

# Chunk a text document
text_document = {
    "content": "This is a sample text document...",
    "path": "/path/to/document.md",
    "type": "markdown"
}
text_chunks = chunk_text(text_document)

# Chunk a code document
code_document = {
    "content": "def example():\n    pass",
    "path": "/path/to/code.py",
    "type": "python"
}
code_chunks = chunk_python_code(code_document)
```

## Configuration

The chunking behavior can be customized through the configuration file at `src/config/chunker_config.yaml`.

Key configuration options include:

- **Model settings**: Model ID, engine, and device
- **Chunking parameters**: Token limits, overlap, and batch size
- **Overlap context settings**: Context storage options
- **Cache settings**: Device-specific caching options
- **Model engine settings**: Availability checks and startup options

For detailed configuration options, see the [Chunker Configuration Guide](../../docs/integration/chunker_configuration.md).

## Testing

The chunking module includes comprehensive tests in the `tests/chunking/` directory:

```bash
# Run all chunking tests
python -m pytest tests/chunking/

# Run specific test file
python -m pytest tests/chunking/test_chonky_original_text.py
```

## Type Safety

The module is fully type-checked with mypy:

```bash
# Run type checking
python -m mypy src/chunking/
```

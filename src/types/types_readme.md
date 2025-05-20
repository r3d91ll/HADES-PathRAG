# Types Module

## Overview

The types module provides centralized type definitions for the entire HADES-PathRAG system. By centralizing type definitions, we achieve:

1. Better type safety through consistent definitions
2. Reduced circular imports
3. Ability to test modules in isolation
4. Clearer documentation of data structures

## Directory Structure

```
src/types/
  ├── __init__.py
  ├── types_readme.md        # This file
  ├── common.py              # Common types used across multiple modules
  ├── pipeline/              # Orchestration pipeline types
  │   ├── __init__.py
  │   ├── queue.py           # Queue and backpressure types
  │   └── worker.py          # Worker pool types
  ├── documents/             # Document and chunk types
  │   ├── __init__.py  
  │   ├── base.py            # Base document types
  │   └── schema.py          # Document schemas
  ├── embedding/             # Embedding-related types
  │   ├── __init__.py
  │   └── vector.py          # Embedding vector types
  └── isne/                  # ISNE-specific types
      ├── __init__.py
      └── models.py          # ISNE model types
```

## Usage Guidelines

1. **Domain-Specific Types**: Place types in the appropriate subdirectory based on the module they primarily relate to
2. **Common Types**: Types used across multiple modules should be placed in `common.py`
3. **Import Structure**: Import types directly from their module, e.g., `from src.types.documents.base import DocumentType`
4. **Type Exports**: Each module should export its types through `__init__.py` for convenience
5. **Documentation**: All types should have docstrings explaining their purpose and structure

## Type Safety

This module supports the team's commitment to type safety:

1. All type definitions include complete type annotations
2. We use TypedDict, Protocol, and structural types appropriately
3. Validation functions are provided where needed
4. All code should pass mypy validation with strict settings

## Integration with Testing

Types defined here support unit testing by allowing test code to easily create valid test instances:

```python
from src.types.documents.schema import DocumentSchema

# Create test document schema
test_doc = DocumentSchema(
    id="test-doc-001",
    title="Test Document",
    content="This is a test document for unit testing",
    metadata={"source": "unit-test"}
)
```

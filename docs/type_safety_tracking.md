# Type Safety Tracking

## Overview

This document tracks the progress of implementing strong type safety in the HADES-PathRAG project using mypy. Each module will be gradually improved to ensure full type safety as we work through the implementation tasks.

## Type Safety Status by Module

| Module | Status | Issues | Notes |
|--------|--------|--------|-------|
| `hades_pathrag/embeddings` | ✅ In Progress | NetworkX type issues (partially resolved), numpy array typing (improved) | Base interface and fallback embedder now use centralized typing module |
| `hades_pathrag/embeddings/adapters` | ✅ In Progress | SentenceTransformer tensor conversion | Added proper type annotations and tensor conversion for embedding adapters |
| `hades_pathrag/graph` | ✅ In Progress | NetworkX typing challenges (fixed with type ignore directives) | Updated with centralized typing module for base interfaces and NetworkX implementation |
| `hades_pathrag/ingestion` | Pending | - | ISNE pipeline will need careful typing with PyTorch Geometric |
| `hades_pathrag/storage` | ✅ In Progress | ArangoDB client typing (fixed with type ignore directives) | Updated base interfaces and ArangoDB implementation with proper typing |
| `hades_pathrag/mcp_server` | Pending | JSON-RPC typing | Need to ensure proper typing for async handlers |
| `hades_pathrag/core` | Pending | - | Core interfaces need to be properly typed first |

## Common Issues & Solutions

### NetworkX Graph Type Parameters

NetworkX's Graph and DiGraph classes cause type errors with mypy because they're generic classes requiring type parameters. Solutions:

1. **Short-term**: Add `# type: ignore[type-arg]` to Graph/DiGraph usage
2. **Medium-term**: Create type stubs for NetworkX in `scripts/type_stubs/`
3. **Long-term**: Properly parameterize all NetworkX types

### NumPy Array Typing

NumPy arrays should use the NDArray type from `numpy.typing`:

```python
from numpy.typing import NDArray
import numpy as np

def process_embeddings(embedding: NDArray[np.float32]) -> NDArray[np.float32]:
    # Process embedding
    return result
```

### Optional Values

Use `Optional[T]` consistently for values that might be None:

```python
from typing import Optional
import numpy as np
from numpy.typing import NDArray

def get_embedding(node_id: str) -> Optional[NDArray[np.float32]]:
    if node_id in self._embeddings:
        return self._embeddings[node_id]
    return None
```

## Milestone Goals

1. ✅ Set up type safety infrastructure (mypy config, scripts)
2. ✅ Create centralized typing module with common type definitions
3. ✅ Fix base embeddings interface
4. ✅ Fix embedding adapters
5. ✅ Fix storage interfaces
6. ✅ Fix ArangoDB implementation
7. ✅ Fix graph interfaces
8. ⬜ Fix MCP server tools
9. ⬜ Fix ISNE pipeline
10. ⬜ Fix integration points

## How to Use Type Checking Tools

Run basic type check on core modules:

```bash
./scripts/enforce_types.py
```

Run type check on specific modules:

```bash
./scripts/enforce_types.py --modules hades_pathrag/embeddings
```

Generate HTML type coverage report:

```bash
./scripts/enforce_types.py --report --report-dir type_report
```

Run original type_check.py script:

```bash
python scripts/type_check.py --modules hades_pathrag/embeddings
```

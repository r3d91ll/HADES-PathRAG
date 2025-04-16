# Type Safety Guide for HADES-PathRAG

## Overview

Type safety is a critical component of HADES-PathRAG development. We use `mypy` for static type checking to catch type-related issues early in the development process.

## Type Safety Requirements

For all code contributions:

1. **All functions must have complete type annotations** for parameters and return values
2. **Class attributes must be properly annotated** when initialized
3. **Type errors must be resolved** before considering any task complete
4. **Run type checks after implementing each subtask**

## Running Type Checks

We've provided a helper script to run type checks:

```bash
# Check the entire project using mypy.ini settings
python scripts/type_check.py

# Check specific modules
python scripts/type_check.py --modules hades_pathrag.embeddings hades_pathrag.mcp_server

# Generate a detailed type report
python scripts/type_check.py --report
```

## Common Type Patterns

### Dependency Imports

When importing potentially missing dependencies:

```python
try:
    from some_package import SomeClass
    PACKAGE_AVAILABLE = True
except ImportError:
    PACKAGE_AVAILABLE = False
    # Create type stub
    class SomeClass:  # type: ignore
        """Stub class for type checking."""
        pass
```

### Working with NumPy

Use `np.ndarray` with proper type parameters:

```python
from typing import Any
import numpy as np

def process_vector(vector: np.ndarray) -> np.ndarray:
    # Process the vector
    return result_vector
```

### Optional Values

Be explicit about optional values:

```python
from typing import Optional

def get_embedding(node_id: str) -> Optional[np.ndarray]:
    """Get embedding for node if it exists."""
    if node_id in self._embeddings:
        return self._embeddings[node_id]
    return None
```

## Common Type Issues and Solutions

1. **Returning Any**: Use explicit return types instead of Any
2. **Missing annotations**: Add type annotations to all function parameters and return values
3. **Unreachable code**: Remove or fix unreachable code blocks
4. **Incompatible types**: Ensure type compatibility, especially when working with libraries

## Type Safety in Tests

While tests can have more relaxed typing requirements, we encourage:

1. Type annotations for test functions
2. Using proper assertion methods that maintain type safety
3. Mocking with appropriate types

## Updating mypy.ini

As we make progress on type safety, we'll gradually remove `ignore_errors` directives from `mypy.ini` and add stricter checking for more modules.

# HADES-PathRAG

Path-based Retrieval Augmented Generation with ISNE (Inductive Shallow Node Embedding) for efficient graph traversal and retrieval.

## Overview

HADES-PathRAG implements a novel approach to path-based retrieval augmented generation using ISNE embeddings. This implementation focuses on:

- Type-safe code with strict mypy compliance
- Efficient graph traversal using structural node embeddings
- Optimized path retrieval for context augmentation

## Features

- **ISNE Embeddings**: Structure-aware node embeddings that capture graph topology
- **PathRAG**: Path-based retrieval mechanism that prioritizes the most relevant paths
- **Type Safety**: Comprehensive type annotations and mypy compliance throughout the codebase
- **Extensible Storage**: Modular storage backends for different graph database systems

## Installation

### Development Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/HADES-PathRAG.git
cd HADES-PathRAG

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install the package in development mode with all dependencies
pip install -e ".[dev]"

# Install type stubs for external dependencies
pip install types-networkx types-PyYAML types-tqdm
```

## Usage

Basic usage example:

```python
from hades_pathrag.embeddings import ISNEEmbedder
from hades_pathrag.storage import NetworkXStorage
from hades_pathrag.pathrag import PathRAG

# Initialize the embedder
embedder = ISNEEmbedder(embedding_dim=128)

# Create a storage backend
storage = NetworkXStorage(graph_path="knowledge_graph.graphml")

# Initialize PathRAG with the embedder and storage
pathrag = PathRAG(embedder=embedder, storage=storage)

# Retrieve paths for a query
results = pathrag.retrieve("How does function X affect component Y?")

# Process results
for path in results:
    print(f"Path score: {path.score}")
    print(f"Path: {' -> '.join(path.nodes)}")
```

## Development

This project follows strict type safety standards:

1. All code must pass mypy type checking with the strict configuration
2. Tests must be written for all components
3. Type stubs must be used for external dependencies

To run type checking:

```bash
mypy hades_pathrag
```

To run tests:

```bash
pytest
```

## License

MIT License

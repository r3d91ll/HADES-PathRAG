# HADES-PathRAG Documentation

This directory contains comprehensive documentation for the HADES-PathRAG project, an enhanced implementation of PathRAG with XnX notation, ArangoDB integration, Ollama support, and the recursive "HADES builds HADES" architecture.

## Documentation Structure

### XnX Notation
- [`/xnx/`](./xnx/) - Documentation for the XnX notation system
  - [Overview](./xnx/overview.md) - Introduction and core concepts
  - [Implementation](./xnx/implementation.md) - Technical implementation details

### Original PathRAG
- [`/original_paper/`](./original_paper/) - Academic research foundation
  - [Research Foundation](./original_paper/research_foundation.md) - Academic papers supporting HADES-PathRAG
  - [PathRAG Implementation](./original_paper/pathrag_implementation.md) - Connecting paper to code

### Integration Guides
- [`/integration/`](./integration/) - Guides for integrating HADES-PathRAG
  - [ArangoDB Setup](./integration/arango_setup.md) - Setting up ArangoDB for HADES
  - [Ollama Setup](./integration/ollama_setup.md) - Setting up Ollama for local LLM inference
  - [HADES Integration](./integration/hades_integration.md) - Meta integration of HADES-PathRAG into HADES
  - [MCP Recursive Implementation](./integration/mcp_recursive_implementation.md) - "HADES builds HADES" pattern

### API Reference
- [`/api/`](./api/) - API documentation
  - [MCP Server API](./api/mcp_server.md) - Comprehensive API reference for the MCP server

### Examples
- [`/examples/`](./examples/) - Usage examples
  - Code snippets
  - Jupyter notebooks
  - Sample applications

## The "HADES Builds HADES" Pattern

A unique feature of this project is the recursive integration pattern where HADES-PathRAG connects to your IDE via MCP, enabling HADES to improve itself - a self-referential architecture we call "HADES builds HADES."

### Recursive Architecture Highlights

```
┌───────────────┐           ┌───────────────┐
│               │  MCP API  │               │
│  HADES System │◄─────────►│  Windsurf IDE │
│               │           │               │
└───────────────┘           └───────────────┘
        │                           │
        │                           │
        ▼                           ▼
┌───────────────┐           ┌───────────────┐
│               │           │               │
│  Self-referential         │  Improvements │
│  Knowledge    │           │  to HADES     │
│               │           │               │
└───────────────┘           └───────────────┘
```

This recursive pattern draws from cutting-edge research on memory-augmented LLMs and computational universality as discussed in our [Research Foundation](./original_paper/research_foundation.md) document.

For step-by-step implementation, see the [MCP Recursive Implementation](./integration/mcp_recursive_implementation.md) guide.

## Key Academic References

This project builds upon several research papers:

1. [PathRAG: Pruning Graph-based Retrieval Augmented Generation with Relational Paths](https://arxiv.org/html/2502.14902v1)
2. [Memory Augmented Large Language Models are Computationally Universal](https://arxiv.org/html/2503.02113v1)
3. [Building Reliable LLM Agents with Model Context Protocol (MCP)](https://modelcontextprotocol.io/tutorials/building-mcp-with-llms)

## Quick Links

- [XnX Notation Overview](./xnx/overview.md)
- [MCP Server API Reference](./api/mcp_server.md)
- [ArangoDB Integration Guide](./integration/arango_setup.md)
- [Ollama Configuration](./integration/ollama_setup.md)
- [HADES Integration Guide](./integration/hades_integration.md)

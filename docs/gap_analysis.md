# HADES-PathRAG Architecture Gap Analysis

This checklist tracks the current gaps between the intended architecture (as described in `docs/mcp_pathrag_architecture.md` and recent planning) and the actual codebase. Check off each item as it is addressed. Update this file as progress is made.

---

## Checklist

### 1. Data Ingestion Pipeline

- [ ] Modular, config-driven ingestion for both text and code
- [ ] Unified pipeline configuration (YAML/TOML) for chunking, loaders, and embedding
- [ ] Orchestration of loading, embedding, and storage
- [ ] Model serving settings (e.g., vLLM endpoint/model) in pipeline config

### 2. Dual Modality Support (Text + Code)

- [ ] End-to-end support for both text and code in ingestion, embedding, storage, and retrieval
- [ ] Tests/validation for dual modality handling

### 3. Storage Layer

- [x] ArangoDB backend for nodes, edges, embeddings, and paths
- [x] Typed edges, robust error handling, path traversal
- [ ] Ensure storage logic is future-proofed for remote embedding/model serving

### 4. Retrieval Engine

- [x] Path-based retrieval, composite scoring, graph traversal
- [ ] Retrieval engine can flexibly use embeddings from local or remote (vLLM) sources
- [ ] Retrieval engine is configurable for embedding source selection

### 5. Model Serving Layer (vLLM or Similar)

- [ ] Support for remote model serving (vLLM, Triton, etc.) for LLM and embedding models
- [ ] Ability to specify model provider, endpoint, and credentials in config
- [ ] Client/adapter for remote inference

### 6. Documentation/Config

- [ ] Unified, up-to-date pipeline configuration
- [ ] Documentation reflects new architecture and model serving options

### 7. De-prioritized Areas

- [ ] MCP Server (present, not a current focus)
- [ ] Git Integration (planned, not implemented)

---

## Instructions

- Use `[x]` to mark items as complete, `[ ]` for incomplete.
- Add notes/dates as progress is made.
- Update this checklist regularly to reflect new gaps or completed work.

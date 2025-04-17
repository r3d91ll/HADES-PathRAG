# HADES-PathRAG Project Status

## Current State

The HADES-PathRAG project has made significant progress in the following areas:

- Core architecture for path-based retrieval and knowledge graph construction is implemented.
- Data ingestion pipeline using ISNE (Inductive Shallow Node Embedding) is complete, with loaders for text, JSON, and CSV.
- ArangoDB integration is robust, tested, and fully documented.
- Ollama is integrated as the default LLM engine for semantic operations.
- Type safety has been systematically applied across embeddings, adapters, storage, and graph modules.
- The next planned step was to improve type safety and functionality in the MCP server tools module.

## On Hold

As of 2025-04-17, active development on HADES-PathRAG is **on hold** while we focus on building a new, independent project:

# HoH-mcp (Hammer of Hephaestus)

HoH-mcp will be a standalone, local MCP server designed to:
- Parse Python codebases using open-source, local tools (e.g., Python's ast, LibCST)
- Extract a full symbol table and code object relationships (functions, classes, files, etc.)
- Output structured MCP objects for ingestion into knowledge graphs like ArangoDB
- Run as a service for integration with IDEs, CI/CD, and other tools
- Ensure privacy and security by never sending code to the cloud

## Next Steps

1. Scaffold a new repository for HoH-mcp.
2. Develop and document the MCP abstraction and parsing logic.
3. Integrate the service with HADES-PathRAG's ingestion pipeline once complete.

---

**Note:** All progress and plans for HADES-PathRAG will resume after HoH-mcp reaches a functional milestone.

For questions or collaboration, please refer to this document or contact the project maintainers.

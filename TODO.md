# HADES-PathRAG Project TODO List

## 🚨 Priority Tasks (May 2025) - Updated: April 29th

### Project Setup & Environment

- [x] Fix Python environment setup with Poetry (requires Python 3.10+)
- [x] Update `pyproject.toml` to properly include all direct and transitive dependencies
- [x] Create reliable development setup script that works across environments
- [x] Clean up project directory structure (moved obsolete docs to old-docs/)
- [x] Fix failing tests and test infrastructure
  - ✅ Fixed ArangoConnection initialization issues with backward compatibility
  - ✅ Improved DoclingPreProcessor test coverage to >85%
  - ✅ Fixed mocking in ChonkyProcessor tests
  - ✅ Enhanced error handling in PythonPreProcessor
- [ ] Implement ArangoDB collection management modes
  - [ ] Add create/initialize mode to recreate collections and graph
  - [ ] Add append mode to add documents to existing structures
  - [ ] Update CLI with options to control database initialization behavior

### Model Engine Integration

#### Phase 1: Foundation & Embedding (Current Priority)

- [x] Add vLLM to project dependencies in pyproject.toml
- [x] Install and configure vLLM locally (verified with `vllm -v` shows 0.8.4)
- [x] Create unified model engine architecture
  - [x] Implement abstract base classes for adapter pattern
  - [x] Create model-agnostic configuration system
  - [x] Refactor `src/isne/adapters/vllm_adapter.py` → `src/model_engine/adapters/vllm_adapter.py`
  - [x] Refactor `src/pathrag/vllm_adapter.py` → `src/pathrag/model_engine_adapter.py`
  - [x] Add server lifecycle management with `ServerManager`
- [x] Update embedding processor to use model_engine
- [x] Create model configuration files
  - [x] Create `src/config/model_config.py` with Pydantic models
  - [x] Create `src/config/model_config.yaml` with specialized models for different tasks
- [x] Create startup helper script `scripts/start_vllm_server.py`
- [x] Add test suite for VLLMAdapter
- [ ] Add missing components
  - [ ] Add systemd / Docker compose example to docs
  - [ ] Create helpers for non-vLLM backends (Ollama, HuggingFace)
- [ ] Benchmark embedding performance with different backends
- [ ] Document model engine setup in `docs/integration/model_engine.md`

#### Phase 2: Advanced LLM Integration

- [x] Create model engine adapter compatible with PathRAG interfaces
- [x] Support multiple specialized models:
  - [x] Code-specific embedding model
  - [x] General document embedding model
  - [x] Chunking model
  - [x] Relationship extraction model
  - [x] Different models for inference (default, code, pathfinder)
- [ ] Add auto-detection of available GPU resources
- [ ] Implement model caching and memory optimization
- [ ] Update example scripts to use model_engine
- [ ] Add dynamic model switching based on content type
- [ ] Integrate vLLM with the FastAPI interface

### ISNE Pipeline Enhancement

- [x] Complete ISNE integration with ingestion pipeline
- [x] Implement path ranking according to PathRAG algorithm (70% semantic, 10% path length, 20% edge strength)
- [x] Optimize embedding generation for code chunks
- [x] Implement specialized embedding for inter-file relationships
- [ ] Train or fine-tune graph neural networks for code representation
- [ ] Scale testing with larger codebases

### Type Safety Implementation

- [x] Create base type definitions for core system components
- [x] Implement type safety for FastAPI interface
  - [x] Added Pydantic models for requests and responses
  - [x] Fixed function signatures with proper return types
  - [x] Created typed core system interface
- [x] Implement type safety for pre-processor module
  - [x] Created typed data models with TypedDict and dataclasses
  - [x] Added proper return type annotations for all functions
  - [x] Implemented type-safe file operations utilities
  - [x] Built a comprehensive API with proper typing
  - [x] Created centralized types module for the pre-processor
  - [x] Fixed type errors in pre-processor registry with Union types
  - [x] Properly type-annotated adapter factory functions
- [x] Update ISNE pipeline with proper types
  - [x] Created comprehensive type-safe data models with dataclasses
  - [x] Implemented neural network layers with proper tensor typing
  - [x] Built type-safe document loaders for various data sources
  - [x] Added processors with strict typing (chunking, embedding, graph)
  - [x] Created a fully typed pipeline orchestration interface
- [ ] Fix integration points with consistent type definitions
  - [x] Fixed PreprocessorManager typed interface
  - [x] Ensured proper typing for adapter pattern classes
  - [ ] Fix remaining docproc adapter type errors (csv, json, html adapters)
- [ ] Add mypy configuration and pre-commit hooks

### Code Pre-Processing Pipeline

- [x] Implement `.symbol_table` directory structure for source code metadata
- [x] Create comprehensive documentation for ingestion system
- [x] Refactor document ingestion pipeline with format-aware adapters
  - [x] Replace legacy pre-processor logic with new docproc module adapters
  - [x] Create robust adapter layer for backward compatibility
  - [x] Implement format-aware processing for multiple file types (PDF, HTML, code, etc.)
  - [x] Update tests with proper mocking and fixtures
  - [x] Fix type errors in pre-processor module
- [ ] Integrate pre-processor with main ingestion pipeline
  - [x] Ensure compatibility with Chonky implementation
  - [x] Prioritize text content processing with Chonky
  - [ ] Complete full integration with semantic embedding and graph embedding phase
  - [ ] Finalize pipeline with RepositoryIngestor after all components are ready
- [ ] Inter-file Relationship Detection & Storage (Foundational)
  - **Phase 0 – Design**
    - [ ] Define edge types (IMPORTS, USES, READS_FILE, WRITES_FILE, etc.)
    - [ ] Select AST analysis library (`astroid` preferred)
    - [ ] Update `.symbol_table` JSON schema for new edge lists
  - **Phase 1 – Implementation**
    - [ ] Implement `ASTRelationExtractor` visitor class
    - [ ] Add `collect_interfile_relations` flag to `PythonPreProcessor`
    - [ ] Append extracted edges to pre-processing output
  - **Phase 2 – Testing**
    - [ ] Create mini-repo fixture (`a.py` imports `b.py`, writes a file)
    - [ ] Unit test that `IMPORTS` & `WRITES_FILE` relations are produced
  - **Phase 3 – Pipeline Integration**
    - [ ] Ensure PreprocessorManager merges new relations
    - [ ] Verify edges persisted to Arango via Repository layer
    - [ ] Document workflow in `docs/ingestion_system.md`
- [x] Implement Chonky-based semantic chunking (see Chonky implementation section)
- [ ] Build graph construction module to prepare for ISNE embedding

### Ingestion Pipeline Optimization

- [x] Modularize pre-processor directory by file type:
  - [x] Extract current code to `python_pre_processor.py`
  - [x] Create interfaces for docling pre-processor
- [ ] Create centralized file batching system (`file_batcher.py`) to handle directory traversal once
- [ ] Implement parallel processing via ThreadPoolExecutor for different file type batches
- [ ] Add support for website documentation pre-processing
- [ ] Add support for PDF document pre-processing
- [ ] Update ingestor.py to orchestrate parallel pre-processing and ingestion

### API Implementation

- [x] Design and implement FastAPI interface (replacing MCP server)
  - [x] Create core models for requests and responses
  - [x] Implement minimal endpoints (write, query, status)
  - [x] Document API interface and usage
- [ ] Connect API to ArangoDB storage backend
- [ ] Implement comprehensive error handling
- [ ] Add authentication mechanism
- [ ] Create client library for API interaction

### 🔧 Infrastructure Hardening & Logging (May 2025)

- [x] **ArangoDB Collection & Key Sanitisation**
  - Context: Ensure collection names and `_key` values meet ArangoDB naming rules to prevent ingestion failures and enable CLI recreate/append modes.
  - Deliverables:
    - ✅ `src/storage/arango/utils.py` with `safe_name` and `safe_key`
    - ✅ Unit tests in `tests/storage/test_utils.py`
    - ✅ Integration into `ArangoStorage` creation and insert logic
  - Acceptance Criteria:
    - ✅ Ingestion pipeline creates collections/documents with arbitrary names without errors.

- [x] **Ingestion Pipeline Test Hardening**
  - Context: Ensure all ingestion pipeline components have robust tests with high coverage, especially after ArangoDB API refactoring.
  - Deliverables:
    - ✅ Updated all repository tests to use new type-safe ArangoDB API
    - ✅ Fixed cursor mocking and parameter extraction in tests
    - ✅ Ensured proper error handling and type safety in tests
    - ✅ Verified high test coverage (93% for repository code)
  - Acceptance Criteria:
    - ✅ All 179 ingestion pipeline tests pass
    - ✅ Repository code has >90% test coverage
    - ✅ Tests properly use the new type-safe API

- [x] **Type Safety Improvements**
  - Context: Fix type errors in the ingestion pipeline to ensure robust type checking with mypy.
  - Deliverables:
    - ✅ Fixed type errors in docling_adapter.py and docling_pre_processor.py
    - ✅ Improved handling of optional dependencies with proper type annotations
    - ✅ Added proper type annotations for BeautifulSoup elements
    - ✅ Updated tests to match implementation changes
  - Acceptance Criteria:
    - ✅ All ingestion pipeline files pass mypy type checking
    - ✅ All 179 ingestion pipeline tests pass
    - ✅ Optional dependencies are properly handled with graceful degradation

- [ ] **Sharding & Indexing Strategy**
  - Context: Current collections lack optimal shard keys and indexes, causing potential performance bottlenecks for large graphs.
  - Deliverables:
    - Config-driven shard/index parameters
    - Automatic index creation (`symbol_path`, embeddings, etc.)
    - Query profiling script `scripts/profile_queries.py`
  - Acceptance Criteria:
    - Explain plans show use of defined indexes; ingestion throughput improves (baseline-to-target KPI TBD).

- [ ] **Robust Logging & Error Handling**
  - Context: Missing structured logs and retries make debugging ingestion difficult.
  - Deliverables:
    - `src/utils/logging.py` using `structlog`
    - Replace `print`/bare `except` with structured logs
    - `tenacity` retry decorator for DB writes/network I/O
  - Acceptance Criteria:
    - Errors surface with stack traces & context; transient errors auto-retry.

- [x] **Arango Connection Refactor**
  - Context: Replace legacy `src.db.arango_connection` with typed `src.storage.arango.connection`.
  - Deliverables:
    - New connection wrapper (done)
    - Update all imports, delete `src/db`
    - Ensure tests pass after migration
  - Acceptance Criteria:
    - `grep -R "src.db.arango_connection"` returns no results in src/ (done - only references in dead-code/)

- [✅] **AST-Based Code Chunker**
  - Context: Deterministic chunking for source code to complement Chonky (semantic text chunker).
  - Deliverables:
    - ✅ Created `src/ingest/chunking/code_chunkers/ast_chunker.py` with `chunk_python_code()`
    - ✅ Created `src/ingest/chunking/code_chunkers/__init__.py` with language dispatcher  
    - ✅ Added `src/config/chunker_config.py` and YAML for configurable chunking
    - ✅ Integrated chunker with `PreprocessorManager.extract_entities_and_relationships()`
    - ✅ Moved `PreprocessorManager` from `processing/` to `pre_processor/manager.py` for better organization
    - ✅ Created basic integration test in `tests/test_ast_chunker.py`
    - ✅ Moved chunker implementation to `src/ingest/chunking/code_chunkers` and deprecated `src/ingest/processing`
    - ✅ Updated all imports & tests to new package
    - [ ] Phase-out compatibility shim once downstream code updated
  - Acceptance Criteria:
    - Large functions/classes split into ≤ 2048-token chunks; pipeline ingests Python repo successfully.

- [x] **Chonky-Based Text Chunker**
  - Context: Handle markdown / plain-text files with semantic paragraph splitting.
  - Deliverables:
    - ✅ Implement `chunk_text()` in `src/ingest/chunking/text_chunkers/chonky_chunker.py`
    - ✅ Extend dispatcher in `chunking/__init__.py` for `markdown` & `text`
    - ✅ Unit/integration tests using sample docs
  - Acceptance Criteria:
    - ✅ Non-code docs are split into ≤ 2048-token chunks and ingested without errors.

- [-] **Embedding Layer Refactor**
  - Context: Separate embedding concerns into `src/ingest/embedding/`.
  - Deliverables:
    - ISNE façade function (`embed_graph_with_isne`) implemented (done)
    - Flag in orchestrator to run graph embeddings automatically
    - Future sub-modules for text/vector store embeddings
  - Acceptance Criteria:
    - Orchestrator can run end-to-end ingestion + ISNE embedding with single flag.

### CLI Tools Reorganization

- [x] **CLI Tools Reorganization**
  - Context: Reorganize scripts into proper CLI structure with consistent interfaces
  - Deliverables:
    - ✅ Core implementation in `src/cli/` modules
    - ✅ Simple executable scripts in `scripts/pathrag-*`
    - ✅ Type-safe interfaces with proper error handling
    - ✅ Consistent parameter naming across tools
  - Acceptance Criteria:
    - ✅ All CLI tools pass mypy type checking
    - ✅ CLI tools support explicit collection management modes

## ✅ Completed Tasks

### Project Structure

- ✅ Migrated HoH_parser to src/ingest/pre_processor
- ✅ Updated package structure in pyproject.toml
- ✅ Merged requirements for pre-processor module
- ✅ Consolidated documentation (merged ingestion files, removed duplication)
- ✅ Created unified README structure with clear navigation

### ArangoDB Integration

- ✅ Fixed ArangoConnection import issue in arango_adapter.py
- ✅ Completed ArangoDB adapter implementation for XnX PathRAG
- ✅ Added tests for ArangoDB integration
- ✅ Created example script showing ArangoDB usage with PathRAG
- ✅ Documented ArangoDB setup and configuration in docs

### XnX Implementation

- ✅ Implemented basic XnX traversal functions as documented
- ✅ Added error handling to XnX traversal functions
- ✅ Integrated XnX traversal with ArangoDB adapter
- ✅ Set up test suite for XnX traversal validation
- ✅ Created detailed XnX guide at `docs/xnx/XnX_README.md`
- ✅ Documented XnX notation format and constraints

### API Implementation (Replacing MCP Server)

- ✅ Created FastAPI implementation with core endpoints
- ✅ Implemented Pydantic models for type-safe requests/responses
- ✅ Documented API interface in docs/api.md
- ✅ Created CLI interface for the API server

### Documentation Improvements

- ✅ Created comprehensive ingestion_system.md documentation
- ✅ Consolidated duplicate documentation files
- ✅ Moved obsolete documentation to old-docs directory
- ✅ Added cross-references between related documentation files
- ✅ Updated project README with accurate HADES definition

### Environment & Structure Updates (April 24th)

- ✅ Updated Python version requirement to >=3.10,<4.0 in pyproject.toml
- ✅ Organized dependencies into logical categories (core, database, LLM clients, data processing)
- ✅ Replaced non-existent 'nan-owl' package with 'nano-vectordb'
- ✅ Enhanced setup_dev_env.sh with better Python version validation
- ✅ Added development dependencies for testing async code and type checking
- ✅ Added FastAPI and related dependencies to requirements.txt
- ✅ Removed deprecated MCP server components

## 📋 Future Tasks

### Document Processing Improvements

- [x] Create modular symbol table implementation for Python code documents
- [ ] Complete Docling pre-processor improvements

### Chonky Implementation for Semantic Chunking

- [x] Document chunking strategies in chunking.md
- [x] Implement semantic chunking using Chonky
  - [x] Create `chonking_processor.py` as implementation of Chonky chunker
  - [x] Update pipeline to support Chonky for non-code content
  - [x] Create testing framework to compare chunking approaches
  - [x] Integrate with vLLM for embedding generation

### Future Chunking Work (After Chonky Implementation)

- [ ] Create modular symbol table implementation for markdown documents
  - [ ] Design document-oriented structure focusing on headers, sections, and references
  - [ ] Support named anchors and cross-document linking
  - [ ] Ensure compatibility with different modalities (text, code, diagrams)
- [ ] Implement code-aware chunking using symbol tables (separate from Chonky)
- [ ] Create configurable chunking parameters

- [ ] Refactor to use centralized configuration system
  - [ ] Unify configuration across all HADES-PathRAG components
  - [ ] Implement hierarchical configuration with inheritance
  - [ ] Add validation and schema checking

### Performance Optimization

- [ ] Benchmark Python implementation of XnX traversal
- [ ] Identify critical paths for Mojo migration
- [ ] Create Mojo implementations of core XnX algorithms
- [ ] Implement parallel processing for path evaluations
- [ ] Develop caching strategy for frequently accessed paths

### Model Integration

- [ ] Implement embedding acceleration (see vLLM Integration section)
- [ ] Implement GNN for graph traversal operations
- [ ] Set up domain detection for model selection
- [ ] Configure code-specific model (Qwen2.5-coder)
- [ ] Configure general-purpose model (Llama3)
- [ ] Develop model switching framework

### Visualization & Error Handling

- [ ] Create web interface for HADES-PathRAG
- [ ] Implement visualization for knowledge graph exploration
- [ ] Add visualization of path traversal and ranking
- [ ] Create debugging visualizations for traversal analysis
- [ ] Create fallback strategies for common error scenarios
- [ ] Add detailed logging for traversal operations

## 🔍 Project Review Notes

- Decision made to replace Ollama with vLLM for better hardware utilization, API compatibility, and model selection
- Decision made to implement Chonky for semantic chunking of non-code content to improve retrieval quality
- Plan to create unified embedding service using vLLM for both Chonky and ISNE
- Phased approach: 1) Implement Chonky, 2) Integrate with pipeline, 3) Set up vLLM, 4) Testing
- Decision made to replace MCP server with a simpler FastAPI implementation for easier maintenance and integration
- Migrated pre-processor code needs integration with the main ingestion pipeline
- Documentation has been consolidated and improved for better clarity and navigation
- Type safety implementation is progressing well with focus on core components and API interfaces
- Project environment setup is stable but test infrastructure needs improvement
- XnX notation has been moved to experimental features while core functionality is prioritized

Last updated: April 29, 2025

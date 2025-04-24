# HADES-PathRAG Project TODO List

## 🚨 Priority Tasks (April 2025) - Updated: April 24th

### Project Setup & Environment

- [x] Fix Python environment setup with Poetry (requires Python 3.10+)
- [x] Update `pyproject.toml` to properly include all direct and transitive dependencies
- [x] Create reliable development setup script that works across environments
- [x] Clean up project directory structure (moved obsolete docs to old-docs/)
- [ ] Fix failing tests and test infrastructure

### vLLM Integration (Replacing Ollama)

- [x] Add vLLM to project dependencies in pyproject.toml
- [ ] Install and configure vLLM locally (requires separate installation via setup_dev_env.sh)
- [ ] Create vLLM adapter compatible with existing PathRAG interfaces
- [ ] Update API clients to use vLLM's OpenAI-compatible endpoints
- [ ] Ensure backward compatibility with existing code
- [ ] Document vLLM setup and configuration in `docs/integration/vllm_setup.md`
- [ ] Update example scripts to use vLLM instead of Ollama
- [ ] Integrate vLLM with the FastAPI interface

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
- [x] Update ISNE pipeline with proper types
  - [x] Created comprehensive type-safe data models with dataclasses
  - [x] Implemented neural network layers with proper tensor typing
  - [x] Built type-safe document loaders for various data sources
  - [x] Added processors with strict typing (chunking, embedding, graph)
  - [x] Created a fully typed pipeline orchestration interface
- [ ] Fix integration points with consistent type definitions
- [ ] Add mypy configuration and pre-commit hooks

### Code Pre-Processing Pipeline

- [x] Implement `.symbol_table` directory structure for source code metadata
- [x] Create comprehensive documentation for ingestion system
- [ ] Integrate pre-processor with main ingestion pipeline
- [ ] Create inter-file relationship detection and storage
- [ ] Implement the hybrid chunking processor
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

### ISNE Pipeline Enhancement

- [ ] Complete ISNE integration with ingestion pipeline
- [ ] Implement path ranking according to PathRAG algorithm (70% semantic, 10% path length, 20% edge strength)
- [ ] Optimize embedding generation for code chunks
- [ ] Implement specialized embedding for inter-file relationships
- [ ] Train or fine-tune graph neural networks for code representation
- [ ] Scale testing with larger codebases

### Document Processing Improvements

- [x] Create modular symbol table implementation for Python code documents
- [ ] Complete Docling pre-processor improvements
- [ ] Implement Chonky-based neural chunking for text documents
- [ ] Create modular symbol table implementation for markdown documents
  - [ ] Design document-oriented structure focusing on headers, sections, and references
  - [ ] Support named anchors and cross-document linking
  - [ ] Ensure compatibility with different modalities (text, code, diagrams)
- [ ] Implement fully-featured HybridChunkingProcessor
  - [x] Document chunking strategies in chunking.md
  - [ ] Implement code-aware chunking using symbol tables
  - [ ] Implement semantic chunking using Chonky
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
- Decision made to replace MCP server with a simpler FastAPI implementation for easier maintenance and integration
- Migrated pre-processor code needs integration with the main ingestion pipeline
- Documentation has been consolidated and improved for better clarity and navigation
- Type safety implementation is progressing well with focus on core components and API interfaces
- Project environment setup is stable but test infrastructure needs improvement
- XnX notation has been moved to experimental features while core functionality is prioritized

Last updated: April 24, 2025

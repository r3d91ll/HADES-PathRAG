# HADES-PathRAG: Pre-Embedding Model Implementation Tasks

This document outlines the critical tasks that must be completed before proceeding with embedding model implementation. Each section contains detailed action items, implementation steps, and validation criteria.

## 1. JSON Object Schema Standardization

### JSONPrimary Tasks

- [x] Define comprehensive JSON schema using Pydantic
- [x] Implement validation checkpoints throughout the pipeline
- [x] Create schema versioning mechanism

### Implementation Steps for Schema

1. ✅ Create dedicated schema module in `src/schema/document_schema.py`
2. ✅ Define base document schema with required fields
3. ✅ Create specialized schemas for different document types (code, text, etc.)
4. ✅ Add validation utilities for schema verification
5. ✅ Implement automatic schema version upgrades

### JSON Schema Current Status

- Implemented `DocumentSchema`, `DocumentRelationSchema`, `DatasetSchema`, and `ChunkMetadata` Pydantic models
- Updated to use Pydantic v2 configuration exclusively, replacing deprecated features
- Created validation module with functions for validating documents and datasets
- Implemented `ValidationCheckpoint` decorator for pipeline validation
- Added `upgrade_schema_version` function for backward compatibility
- Unit tests for schema and validation modules created

### JSON Schema Completed Tasks

- ✅ Complete and finalize unit tests for schema and validation modules
  - ✅ Created test_document_schema.py to test all schema models
  - ✅ Finalized test_validation.py with tests for all validation functions
  - ✅ Added test_validation_extended.py with additional validation tests
- ✅ Run mypy on schema modules to check for type errors
  - ✅ Ensured all Pydantic models have proper type annotations
  - ✅ Fixed type errors identified by mypy
- ✅ Run test suite with coverage to ensure ≥85% coverage
  - ✅ Achieved 93% test coverage for schema and chunking modules
  - ✅ Added tests for edge cases and error handling
- ✅ Integrated chunking module with schema validation
  - ✅ Fixed MockHaystackModelEngine in integration tests
  - ✅ Added proper type annotations to chunking modules
  - ✅ Created integration tests for chunking and validation

### Remaining Tasks

- ✅ Integrate validation checkpoints into the ingestion pipeline
  - ✅ Added validation to document processing pipeline in src/docproc/core.py
  - ✅ Implemented schema validation in the document processing workflow

## 7. Type Safety Implementation

### Primary Tasks

- [ ] Complete type safety implementation across the codebase following the roadmap
- [ ] Fix all mypy errors in key modules
- [ ] Ensure consistent type annotations throughout the codebase

### Implementation Steps for Type Safety

1. ✅ Create centralized typing module with common type definitions
2. ✅ Fix base embeddings interface
3. ✅ Fix embedding adapters
4. ✅ Fix storage interfaces
5. ✅ Fix ArangoDB implementation
6. ✅ Fix graph interfaces
7. ✅ Fix Haystack engine module type safety issues
8. [ ] Fix ISNE pipeline
9. [ ] Fix integration points

### Type SafetyCurrent Status

- Completed type safety implementation for the Haystack engine module
- Fixed all mypy errors in Haystack client, server, and engine classes
- Added proper type annotations for all functions in the module
- Improved error handling to maintain consistent state

### Type SafetyCompleted Tasks

- ✅ Fix type safety issues in src/model_engine/engines/haystack/__init__.py
  - ✅ Added proper type annotations for all methods
  - ✅ Fixed unreachable code errors identified by mypy
  - ✅ Improved class attribute type definitions
  - ✅ Enhanced error handling with proper type checking
- ✅ Fix type safety issues in src/model_engine/engines/haystack/runtime/__init__.py
  - ✅ Added missing imports from typing module
  - ✅ Fixed function parameter type annotations
  - ✅ Added proper handling of Optional parameters
- ✅ Fix type safety issues in src/model_engine/engines/haystack/runtime/server.py
  - ✅ Added type annotations for all functions
  - ✅ Fixed unreachable code in _get_model_info function
  - ✅ Refactored code to use helper functions for better type safety
  - ✅ Enhanced model info structure for consistent typing

### Remaining Type Safety Tasks

- [ ] Fix MCP server tools module typing issues
- [ ] Complete ISNE pipeline type safety implementation
- [ ] Address type safety in integration points between modules
  - ✅ Added error handling for validation failures with proper logging

### Validation Criteria for Schema

- All schema classes have complete type annotations
- Schema validation catches malformed documents
- Unit tests cover various document scenarios
- Schema versioning handles backward compatibility

## 2. Chunking System Validation

### Completed Chunking Tasks

- ✅ Create test suite for chunking validation
  - ✅ Implemented comprehensive tests for AST code chunker
  - ✅ Implemented comprehensive tests for Chonky text chunker
  - ✅ Created integration tests with schema validation
- ✅ Implement different chunking strategies
  - ✅ Implemented code-aware chunking via AST chunker
  - ✅ Implemented semantic chunking via Chonky chunker
- ✅ Enhance Chonky chunker configuration
  - ✅ Added overlap context structure to preserve content surrounding chunks
  - ✅ Implemented device-specific caching to prevent collisions
  - ✅ Added early availability checks for model engine
  - ✅ Created comprehensive documentation in chunker_configuration.md
- ✅ Achieved >85% test coverage for chunking module (currently at 93%)

### Remaining Chunking Tasks

- [x] Validate chunking logic across all supported document types
  - [x] Add tests for PDF documents via Docling
  - [x] Add tests for TXT documents
  - [x] Add tests for CSV, XML, JSON, YAML, and TOML documents
- [ ] Ensure semantic coherence in chunks
  - [ ] Implement metrics for semantic coherence using Chonky
  - [ ] Add tests that verify semantic integrity of chunks
- [x] Implement chunk boundary verification
  - [x] Add validation for context preservation in text chunks
  - [x] Add validation for function/class boundary preservation in code chunks
  - [x] Create tests for boundary edge cases
- [x] Fix type errors and unreachable statements in chunking modules
  - [x] Fix unreachable statements in get_model_engine function
  - [x] Add proper type checking for client attributes
  - [x] Update Pydantic model handling to use model_dump() with fallback
  - [x] Fix client.ping() None reference errors
  - [x] Resolve type incompatibilities in dict returns
  - [x] Ensure all mypy checks pass with no errors
  - [x] Improve error handling with structured helper functions
- [ ] Create visualization tools for chunk analysis
  - [ ] Implement chunk distribution visualizer for token distribution
  - [ ] Create chunk overlap visualization tool for semantic overlap

### Chunking System Recent Updates

- ✅ Enhanced Chonky integration with Haystack model engine
  - ✅ Fixed cache key management to ensure proper model loading
  - ✅ Improved error handling for edge cases with missing models
  - ✅ Added additional logging to track model loading status
- ✅ Optimized resource utilization in chunking system
  - ✅ Implemented caching to prevent redundant model loading
  - ✅ Added proper cleanup of resources when models are no longer needed
- ✅ Enhanced documentation of chunking system architecture
  - ✅ Added detailed explanations of chunking strategy selection

### Validation Criteria for Chunking

- Chunks maintain semantic coherence
- Code chunks preserve function/class boundaries
- Chunk size distribution follows expected patterns
- Chunk overlap is consistent and appropriate
- Test coverage ≥ 85% for chunking module

## 3. Metadata Enhancement and Standardization

### Metadata Standardization Primary Tasks

- [ ] Define comprehensive metadata schema
- [ ] Ensure source document linkage in all chunks
- [ ] Add position tracking (offsets) to chunks
- [ ] Include model information in metadata

### Implementation Steps for Metadata

1. Create metadata standardization module in `src/schema/metadata.py`
2. Implement metadata validators and enrichers
3. Add source tracking with unique identifiers
4. Create utilities for metadata extraction and verification

### Validation Criteria for Metadata

- All chunks have complete metadata
- Source document is always traceable
- Position information is accurate
- Metadata format is consistent across document types

## 4. Model Management Infrastructure

### Model Management Primary Tasks

- [ ] Create model loading abstraction layer
- [ ] Implement model caching mechanism
- [ ] Develop model version tracking
- [ ] Add resource monitoring for model usage

### Implementation Steps for Model Management

1. Design model manager interface in `src/model_engine/model_manager.py`
2. Implement Haystack-based model loader
3. Add caching with configurable memory limits
4. Create model registry for version tracking
5. Implement resource monitoring hooks

### Validation Criteria for Model Management

- Models load and unload efficiently
- Caching reduces redundant model loading
- Resource usage stays within defined limits
- Model versioning prevents inconsistencies
- Unit tests verify model management behavior

### Model Management Recent Updates

- ✅ Implemented ModernBERT embedding adapter for CPU inference
  - ✅ Created direct HuggingFace transformers integration without requiring GPU
  - ✅ Added configuration support via embedding_config.yaml
  - ✅ Implemented multiple pooling strategies (cls, mean, max)
  - ✅ Added batch processing to optimize throughput
  - ✅ Enabled proper CPU/GPU detection and configuration
- ✅ Enhanced Haystack model engine integration
  - ✅ Fixed model caching issues for embedding generation
  - ✅ Improved error handling for model loading failures
  - ✅ Added detailed diagnostics for troubleshooting model loading issues

## 5. Pipeline Logging and Monitoring System

### Logging and Monitoring Primary Tasks

- [ ] Implement structured logging framework
- [ ] Add performance monitoring hooks
- [ ] Create pipeline stage tracking
- [ ] Develop error aggregation system

### Implementation Steps for Logging

1. Configure structured logging in `src/utils/logging.py`
2. Add performance monitoring decorators
3. Implement pipeline stage tracking
4. Create error collection and reporting mechanism
5. Add visualization tools for log analysis

### Validation Criteria for Logging

- Logs capture all critical pipeline operations
- Performance metrics are consistently tracked
- Error reporting provides actionable information
- Log levels are appropriately used

## 6. Performance Benchmarking Framework

### Benchmarking Primary Tasks

- [ ] Develop benchmarking framework
- [ ] Identify and measure pipeline bottlenecks
- [ ] Create performance baseline
- [ ] Implement optimization strategies

### Implementation Steps for Benchmarking

1. Create benchmarking utilities in `src/utils/benchmarking.py`
2. Define key performance indicators (KPIs)
3. Implement measurement points throughout pipeline
4. Develop reporting mechanism for performance results
5. Create optimization plan based on findings

### Validation Criteria for Benchmarking

- Benchmark results are reproducible
- All pipeline stages have performance metrics
- Bottlenecks are clearly identified
- Optimization priorities are established

## 7. ArangoDB Integration Enhancement

### ArangoDB Integration Primary Tasks

- [ ] Define vector field schema for ArangoDB
- [ ] Implement vector indexing strategy
- [ ] Create collection management tools
- [ ] Develop query utilities for vector operations

### Implementation Steps for ArangoDB

1. Extend ArangoDB adapter in `src/db/arango_adapter.py`
2. Define vector field specifications
3. Implement index creation for vector fields
4. Create utilities for vector operations
5. Add test suite for vector operations

### Validation Criteria for ArangoDB

- Vector fields are properly defined
- Indexes support efficient vector operations
- Collection management handles vector data
- Query performance meets requirements
- Test coverage ≥ 85% for ArangoDB vector operations

## 8. Documentation and Knowledge Management

### Documentation Primary Tasks

- [ ] Update pipeline documentation
- [ ] Document JSON schema and validation process
- [ ] Create embedding strategy documentation
- [ ] Update developer guidelines

### Implementation Steps for Documentation

1. Update README.md with current architecture
2. Create dedicated documentation for embedding pipeline
3. Document JSON schema with examples
4. Create embedding strategy guide
5. Update developer onboarding documentation

### Validation Criteria for Documentation

- Documentation is comprehensive and accurate
- Examples demonstrate key concepts
- Schema documentation includes all fields
- Developer guidelines facilitate onboarding

## 8. GPU Pipeline Test Coverage

### GPU Pipeline Test Coverage Tasks

- [x] Enhance test coverage for GPU-orchestrated pipeline
  - [x] Improve DocProcStage test coverage (now at 91%)
  - [x] Improve ChunkStage test coverage (now at 93%)
  - [x] Ensure StageBase has 100% test coverage
  - [x] Maintain orchestrator test coverage above 85% (currently at 87%)
- [x] Fix test issues and edge cases
  - [x] Fix tests for document processing with unsupported formats
  - [x] Add tests for timeout handling in document processing
  - [x] Add tests for invalid results from document processing
  - [x] Add tests for error handling in chunking stage
- [x] Address deprecation warnings
  - [x] Update Pydantic V1 style validators to V2 in engine_config.py
  - [x] Replace class Config with ConfigDict
  - [x] Update parse_obj() to model_validate()

### Implementation Steps for GPU Pipeline Testing

1. ✅ Identify modules with insufficient coverage
2. ✅ Create targeted unit tests for uncovered code
3. ✅ Implement tests for error handling and edge cases
4. ✅ Fix deprecation warnings in configuration modules
5. ✅ Verify overall test coverage meets or exceeds 85%

### Validation Criteria for GPU Pipeline Testing

- All critical paths have dedicated tests
- Edge cases are properly tested
- Type safety is enforced throughout the pipeline
- Overall test coverage is at least 85% (currently at 91%)

## 9. Test Coverage Enhancement

### Test Coverage Primary Tasks

- [ ] Achieve ≥85% test coverage across all modules
- [ ] Implement integration tests for all critical paths
- [ ] Create performance regression tests
- [ ] Add edge case testing for error handling

### Implementation Steps for Test Coverage

1. Identify modules with insufficient coverage
2. Create targeted unit tests for uncovered code
3. Implement integration tests for module interactions
4. Add performance regression tests
5. Create edge case tests for error handling

### Validation Criteria for Test Coverage

- All critical paths have dedicated tests
- Integration tests validate end-to-end functionality
- CI/CD pipeline catches regressions

## 10. Pre-Launch Verification

### Verification Primary Tasks

- [ ] Perform end-to-end pipeline validation
- [ ] Verify resource utilization
- [ ] Conduct data quality assessment
- [ ] Complete security review

### Implementation Steps for Verification

1. Create validation script for end-to-end testing
2. Implement resource monitoring for full pipeline
3. Develop data quality metrics and validation
4. Conduct security review of pipeline

### Validation Criteria for Verification
  
- Pipeline functions end-to-end without errors
- Resource utilization is within acceptable limits
- Data quality meets defined standards
- Security considerations are addressed

## 11. GPU-Orchestrated Batch Engine

### Overview

Design and implement an asynchronous, double-buffered, multi-GPU engine that moves batches through DocProc, Chunking, Embedding and ISNE stages with configurable stage-to-GPU layout.

### GPU Engine Primary Tasks

- [ ] Create `engine_config.yaml` with pipeline layout, batch size, queue depth, NVLink flag
- [ ] Implement `engine_config.py` (Pydantic / dataclass loader)
- [ ] Define shared `PipelineBatch` dataclass in `src/engine/batch_types.py`
- [ ] Build stage wrappers under `src/engine/stages/` for DocProc, Chunk, Embed, ISNE
- [ ] Develop `StageBase` with device selection & CUDA stream handling
- [ ] Implement `orchestrator.py` (async queues, double-buffer, NVLink peer copy)
- [ ] Add CLI command `ingestor orchestrate` with config path & input path
- [ ] Add Prometheus metrics (`gpu_util`, queue length, batch latency)
- [ ] Write unit tests for config loader, batch types, dummy orchestrator run (≥85 % coverage)
- [ ] Add integration smoke test with 10 docs verifying Arango writes
- [ ] Document architecture in `docs/engine/architecture.md`

### Implementation Steps

1. __Config Layer__ – add YAML + loader with validation.
2. __Types__ – create `PipelineBatch` and ensure mypy passes.
3. __Stage Wrappers__ – thin adapters calling existing modules.
4. __Orchestrator Skeleton__ – async queues with dummy `sleep` to prove overlap.
5. __GPU Integration__ – add CUDA streams, NVLink copy, real model calls.
6. __Metrics & CLI__ – expose metrics, wire into command-line.
7. __Tests & Docs__ – ensure coverage, write architecture docs.

### Validation Criteria

- Pipeline sustains \>90 % utilisation on both RTX A6000 GPUs
- End-to-end latency per 128-doc batch \< 1.5× sum of slowest two stages
- Models remain resident (no load/unload in steady state)
- Unit + integration tests ≥ 85 % coverage, mypy clean

## 12. Embedding Configuration and Adapter System

### Embedding System Implementation Status

1. __Embedding Configuration System__
   - [x] Implemented `embedding_config.py` and `embedding_config.yaml` for flexible model configuration
   - [x] Created adapter registration and factory system for easy switching between models
   - [x] Added configuration validation using Pydantic
   - [x] Implemented environment variable support for configuration overrides

2. __Embedding Adapters Implementation__
   - [x] Created ModernBERT adapter for CPU inference
     - [x] Implemented direct HuggingFace transformers integration
     - [x] Added batching support for efficient processing
     - [x] Implemented multiple pooling strategies (cls, mean, max)
   - [x] Maintained CPU adapter for lightweight embedding
   - [x] Fixed type safety issues in embedding interfaces
   - [x] Added extensive logging for troubleshooting

3. __PDF Pipeline Integration__
   - [x] Updated PDF pipeline to use configurable embedding adapters
   - [x] Added validation testing for full pipeline with ModernBERT embedding
   - [x] Created test framework for comparing embedding approaches
   - [x] Implemented CPU-based embedding for resource-constrained environments

### Remaining Embedding Implementation Tasks

1. __Complete Unit Testing of Embedding Modules__
   - [ ] Create dedicated unit tests for ModernBERT adapter
   - [ ] Add tests for embedding configuration system
   - [ ] Ensure ≥85% test coverage for all embedding modules

2. __Finalize Model Engine Management__
   - [ ] Complete remaining type fixes in Haystack model engine
   - [ ] Resolve unreachable statements in model engine code
   - [ ] Add proper runtime error handling and recovery
   - [ ] Create standardized model loading interface

3. __Metadata Enhancement__
   - [ ] Implement position tracking for text chunks
   - [ ] Add semantic coherence metrics
   - [ ] Standardize metadata format across document types
   - [ ] Ensure source document linkage in all chunks

4. __Document Processing Pipeline__
   - [x] Test docproc module with all supported document types
   - [x] Verify compatibility with schema standards
   - [x] Integrate with chunking validation
   - [x] Add support for PDF documents via Docling
   - [x] Enhance error handling for document processing failures

### Implementation Timeline

- Week 1-2: Chunking module unit tests and type fixes (completed)
- Week 2-3: Model engine management and metadata enhancements (completed)
- Week 3-4: Document processing pipeline and embedding system implementation (completed)
- Week 5: Performance optimization and benchmarking (in progress)

### CPU-Focused Implementation Strategy

- [x] Prioritized CPU implementation for chunking and embedding systems
  - [x] Created CPU-optimized ModernBERT adapter for inference without GPU requirements
  - [x] Enhanced Chonky semantic chunking with proper CPU fallback
  - [x] Implemented configuration system to easily switch between CPU and GPU modes
  - [x] Focused on optimizing performance on CPU before scaling to GPU
  
### ArangoDB Integration Enhancements

- [x] Enhanced ArangoDB integration for document storage
  - [x] Added explicit collection management in ingestion pipeline
  - [x] Implemented modes for creating new collections vs. appending to existing ones
  - [x] Added proper verification of database structures during ingestion
  - [x] Created comprehensive test suite for ArangoDB adapter
  - [x] Ensured all collections and graphs are properly created

## Next Steps After Completion

Once all these tasks are completed and validated, we can proceed with confidence to:

1. Implement the embedding model integration
2. Scale the system for production usage
3. Optimize the pipeline for specific use cases
4. Enhance the system with advanced features

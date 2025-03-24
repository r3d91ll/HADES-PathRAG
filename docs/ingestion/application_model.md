# Application-Centric Knowledge Graph Model

This document outlines the application-centric approach for ingesting and modeling codebases in HADES-PathRAG using the Qwen2.5 model integration.

## Overview

Rather than treating code as a collection of isolated files and functions, our approach views the entire application as a cohesive system with multiple abstraction layers, capturing relationships that span across traditional boundaries.

## Abstraction Layers

### 1. Application Level
- **Entity Type**: The entire application as a top-level entity
- **Properties**: 
  - Name
  - Purpose/domain
  - Version
  - Architecture style (monolith, microservices, etc.)
- **Relationships**: 
  - Contains components
  - Implements requirements
  - Depends on external systems

### 2. Component Level
- **Entity Type**: Major functional components (services, subsystems, modules)
- **Properties**:
  - Name
  - Purpose
  - Interface definitions
  - Responsibility boundaries
- **Relationships**:
  - Part of application
  - Interacts with other components
  - Implements features
  - Exposes APIs
  - Consumes resources

### 3. Artifact Level
- **Entity Types**:
  - Code files (Python modules, classes)
  - Documentation (README, design docs, API docs)
  - Configuration (YAML, JSON, env files)
  - Data models/schemas
  - Tests
- **Properties**: (Specific to artifact type)
  - Filepath
  - Type
  - Purpose
  - Format
- **Relationships**:
  - Belongs to component
  - Imports/depends on other artifacts
  - Defines elements
  - Implements requirements

### 4. Element Level
- **Entity Types**:
  - Code: functions, methods, variables
  - Docs: sections, diagrams, examples
  - Config: settings, parameters
- **Properties**: (Specific to element type)
  - Name
  - Signature (for functions)
  - Type (for variables)
  - Return type (for functions)
  - Docstring
- **Relationships**:
  - Defined in artifact
  - Calls/references other elements
  - Implements functionality
  - Configured by settings

## Cross-Cutting Relationships

One of the key advantages of this approach is capturing cross-cutting relationships:

- `Documentation --[describes]→ Function`
- `Configuration --[configures]→ Service`
- `Test --[verifies]→ Function`
- `API Endpoint --[implemented by]→ Function`
- `Database Schema --[accessed by]→ Class`

## XnX Notation Integration

We use XnX notation to express the strength, directionality, and temporal aspects of these relationships:

```
0.95 ServerComponent -2        // Strong outbound relationship to ServerComponent
0.80 UserAuth +1[2023→2024]    // Temporal relationship to UserAuth function
0.70 ConfigSettings -1         // Accesses configuration
```

## Ingestion Process with Qwen2.5

The Qwen2.5 model plays a crucial role in our ingestion process:

1. **Holistic Analysis**: Qwen2.5 analyzes the entire application structure, identifying components and their purposes.

2. **Entity Extraction**: For each artifact (file, document), Qwen2.5 extracts entities and their properties.

3. **Relationship Identification**: The model identifies both explicit relationships (imports, function calls) and implicit ones (semantic connections, functional dependencies).

4. **Weight Assignment**: Based on its analysis, Qwen2.5 assigns confidence scores to relationships using XnX notation.

5. **Graph Construction**: Entities and relationships are inserted into the ArangoDB graph database, creating a comprehensive knowledge graph of the application.

## Benefits Over File-Level Approaches

- **Architectural Understanding**: Captures high-level design and component interactions
- **Cross-Cutting Concerns**: Identifies relationships that span traditional boundaries
- **Contextual Awareness**: Provides richer context for code understanding
- **Holistic Search**: Enables queries that span multiple abstraction layers

## Schema Compatibility

This application-centric model is compatible with our existing ArangoDB schema. The generic node/edge structure with metadata fields allows us to represent entities at any abstraction level while storing their specific properties and relationships.

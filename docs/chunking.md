# Hybrid Chunking Strategy for HADES-PathRAG

This document outlines the approach for implementing a hybrid chunking strategy that combines symbol table-based code chunking with neural text chunking using Chonky.

## Overview

The HADES-PathRAG system processes both code and documentation. Each content type requires specialized chunking:

- **Code Files**: Require structure-aware chunking that respects function, class, and method boundaries
- **Text Documents**: Benefit from semantic chunking that identifies natural paragraph breaks

Our hybrid approach uses the best tool for each content type:

1. **Symbol Tables** for code structure preservation
2. **Chonky Neural Chunker** for semantic text chunking

## Integration Plan

### 1. Dependencies

Add Chonky to project dependencies:

```python
# pyproject.toml
"chonky>=0.1.0",  # Neural text chunking
```

### 2. Hybrid Chunking Processor

Create a hybrid processor that selects the appropriate chunking method based on content type:

```python
class HybridChunkingProcessor(ChunkingProcessor):
    """
    Hybrid chunking processor using Chonky for text and symbol tables for code.
    """
    
    def __init__(
        self,
        processor_config: Optional[ProcessorConfig] = None,
        chonky_model_id: str = "mirth/chonky_modernbert_base_1",
        device: str = "cpu",
        symbol_table_dir: str = ".symbol_table",
        **kwargs
    ) -> None:
        super().__init__(processor_config=processor_config, **kwargs)
        
        # Initialize Chonky for text documents
        self.text_splitter = TextSplitter(model_id=chonky_model_id, device=device)
        self.symbol_table_dir = symbol_table_dir
```

### 3. Content-Specific Processing

The processor selects the appropriate method based on content type:

- **Python Code**: Uses symbol tables created by PythonPreProcessor
- **Other Code**: Uses syntax-aware heuristics
- **Text Documents**: Uses Chonky neural chunker

### 4. Symbol Table Format

The Python pre-processor creates symbol tables in `.symbol_table/filename.symbols` with the format:

```text
FILE:file_name.py
CLASS:ClassName:line_start-line_end
METHOD:ClassName.method_name
FUNCTION:function_name:line_start-line_end
IMPORT:module.name
```

### 5. Chunk Metadata & Relationships

For each chunk, we store:

- Parent document reference
- Chunk index and total count
- Chunking method used
- Symbol information (for code chunks)

We also create explicit relationships between chunks and parent documents using:

```python
relation = DocumentRelation(
    source_id=document.id,
    target_id=chunk_id,
    relation_type=RelationType.CONTAINS,
    weight=1.0,
    metadata={...}
)
```

## Configuration Options

The hybrid chunking processor supports configuration via:

```json
"chunking": {
    "use_hybrid_chunker": True,
    "chonky_model": "mirth/chonky_modernbert_base_1",
    "device": "cpu",
    "symbol_table_dir": ".symbol_table"
}
```

## Benefits

1. **Improved Code Context**: Symbol tables ensure code chunks maintain functional boundaries
2. **Better Text Chunking**: Chonky provides semantically coherent text chunks
3. **Enhanced Embeddings**: Better chunks lead to more relevant embeddings
4. **Unified API**: Common interface for both code and text
5. **Fallback Strategies**: Graceful degradation if symbol tables aren't available

## Implementation Timeline

1. Complete pre-processor integration
2. Add Chonky dependency
3. Implement HybridChunkingProcessor
4. Update ISNE pipeline configuration
5. Create tests for both code and text chunking approaches

# PDF Pipeline Module Documentation

## Overview

The PDF Pipeline module provides a complete processing pipeline for PDF documents in the HADES-PathRAG system. It integrates document processing, chunking, and embedding generation into a unified workflow.

## Components

### PDFPipeline Class

The main class that orchestrates the entire pipeline:

- `process_document()`: Processes a PDF file using the docproc module
- `chunk_document()`: Chunks a processed document using the chunking module
- `add_embeddings()`: Adds embeddings to document chunks
- `process_pdf()`: Runs the complete pipeline on a PDF file

### Utility Functions

- `run_pipeline()`: Processes multiple PDF files through the pipeline

## Usage Examples

### Basic Usage

```python
from src.pipelines.ingest.orchestrator.pdf_pipeline_prototype import PDFPipeline
import asyncio

# Initialize the pipeline
pipeline = PDFPipeline(
    output_dir="./output",
    save_intermediate_results=True
)

# Process a single PDF
async def process_single_pdf():
    result = await pipeline.process_pdf("/path/to/document.pdf")
    print(f"Generated {len(result['chunks'])} chunks")

# Run the async function
asyncio.run(process_single_pdf())
```

### Processing Multiple PDFs

```python
from src.pipelines.ingest.orchestrator.pdf_pipeline_prototype import run_pipeline
import asyncio
from pathlib import Path

# Process multiple PDFs
async def process_multiple_pdfs():
    pdf_paths = list(Path("./data").glob("*.pdf"))
    results = await run_pipeline(pdf_paths, "./output")
    print(f"Successfully processed {sum(1 for r in results if r['success'])} of {len(results)} PDFs")

# Run the async function
asyncio.run(process_multiple_pdfs())
```

## Testing Requirements

- Unit tests for each pipeline stage
- Integration tests for the complete pipeline
- Performance benchmarks for processing times

## Future Work

- Integration with ArangoDB for persistent storage
- Support for additional document types
- Parallel processing of multiple documents
- Custom chunking strategies based on document type

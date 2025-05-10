"""
Integration test for document processing, chunking, and schema validation.

This test validates the integration between the document processing module (docproc),
the chunking module, and the schema validation module. It processes input files from 
the data directory, chunks them, validates using the schema module, and saves the 
validated JSON objects to the test-output directory.
"""

import os
import json
import unittest
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

# Import schema validation modules
from src.schema.document_schema import DocumentSchema, DatasetSchema, ChunkMetadata
from src.schema.validation import validate_document as schema_validate_document
from src.schema.validation import validate_dataset as schema_validate_dataset
from src.schema.validation import ValidationStage, ValidationResult

# Import chunking modules
from src.chunking import chunk_text, chunk_code

# Mock dependencies to avoid issues during testing
class MockHaystackModelEngine:
    """Mock for HaystackModelEngine to avoid actual model loading during tests."""
    
    def __init__(self):
        self.started = False
        self.loaded_models = {}
    
    def start(self):
        self.started = True
        return {"status": "ok"}
    
    def status(self):
        return {"running": self.started}
    
    def load_model(self, model_id):
        """Mock implementation of load_model method."""
        self.loaded_models[model_id] = True
        return {"success": True, "model_id": model_id}


# Patch the model engine in the chunking module
import src.chunking.text_chunkers.chonky_chunker as chonky_chunker
chonky_chunker._MODEL_ENGINE = MockHaystackModelEngine()

# Mock the ParagraphSplitter to avoid actual model inference
class MockParagraphSplitter:
    """Mock for Chonky ParagraphSplitter."""
    
    def __init__(self, *args, **kwargs):
        pass
    
    def __call__(self, text):
        """Simple paragraph splitting by double newlines."""
        return [p for p in text.split("\n\n") if p.strip()]


# Patch the _get_splitter_with_engine function
original_get_splitter = chonky_chunker._get_splitter_with_engine

def mock_get_splitter(*args, **kwargs):
    """Mock implementation that returns our MockParagraphSplitter."""
    return MockParagraphSplitter()

chonky_chunker._get_splitter_with_engine = mock_get_splitter


class TestDocProcChunkingIntegration(unittest.TestCase):
    """Integration test for document processing, chunking, and schema validation."""

    def setUp(self):
        """Set up test environment."""
        # Define paths
        self.data_dir = Path("/home/todd/ML-Lab/Olympus/HADES-PathRAG/data")
        self.output_dir = Path("/home/todd/ML-Lab/Olympus/HADES-PathRAG/test-output")
        
        # Ensure output directory exists
        self.output_dir.mkdir(exist_ok=True)
        
        # Clean output directory
        for file in self.output_dir.glob("*.json"):
            file.unlink()
    
    def tearDown(self):
        """Clean up after tests."""
        # Restore original function
        chonky_chunker._get_splitter_with_engine = original_get_splitter
    
    def test_markdown_document_chunking(self):
        """Test processing, chunking, and validating a markdown document."""
        # Read the markdown file content directly
        markdown_file = self.data_dir / "docproc.md"
        with open(markdown_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Create a document for chunking
        doc_id = "test_markdown_doc_001"
        document = {
            "id": doc_id,
            "path": str(markdown_file),
            "content": content,
            "type": "markdown"
        }
        
        # Process the document with the chunking module
        chunks = chunk_text(document, max_tokens=1024, output_format="python")
        
        # Create a document schema with chunks
        schema_doc = {
            "id": doc_id,
            "title": "Document Processing Module",
            "content": content,
            "source": str(markdown_file),
            "document_type": "markdown",
            "metadata": {
                "format": "markdown",
                "language": "en",
                "creation_date": "2025-05-10",
                "author": "HADES-PathRAG Team",
            },
            "chunks": []
        }
        
        # Convert chunker output to ChunkMetadata format
        for idx, chunk in enumerate(chunks):
            chunk_content = chunk.get("content", "")
            chunk_metadata = {
                "start_offset": 0,  # Placeholder, would be calculated from actual position
                "end_offset": len(chunk_content),
                "chunk_type": "text",
                "chunk_index": idx,
                "parent_id": doc_id,
                "metadata": {
                    "content": chunk_content,
                    "symbol_type": chunk.get("symbol_type", "paragraph"),
                    "name": chunk.get("name", f"chunk_{idx}"),
                    "token_count": chunk.get("token_count", 0)
                }
            }
            schema_doc["chunks"].append(chunk_metadata)
        
        # Validate using schema validation
        validation_result = schema_validate_document(schema_doc, ValidationStage.INGESTION)
        
        # Check validation result
        self.assertTrue(validation_result.is_valid, 
                        f"Schema validation failed: {validation_result.errors}")
        
        # Save schema document with chunks
        output_path = self.output_dir / "markdown_chunked.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(schema_doc, f, indent=2)
        
        # Verify chunks were created
        self.assertGreater(len(schema_doc["chunks"]), 0, 
                          "No chunks were created for the markdown document")
    
    def test_python_document_chunking(self):
        """Test processing, chunking, and validating a Python document."""
        # Read the Python file content directly
        python_file = self.data_dir / "file_batcher.py"
        with open(python_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Create a document for chunking
        doc_id = "test_python_doc_001"
        document = {
            "id": doc_id,
            "path": str(python_file),
            "content": content,
            "type": "python",
            # Add symbol table information needed by the code chunker
            "functions": [
                {"name": "find_files", "line_start": 30, "line_end": 45},
                {"name": "batch_by_type", "line_start": 50, "line_end": 70}
            ],
            "classes": [
                {"name": "FileBatcher", "line_start": 15, "line_end": 80}
            ]
        }
        
        # Process the document with the code chunking module
        chunks = chunk_code(document, max_tokens=1024, output_format="python")
        
        # Create a document schema with chunks
        schema_doc = {
            "id": doc_id,
            "title": "File Batcher Utility",
            "content": content,
            "source": str(python_file),
            "document_type": "code",
            "metadata": {
                "format": "python",
                "language": "python",
                "creation_date": "2025-05-10",
                "author": "HADES-PathRAG Team",
            },
            "chunks": []
        }
        
        # Convert chunker output to ChunkMetadata format
        for idx, chunk in enumerate(chunks):
            line_start = chunk.get("line_start", 0)
            line_end = chunk.get("line_end", 0)
            chunk_content = chunk.get("content", "")
            
            chunk_metadata = {
                "start_offset": line_start,
                "end_offset": line_end,
                "chunk_type": "code",
                "chunk_index": idx,
                "parent_id": doc_id,
                "metadata": {
                    "content": chunk_content,
                    "symbol_type": chunk.get("symbol_type", "code"),
                    "name": chunk.get("name", f"chunk_{idx}"),
                    "line_start": line_start,
                    "line_end": line_end
                }
            }
            schema_doc["chunks"].append(chunk_metadata)
        
        # Validate using schema validation
        validation_result = schema_validate_document(schema_doc, ValidationStage.INGESTION)
        
        # Check validation result
        self.assertTrue(validation_result.is_valid, 
                        f"Schema validation failed: {validation_result.errors}")
        
        # Save schema document with chunks
        output_path = self.output_dir / "python_chunked.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(schema_doc, f, indent=2)
        
        # Verify chunks were created
        self.assertGreater(len(schema_doc["chunks"]), 0, 
                          "No chunks were created for the Python document")
    
    def test_create_and_validate_chunked_dataset(self):
        """Test creating and validating a dataset from chunked documents."""
        # Create markdown document
        markdown_file = self.data_dir / "docproc.md"
        with open(markdown_file, 'r', encoding='utf-8') as f:
            markdown_content = f.read()
        
        markdown_doc_id = "test_markdown_doc_001"
        markdown_doc = {
            "id": markdown_doc_id,
            "title": "Document Processing Module",
            "content": markdown_content,
            "source": str(markdown_file),
            "document_type": "markdown",
            "metadata": {
                "format": "markdown",
                "language": "en",
                "creation_date": "2025-05-10",
                "author": "HADES-PathRAG Team",
            },
            "chunks": []
        }
        
        # Process markdown document with chunking
        markdown_chunks = chunk_text({
            "id": markdown_doc_id,
            "path": str(markdown_file),
            "content": markdown_content,
            "type": "markdown"
        }, max_tokens=1024, output_format="python")
        
        # Add chunks to markdown document
        for idx, chunk in enumerate(markdown_chunks):
            chunk_content = chunk.get("content", "")
            chunk_metadata = {
                "start_offset": 0,
                "end_offset": len(chunk_content),
                "chunk_type": "text",
                "chunk_index": idx,
                "parent_id": markdown_doc_id,
                "metadata": {
                    "content": chunk_content,
                    "symbol_type": chunk.get("symbol_type", "paragraph"),
                    "name": chunk.get("name", f"chunk_{idx}"),
                    "token_count": chunk.get("token_count", 0)
                }
            }
            markdown_doc["chunks"].append(chunk_metadata)
        
        # Create Python document
        python_file = self.data_dir / "file_batcher.py"
        with open(python_file, 'r', encoding='utf-8') as f:
            python_content = f.read()
        
        python_doc_id = "test_python_doc_001"
        python_doc = {
            "id": python_doc_id,
            "title": "File Batcher Utility",
            "content": python_content,
            "source": str(python_file),
            "document_type": "code",
            "metadata": {
                "format": "python",
                "language": "python",
                "creation_date": "2025-05-10",
                "author": "HADES-PathRAG Team",
            },
            "chunks": []
        }
        
        # Process Python document with chunking
        python_chunks = chunk_code({
            "id": python_doc_id,
            "path": str(python_file),
            "content": python_content,
            "type": "python",
            "functions": [
                {"name": "find_files", "line_start": 30, "line_end": 45},
                {"name": "batch_by_type", "line_start": 50, "line_end": 70}
            ],
            "classes": [
                {"name": "FileBatcher", "line_start": 15, "line_end": 80}
            ]
        }, max_tokens=1024, output_format="python")
        
        # Add chunks to Python document
        for idx, chunk in enumerate(python_chunks):
            line_start = chunk.get("line_start", 0)
            line_end = chunk.get("line_end", 0)
            chunk_content = chunk.get("content", "")
            
            chunk_metadata = {
                "start_offset": line_start,
                "end_offset": line_end,
                "chunk_type": "code",
                "chunk_index": idx,
                "parent_id": python_doc_id,
                "metadata": {
                    "content": chunk_content,
                    "symbol_type": chunk.get("symbol_type", "code"),
                    "name": chunk.get("name", f"chunk_{idx}"),
                    "line_start": line_start,
                    "line_end": line_end
                }
            }
            python_doc["chunks"].append(chunk_metadata)
        
        # Create a dataset
        dataset_id = "test_chunked_dataset_001"
        dataset = {
            "id": dataset_id,
            "name": "Test Chunked Dataset",
            "description": "A test dataset with chunked documents for integration testing",
            "documents": {
                markdown_doc["id"]: markdown_doc,
                python_doc["id"]: python_doc
            },
            "relations": [
                {
                    "source_id": markdown_doc["id"],
                    "target_id": python_doc["id"],
                    "relation_type": "references",
                    "weight": 0.8,
                    "bidirectional": False
                }
            ],
            "metadata": {
                "created_by": "integration_test",
                "version": "1.0"
            }
        }
        
        # Convert datetime objects to strings to make them JSON serializable
        for doc_id, doc in dataset["documents"].items():
            if 'created_at' in doc and doc['created_at']:
                doc['created_at'] = doc['created_at'].isoformat()
            if 'updated_at' in doc and doc['updated_at']:
                doc['updated_at'] = doc['updated_at'].isoformat()
        
        # Validate dataset
        validation_result = schema_validate_dataset(dataset, ValidationStage.INGESTION)
        
        # Check validation result
        self.assertTrue(validation_result.is_valid, 
                        f"Dataset validation failed: {validation_result.errors}")
        
        # Save dataset
        output_path = self.output_dir / "chunked_dataset.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2)
        
        # Create DatasetSchema object
        dataset_obj = DatasetSchema.model_validate(dataset)
        
        # Validate DatasetSchema object
        self.assertEqual(dataset_obj.id, dataset["id"])
        self.assertEqual(dataset_obj.name, dataset["name"])
        self.assertEqual(len(dataset_obj.documents), 2)
        self.assertEqual(len(dataset_obj.relations), 1)
        
        # Verify chunks are present in the dataset
        total_chunks = 0
        for doc_id, doc in dataset["documents"].items():
            total_chunks += len(doc.get("chunks", []))
        
        self.assertGreater(total_chunks, 0, 
                          "No chunks were included in the dataset documents")


if __name__ == "__main__":
    unittest.main()

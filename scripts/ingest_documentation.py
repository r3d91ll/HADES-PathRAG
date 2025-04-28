#!/usr/bin/env python3
"""
Documentation ingestion script for HADES-PathRAG.

This script ingests documentation from a target directory,
processing it with Chonky semantic chunking and storing it in ArangoDB.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import json
from datetime import datetime

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.ingest.orchestrator.ingestor import RepositoryIngestor
from src.db.arango_connection import ArangoConnection
from src.types.common import StorageConfig, PreProcessorConfig

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_documentation_config() -> Dict[str, Any]:
    """
    Create a configuration optimized for documentation ingestion.
    
    Returns:
        Configuration dictionary
    """
    # Storage configuration
    storage_config = {
        "host": "localhost",
        "port": 8529,
        "username": "root",
        "password": "",
        "database": "pathrag_docs",
        "collection_prefix": "docs_",
        "use_vector_index": True,
        "vector_dimensions": 768,
        "embedding": {
            "model_name": "mirth/chonky_modernbert_large_1",
            "model_dimension": 768,
            "batch_size": 8,
            "use_gpu": True,
            "normalize_embeddings": True,
            "use_vllm": True,
            "vllm_server_url": "http://localhost:8000"
        }
    }
    
    # Preprocessor configuration
    preprocessor_config = {
        "exclude_patterns": [
            r"\.git/.*",
            r"__pycache__/.*", 
            r".*\.pyc$",
            r"node_modules/.*",
            r".*\.swp$",
            r".*\.swo$",
            r"\.DS_Store$"
        ],
        "max_workers": 4,
        "batch_size": 10,
        "file_type_map": {
            "markdown": [".md", ".mdx", ".markdown"],
            "text": [".txt", ".text"],
            "pdf": [".pdf"],
            "html": [".html", ".htm"],
            "xml": [".xml"],
            "json": [".json"],
            "yaml": [".yml", ".yaml"],
            "rst": [".rst"]
        }
    }
    
    return {
        "storage": storage_config,
        "preprocessor": preprocessor_config
    }


def ingest_documentation(docs_dir: Path, dataset_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Ingest documentation from a directory.
    
    Args:
        docs_dir: Directory containing documentation
        dataset_name: Optional name for the dataset
        
    Returns:
        Ingestion results and statistics
    """
    # Get configuration
    config = create_documentation_config()
    
    # Create repository ingestor
    ingestor = RepositoryIngestor(
        storage_config=config["storage"],
        preprocessor_config=config["preprocessor"]
    )
    
    # Ingest documentation
    logger.info(f"Starting documentation ingestion from {docs_dir}")
    results = ingestor.ingest(docs_dir, dataset_name)
    
    # Log results
    logger.info(f"Documentation ingestion completed. Stats: {results}")
    
    # Return results with repository reference for querying
    return {
        "results": results,
        "repository": ingestor.get_repository()
    }


def main():
    parser = argparse.ArgumentParser(description="Ingest documentation into HADES-PathRAG")
    parser.add_argument("--docs-dir", type=str, required=True, help="Directory containing documentation files")
    parser.add_argument("--dataset-name", type=str, help="Optional name for the dataset")
    parser.add_argument("--output-file", type=str, help="Optional file to save ingestion statistics")
    args = parser.parse_args()
    
    # Convert to Path
    docs_dir = Path(args.docs_dir)
    
    # Verify documentation directory exists
    if not docs_dir.exists() or not docs_dir.is_dir():
        logger.error(f"Documentation directory does not exist: {docs_dir}")
        return 1
    
    # Check for supported files
    supported_extensions = [
        ".md", ".mdx", ".markdown", ".txt", ".text", ".pdf", 
        ".html", ".htm", ".xml", ".json", ".yml", ".yaml", ".rst"
    ]
    
    has_supported_files = False
    for ext in supported_extensions:
        if list(docs_dir.glob(f"**/*{ext}")):
            has_supported_files = True
            break
    
    if not has_supported_files:
        logger.warning(f"No supported documentation files found in {docs_dir}")
        return 1
    
    # Ingest documentation
    results = ingest_documentation(docs_dir, args.dataset_name)
    
    # Save results if requested
    if args.output_file:
        # Convert to serializable format
        serializable_results = {
            "results": results["results"],
            "timestamp": datetime.now().isoformat()
        }
        
        with open(args.output_file, "w") as f:
            json.dump(serializable_results, f, indent=2)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

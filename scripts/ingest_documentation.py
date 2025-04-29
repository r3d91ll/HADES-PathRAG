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


def create_documentation_config(db_name: str = "pathrag_docs") -> Dict[str, Any]:
    """
    Create a configuration optimized for documentation ingestion.
    
    Args:
        db_name: Name of the ArangoDB database to use
        
    Returns:
        Configuration dictionary
    """
    # Storage configuration
    storage_config = {
        "host": "localhost",
        "port": 8529,
        "username": "root",
        "password": "",
        "database": db_name,
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


def ingest_documentation(docs_dir: Path, dataset_name: Optional[str] = None, 
                    db_name: str = "pathrag_docs", db_mode: str = "append",
                    force: bool = False) -> Dict[str, Any]:
    """
    Ingest documentation from a directory.
    
    Args:
        docs_dir: Directory containing documentation
        dataset_name: Optional name for the dataset
        db_name: Name of ArangoDB database to use
        db_mode: Database mode ('create' or 'append')
        force: Force recreation of collections even if they exist
        
    Returns:
        Ingestion results and statistics
    """
    # Get configuration
    config = create_documentation_config(db_name=db_name)
    
    # Create repository ingestor
    ingestor = RepositoryIngestor(
        storage_config=config["storage"],
        preprocessor_config=config["preprocessor"]
    )
    
    # Handle database mode
    repository = ingestor.get_repository()
    if db_mode == "create":
        # If force is True, first drop collections if they exist
        if force and repository:
            try:
                conn = repository.connection
                graph_name = repository.graph_name
                node_col = repository.node_collection_name
                edge_col = repository.edge_collection_name
                
                # Delete graph first (important)
                if conn.graph_exists(graph_name):
                    logger.info(f"Dropping existing graph {graph_name}")
                    conn.delete_graph(graph_name, drop_collections=True)
                
                # Delete collections as fallback
                if conn.collection_exists(node_col):
                    logger.info(f"Dropping existing node collection {node_col}")
                    conn.delete_collection(node_col)
                    
                if conn.collection_exists(edge_col):
                    logger.info(f"Dropping existing edge collection {edge_col}")
                    conn.delete_collection(edge_col)
            except Exception as e:
                logger.error(f"Error during forced collection cleanup: {e}")
        
        # Explicitly setup collections
        if repository:
            logger.info("Setting up collections explicitly in 'create' mode")
            repository.setup_collections()
    
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
    results = ingest_documentation(
        docs_dir, 
        dataset_name=args.dataset_name,
        db_name=args.db_name if hasattr(args, 'db_name') else "pathrag_docs",
        db_mode=args.db_mode if hasattr(args, 'db_mode') else "append",
        force=args.force if hasattr(args, 'force') else False
    )
    
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

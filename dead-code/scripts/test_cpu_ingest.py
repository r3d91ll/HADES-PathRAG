#!/usr/bin/env python3
"""Test script for CPU-only ingestion pipeline.

This script demonstrates the usage of the repository ingestor with CPU-only
configuration for both chunking and embedding processes. It processes a small
test dataset and reports timing and performance metrics.
"""

import os
import sys
import time
import logging
import asyncio
from pathlib import Path
from typing import Dict, Any
import argparse

# Add the root directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import our mock repository implementation
from scripts.mock_repo import MockArangoConnection, MockArangoRepository

# Patch the imports in the ingestor module to use our mock
import sys
import types
sys.modules['src.storage.arango.connection'] = types.ModuleType('src.storage.arango.connection')
sys.modules['src.storage.arango.connection'].ArangoConnection = MockArangoConnection
sys.modules['src.storage.arango.repository'] = types.ModuleType('src.storage.arango.repository')
sys.modules['src.storage.arango.repository'].ArangoRepository = MockArangoRepository

from src.pipelines.ingest.orchestrator.ingestor import RepositoryIngestor
from src.pipelines.ingest.orchestrator.config import (
    IngestionConfig,
    ChunkingConfig,
    EmbeddingConfig
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("cpu_ingest_test")


async def test_cpu_ingest(
    repo_path: str,
    include_patterns: list[str] = None,
    exclude_patterns: list[str] = None,
    num_workers: int = 4,
    initialize_db: bool = False,
) -> Dict[str, Any]:
    """Run the CPU-only ingestion test.
    
    Args:
        repo_path: Path to the repository/directory to ingest
        include_patterns: Glob patterns for files to include
        exclude_patterns: Glob patterns for files to exclude
        num_workers: Number of CPU workers for parallel processing
        initialize_db: Whether to initialize the database
        
    Returns:
        Dictionary with test results and statistics
    """
    start_time = time.time()
    
    # Configure CPU-only ingestion
    config = IngestionConfig(
        chunking=ChunkingConfig(
            use_gpu=False,
            device="cpu",
            use_semantic_chunking=True,
            num_cpu_workers=num_workers,
        ),
        embedding=EmbeddingConfig(
            use_gpu=False,
            device="cpu",
        ),
        initialize_db=initialize_db,
        batch_size=16,
        max_concurrency=num_workers
    )
    
    logger.info(f"Creating CPU-only ingestor with {num_workers} workers")
    logger.info(f"Processing repository: {repo_path}")
    
    # Create ingestor with CPU configuration
    ingestor = RepositoryIngestor(config=config)
    
    # Run ingestion process
    stats = await ingestor.ingest(
        repo_path=repo_path,
        include_patterns=include_patterns,
        exclude_patterns=exclude_patterns
    )
    
    # Calculate total time
    end_time = time.time()
    total_time = end_time - start_time
    
    # Prepare results
    results = {
        "total_time_seconds": total_time,
        "stats": stats.to_dict(),
        "config": {
            "num_workers": num_workers,
            "initialize_db": initialize_db,
            "repo_path": repo_path
        }
    }
    
    # Display summary
    logger.info("=" * 80)
    logger.info("CPU INGESTION TEST RESULTS")
    logger.info("=" * 80)
    logger.info(f"Total processing time: {total_time:.2f} seconds")
    logger.info(f"Files processed: {stats.processed_files}/{stats.total_files}")
    logger.info(f"Entities created: {stats.entities_created}")
    logger.info(f"Relationships created: {stats.relationships_created}")
    logger.info(f"Embeddings created: {stats.embeddings_created}")
    logger.info(f"Database nodes created: {stats.nodes_created}")
    logger.info(f"Database edges created: {stats.edges_created}")
    logger.info("=" * 80)
    
    return results


async def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Test CPU-only ingestion pipeline")
    parser.add_argument("--repo-path", type=str, required=True, 
                        help="Path to the repository/directory to ingest")
    parser.add_argument("--include", type=str, nargs="+", 
                        help="Glob patterns for files to include")
    parser.add_argument("--exclude", type=str, nargs="+", 
                        help="Glob patterns for files to exclude")
    parser.add_argument("--workers", type=int, default=4, 
                        help="Number of CPU workers for parallel processing")
    parser.add_argument("--init-db", action="store_true", 
                        help="Initialize the database")
    
    args = parser.parse_args()
    
    # Force CPU mode through environment variable
    os.environ["HADES_DEFAULT_DEVICE"] = "cpu"
    
    await test_cpu_ingest(
        repo_path=args.repo_path,
        include_patterns=args.include,
        exclude_patterns=args.exclude,
        num_workers=args.workers,
        initialize_db=args.init_db
    )


if __name__ == "__main__":
    asyncio.run(main())

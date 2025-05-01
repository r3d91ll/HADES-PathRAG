#!/usr/bin/env python3
"""
Documentation query script for HADES-PathRAG.

This script queries the documentation store using the PathRAG algorithm,
returning ranked paths through semantically chunked documentation.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import json
from datetime import datetime
from tabulate import tabulate

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.storage.arango.connection import ArangoConnection
from src.ingest.repository.arango_repository import ArangoRepository
from src.isne.integrations.pathrag_connector import PathRAGConnector
from src.isne.pipeline import ISNEPipeline, PipelineConfig
from src.types.common import PathRankingConfig

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_query_config() -> Dict[str, Any]:
    """
    Create a configuration for querying documentation.
    
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
    
    # PathRAG configuration
    pathrag_config = {
        "semantic_weight": 0.7,
        "path_length_weight": 0.1, 
        "edge_strength_weight": 0.2,
        "max_path_length": 5,
        "max_paths": 20
    }
    
    return {
        "storage": storage_config,
        "pathrag": pathrag_config
    }


def setup_connector() -> PathRAGConnector:
    """
    Set up PathRAG connector for querying.
    
    Returns:
        Initialized PathRAGConnector
    """
    # Get configuration
    config = create_query_config()
    
    # Connect to ArangoDB
    connection = ArangoConnection(
        db_name=config["storage"]["database"],
        host=config["storage"]["host"],
        username=config["storage"]["username"],
        password=config["storage"]["password"]
    )
    
    # Create repository
    repository = ArangoRepository(
        connection=connection,
        node_collection=f"{config['storage']['collection_prefix']}nodes",
        edge_collection=f"{config['storage']['collection_prefix']}edges",
        graph_name=f"{config['storage']['collection_prefix']}pathrag"
    )
    
    # Create ISNE pipeline
    pipeline_config = PipelineConfig(
        embedding_config=config["storage"]["embedding"],
        enable_chunking=True,
        chonky_model_id=config["storage"]["embedding"]["model_name"]
    )
    isne_pipeline = ISNEPipeline(config=pipeline_config)
    
    # Create and return PathRAG connector
    return PathRAGConnector(
        repository=repository,
        isne_pipeline=isne_pipeline,
        path_ranking_config=config["pathrag"]
    )


def query_documentation(query: str, max_results: int = 10, filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Query documentation using PathRAG.
    
    Args:
        query: Query string
        max_results: Maximum number of results to return
        filters: Optional filters to apply
        
    Returns:
        Query results
    """
    # Set up connector
    connector = setup_connector()
    
    # Query documentation
    logger.info(f"Querying documentation with: {query}")
    results = connector.rank_paths(
        query=query,
        document_limit=50,
        path_limit=max_results,
        filters=filters
    )
    
    return results


def display_results(results: Dict[str, Any]) -> None:
    """
    Display query results in a formatted way.
    
    Args:
        results: Query results from PathRAG
    """
    if "error" in results:
        print(f"Error: {results['error']}")
        return
    
    paths = results.get("paths", [])
    metrics = results.get("metrics", {})
    
    # Print metrics
    print(f"\nPathRAG Query Metrics:")
    print(f"- Query time: {metrics.get('query_time', 0):.2f} seconds")
    print(f"- Documents retrieved: {metrics.get('documents_retrieved', 0)}")
    print(f"- Relations analyzed: {metrics.get('relations_retrieved', 0)}")
    print(f"- Paths found: {metrics.get('paths_found', 0)}")
    
    if not paths:
        print("\nNo relevant paths found.")
        return
    
    # Print paths
    print(f"\nTop {len(paths)} Ranked Paths:")
    
    for i, path in enumerate(paths):
        print(f"\n{i+1}. Path Score: {path['score']:.4f} (Semantic: {path['semantic_score']:.4f}, "
              f"Length: {path['path_length_score']:.4f}, Edge: {path['edge_strength']:.4f})")
        
        # Create table for path
        table_data = []
        for j, node in enumerate(path.get("nodes", [])):
            content_preview = node.get("content_preview", "")
            if content_preview and len(content_preview) > 100:
                content_preview = content_preview[:100] + "..."
            
            table_data.append([
                f"{j+1}",
                node.get("title", "Untitled"),
                node.get("document_type", "unknown"),
                content_preview
            ])
        
        headers = ["#", "Title", "Type", "Content Preview"]
        print(tabulate(table_data, headers=headers, tablefmt="grid"))


def main():
    parser = argparse.ArgumentParser(description="Query documentation in HADES-PathRAG")
    parser.add_argument("query", type=str, help="Query to search for in documentation")
    parser.add_argument("--max-results", type=int, default=5, help="Maximum number of results to return")
    parser.add_argument("--document-type", type=str, help="Filter by document type")
    parser.add_argument("--output-file", type=str, help="Optional file to save query results as JSON")
    args = parser.parse_args()
    
    # Create filters if document type specified
    filters = None
    if args.document_type:
        filters = {"document_type": args.document_type}
    
    # Query documentation
    results = query_documentation(args.query, args.max_results, filters)
    
    # Display results
    display_results(results)
    
    # Save results if requested
    if args.output_file:
        with open(args.output_file, "w") as f:
            json.dump(results, f, indent=2)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

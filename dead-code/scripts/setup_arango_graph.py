#!/usr/bin/env python3
"""
Set up ArangoDB graph structure correctly for HADES-PathRAG.

This script creates the proper collections and graph structure according to ArangoDB best practices.
"""

import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from pprint import pprint

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from arango import ArangoClient
from src.storage.arango.connection import get_client, ArangoConnection

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_graph_structure(db_name: str, 
                          graph_name: str = "pathrag",
                          node_collection: str = "nodes", 
                          edge_collection: str = "edges",
                          force: bool = False,
                          username: str = "root",
                          password: str = ""):
    """
    Set up the ArangoDB graph structure properly.
    
    Args:
        db_name: Name of the database
        graph_name: Name of the graph
        node_collection: Name of the node collection
        edge_collection: Name of the edge collection
        force: Whether to drop existing collections if they exist
        username: ArangoDB username
        password: ArangoDB password
    
    Returns:
        Dictionary with information about the created structure
    """
    # Connect to ArangoDB
    client = get_client()
    
    # Check if database exists, create if not
    sys_db = client.db('_system', username=username, password=password)
    if not sys_db.has_database(db_name):
        sys_db.create_database(db_name)
        logger.info(f"Created database {db_name}")
    
    # Connect to the database
    db = client.db(db_name, username=username, password=password)
    logger.info(f"Connected to database {db_name}")
    
    # Handle force option - delete existing collections
    if force:
        # First check if graph exists and delete it
        if db.has_graph(graph_name):
            db.delete_graph(graph_name)
            logger.info(f"Dropped existing graph {graph_name}")
            
        # Delete collections if they exist to ensure clean state
        if db.has_collection(node_collection):
            db.delete_collection(node_collection)
            logger.info(f"Dropped existing collection {node_collection}")
            
        if db.has_collection(edge_collection):
            db.delete_collection(edge_collection)
            logger.info(f"Dropped existing collection {edge_collection}")
    
    # Create vertex collection if it doesn't exist
    if not db.has_collection(node_collection):
        db.create_collection(node_collection)
        logger.info(f"Created vertex collection {node_collection}")
        
        # Create indexes on the node collection
        nodes = db.collection(node_collection)
        nodes.add_persistent_index(fields=['type'], unique=False)
        logger.info(f"Created persistent index on {node_collection}.type")
        
        try:
            # Create full-text index
            nodes.add_fulltext_index(fields=['content'], min_length=3)
            logger.info(f"Created fulltext index on {node_collection}.content")
        except Exception as e:
            logger.warning(f"Could not create fulltext index: {e}")
    else:
        logger.info(f"Using existing vertex collection {node_collection}")
    
    # Create edge collection if it doesn't exist
    if not db.has_collection(edge_collection):
        # Create a collection and configure it as an edge collection
        db.create_collection(edge_collection, edge=True)
        logger.info(f"Created edge collection {edge_collection}")
    else:
        logger.info(f"Using existing edge collection {edge_collection}")
    
    # Create graph if it doesn't exist
    if not db.has_graph(graph_name):
        edge_definition = [{
            'edge_collection': edge_collection,
            'from_vertex_collections': [node_collection],
            'to_vertex_collections': [node_collection]
        }]
        
        graph = db.create_graph(graph_name, edge_definitions=edge_definition)
        logger.info(f"Created graph {graph_name}")
    else:
        graph = db.graph(graph_name)
        logger.info(f"Using existing graph {graph_name}")
    
    # Return information about the created structure
    return {
        'database': db_name,
        'graph': graph_name,
        'nodes': node_collection,
        'edges': edge_collection,
        'collections': list(db.collections()),
        'graphs': list(db.graphs())
    }

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Set up ArangoDB graph structure for HADES-PathRAG")
    parser.add_argument("--db-name", type=str, default="pathrag_docs", 
                       help="Name of the database to use")
    parser.add_argument("--graph-name", type=str, default="pathrag",
                       help="Name of the graph to create")
    parser.add_argument("--node-collection", type=str, default="nodes",
                       help="Name of the node collection")
    parser.add_argument("--edge-collection", type=str, default="edges",
                       help="Name of the edge collection")
    parser.add_argument("--force", action="store_true",
                       help="Force recreation of collections even if they exist")
    parser.add_argument("--username", type=str, default="root",
                       help="ArangoDB username")
    parser.add_argument("--password", type=str, default="",
                       help="ArangoDB password")
    
    args = parser.parse_args()
    
    # Call the setup function
    result = setup_graph_structure(
        db_name=args.db_name,
        graph_name=args.graph_name,
        node_collection=args.node_collection,
        edge_collection=args.edge_collection,
        force=args.force,
        username=args.username,
        password=args.password
    )
    
    # Print the result
    print("\n=== ArangoDB Structure Setup Summary ===")
    print(f"Database: {result['database']}")
    print(f"Graph: {result['graph']}")
    print(f"Node Collection: {result['nodes']}")
    print(f"Edge Collection: {result['edges']}")
    
    print("\n== Collections ==")
    for collection in result['collections']:
        if not collection['name'].startswith('_'):  # Skip system collections
            print(f" - {collection['name']} ({collection['type']})")
    
    print("\n== Graphs ==")
    for graph in result['graphs']:
        print(f" - {graph['name']}")
    
    print("\nArangoDB structure setup complete!")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

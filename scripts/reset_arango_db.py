#!/usr/bin/env python3
"""
Reset ArangoDB Database for HADES-PathRAG

This script drops existing PathRAG databases and creates a fresh one with proper
XnX schema ready for testing.
"""

import os
import sys
import logging
from typing import List, Dict, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import ArangoDB connection utilities
from src.db.arango_connection import get_client, ArangoConnection

# Databases that might exist and need to be reset
PATHRAG_DBS = ['hades', 'pathrag_demo']


def reset_database(db_name: str) -> bool:
    """
    Drop and recreate database with required collections.
    
    Args:
        db_name: Name of database to reset
        
    Returns:
        True if successful, False otherwise
    """
    logger.info(f"Resetting database: {db_name}")
    
    # Get ArangoDB client
    client = get_client()
    logger.info(f"Connected to ArangoDB at {os.environ.get('HADES_ARANGO_URL', 'http://localhost:8529')}")
    
    # Drop database if it exists
    sys_db = client.db('_system')
    
    if sys_db.has_database(db_name):
        logger.info(f"Dropping database: {db_name}")
        sys_db.delete_database(db_name)
        logger.info(f"Database {db_name} dropped")
    
    # Create new database
    logger.info(f"Creating new database: {db_name}")
    sys_db.create_database(
        name=db_name,
        users=[{
            'username': os.environ.get('HADES_ARANGO_USER', 'root'),
            'password': os.environ.get('HADES_ARANGO_PASSWORD', ''),
            'active': True
        }]
    )
    logger.info(f"Database {db_name} created")
    
    # Initialize with XnX schema
    conn = ArangoConnection(db_name=db_name)
    
    # Create collections
    for collection_name in ['pathrag_nodes', 'pathrag_edges']:
        if collection_name.endswith('_edges'):
            conn.create_edge_collection(collection_name)
        else:
            conn.create_collection(collection_name)
    
    # Create graph
    conn.create_graph(
        'pathrag_graph',
        edge_definitions=[{
            'edge_collection': 'pathrag_edges',
            'from_vertex_collections': ['pathrag_nodes'],
            'to_vertex_collections': ['pathrag_nodes']
        }]
    )
    
    logger.info(f"Database {db_name} initialized with XnX schema")
    return True


def main():
    """Reset databases used by PathRAG."""
    logger.info("Starting ArangoDB database reset")
    
    for db_name in PATHRAG_DBS:
        try:
            reset_database(db_name)
        except Exception as e:
            logger.error(f"Error resetting database {db_name}: {e}")
    
    logger.info("Database reset complete")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Verify ArangoDB database contents for HADES-PathRAG.

This script displays information about collections, documents, and edges in the database.
"""

import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, Any
from pprint import pprint

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.db.arango_connection import ArangoConnection

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def display_database_info(db_name: str):
    """
    Display information about the database.
    
    Args:
        db_name: Name of the database to connect to
    """
    # Connect to the database
    conn = ArangoConnection(db_name=db_name)
    
    print(f"\n=== Database '{db_name}' Info ===")
    
    # List collections
    try:
        collections = conn.db.collections()
        
        print(f"\n== Collections ==")
        for collection in collections:
            coll_name = collection['name']
            
            # Check collection type - we need to get the collection and check its type
            try:
                coll = conn.db.collection(coll_name)
                coll_type = "Edge" if coll.type == 3 else "Document"  # ArangoDB uses type 3 for edge collections
            except Exception:
                coll_type = "Unknown"
            print(f" - {coll_name} ({coll_type})")
            
            # Get collection statistics
            if not collection['name'].startswith('_'):  # Skip system collections
                count = conn.db.collection(coll_name).count()
                print(f"   Documents: {count}")
        
        # List graphs
        print(f"\n== Graphs ==")
        graphs = conn.db.graphs()
        if not graphs:
            print("No graphs found")
        else:
            for graph in graphs:
                print(f" - {graph['name']}")
                
                # Get edge definitions
                edge_defs = graph.get('edge_definitions', [])
                print(f"   Edge Collections: {', '.join([e['edge_collection'] for e in edge_defs]) if edge_defs else 'None'}")
                
                # Get vertices
                from_verts = {v for e in edge_defs for v in e.get('from_vertex_collections', [])}
                to_verts = {v for e in edge_defs for v in e.get('to_vertex_collections', [])}
                verts = sorted(list(from_verts.union(to_verts)))
                print(f"   Vertex Collections: {', '.join(verts) if verts else 'None'}")
                
        # Sample documents - determine collection types directly
        doc_collections = []
        edge_collections = []
        
        for collection in collections:
            if collection['name'].startswith('_'):
                # Skip system collections
                continue
                
            try:
                coll = conn.db.collection(collection['name'])
                if coll.type == 3:  # Edge collection
                    edge_collections.append(collection)
                else:  # Document collection
                    doc_collections.append(collection)
            except Exception as e:
                logger.warning(f"Error determining type for collection {collection['name']}: {e}")
        
        if doc_collections:
            print(f"\n== Sample Documents ==")
            for collection in doc_collections[:3]:  # Limit to first 3 collections
                coll_name = collection['name']
                print(f"\nCollection: {coll_name}")
                
                # Get 3 sample documents
                query = f"FOR doc IN {coll_name} LIMIT 3 RETURN doc"
                cursor = conn.query(query)
                if cursor:
                    for i, doc in enumerate(cursor):
                        print(f"\nDocument {i+1}:")
                        # Print selective fields to avoid overwhelming output
                        safe_doc = {
                            "_key": doc.get("_key", "N/A"),
                            "_id": doc.get("_id", "N/A"),
                            "type": doc.get("type", "N/A"),
                            "title": doc.get("title", "N/A") if isinstance(doc.get("title"), str) else "N/A"
                        }
                        # Add other fields that might be interesting
                        if "metadata" in doc:
                            safe_doc["metadata"] = doc["metadata"]
                        pprint(safe_doc)
        
        if edge_collections:
            print(f"\n== Sample Edges ==")
            for collection in edge_collections[:2]:  # Limit to first 2 collections
                coll_name = collection['name']
                print(f"\nEdge Collection: {coll_name}")
                
                # Get 3 sample edges
                query = f"FOR edge IN {coll_name} LIMIT 3 RETURN edge"
                cursor = conn.query(query)
                if cursor:
                    for i, edge in enumerate(cursor):
                        print(f"\nEdge {i+1}:")
                        # Print selective fields
                        safe_edge = {
                            "_key": edge.get("_key", "N/A"),
                            "_from": edge.get("_from", "N/A"),
                            "_to": edge.get("_to", "N/A"),
                            "type": edge.get("type", "N/A"),
                            "weight": edge.get("weight", "N/A")
                        }
                        pprint(safe_edge)
                
    except Exception as e:
        logger.error(f"Error getting database info: {e}")

def main():
    parser = argparse.ArgumentParser(description="Verify ArangoDB database contents for HADES-PathRAG")
    parser.add_argument("--db-name", type=str, default="pathrag_docs", 
                        help="Name of the ArangoDB database to verify")
    args = parser.parse_args()
    
    display_database_info(args.db_name)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

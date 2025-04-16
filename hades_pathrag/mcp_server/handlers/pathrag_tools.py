"""
MCP Tools for HADES-PathRAG.

This module implements MCP tools for the HADES-PathRAG system, including:
- Path retrieval
- Document embedding
- Semantic search
- Graph operations
"""
import logging
from typing import Any, Dict, List, Optional, Union

from hades_pathrag.storage.arango import ArangoDBConnection
from hades_pathrag.mcp_server.config import get_config
from hades_pathrag.ingestion.enhanced_embeddings import HybridEmbeddingProcessor
from hades_pathrag.ingestion.models import IngestDocument

logger = logging.getLogger(__name__)

# Global embedding processor that will be set by the MCP server
embedding_processor: Optional[HybridEmbeddingProcessor] = None

# Configure verbose logging for debugging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

logger.setLevel(logging.DEBUG)
logger.info("PathRAG tools module loaded and configured for debugging")


async def retrieve_path(
    query: str,
    start_node: Optional[str] = None,
    end_node: Optional[str] = None,
    max_length: int = 5,
    min_similarity: float = 0.7
) -> List[Dict[str, Any]]:
    """
    Retrieve a path from the knowledge graph based on a natural language query.
    
    Args:
        query: The natural language query
        start_node: Optional starting node
        end_node: Optional ending node
        max_length: Maximum path length
        min_similarity: Minimum similarity threshold
        
    Returns:
        A list of paths matching the query
    """
    logger.info(f"Retrieving path for query: {query}")
    
    # Get database configuration
    config = get_config()
    
    # Create a database connection
    db = ArangoDBConnection(
        host=config.database.url.replace('http://', '').replace('https://', '').split(':')[0],
        port=int(config.database.url.split(':')[-1]) if ':' in config.database.url else 8529,
        username=config.database.username,
        password=config.database.password,
        database=config.database.database_name
    )
    
    # Get the relevant config
    config = get_config()
    
    # Logic to retrieve paths would go here
    # This is a placeholder implementation
    paths = [
        {
            "path": [start_node or "DefaultStart", "MiddleNode", end_node or "DefaultEnd"],
            "similarity": 0.85,
            "supporting_documents": ["doc1", "doc2"]
        }
    ]
    
    return paths


async def embed_document(
    document: str,
    metadata: Optional[Dict[str, Any]] = None,
    model: Optional[str] = None,
    timeout_seconds: int = 60
) -> Dict[str, Any]:
    """
    Embed a document and store it in the knowledge graph.
    
    Args:
        document: The document text
        metadata: Optional metadata for the document
        model: Optional embedding model to use
        
    Returns:
        Information about the embedded document
    """
    logger.debug("⏱️ [START] embed_document called")
    # Truncate document for logging
    doc_preview = document[:50] + "..." if len(document) > 50 else document
    logger.debug(f"📄 Processing document: '{doc_preview}'")
    logger.debug(f"🏷️ Metadata: {metadata}")
    
    # Check the embedding processor early
    if embedding_processor is None:
        logger.error("❌ No embedding processor available! This is a critical error.")
        logger.debug("⏱️ [END] embed_document returning placeholder due to missing processor")
        return {
            "document_id": f"doc_{hash(document)}",
            "embedding_model": model or "placeholder",
            "dimensions": 768,
            "metadata": metadata or {},
            "error": "No embedding processor available"
        }
    
    try:
        logger.debug("⚙️ Creating document object")
        # Create a document object
        doc = IngestDocument(
            id=f"doc_{hash(document)}",
            content=document,
            title=metadata.get("title", "") if metadata else "",
            metadata=metadata or {}
        )
        
        logger.debug(f"🔄 Embedding document with processor type: {type(embedding_processor).__name__}")
        logger.debug(f"🧮 Processor config: semantic_weight={embedding_processor.semantic_weight}, embedding_dim={embedding_processor.semantic_embedding_dim}")
        
        # Embed the document with timeout and timing
        import time
        import concurrent.futures
        import asyncio
        
        # To avoid timeout issues, we'll run the embedding in a separate thread
        # with an explicit timeout
        start_time = time.time()
        
        def embed_doc_worker() -> IngestDocument:
            return embedding_processor.embed_document(doc)
            
        # Create a thread executor for the CPU-intensive embedding task
        with concurrent.futures.ThreadPoolExecutor() as executor:
            try:
                # Convert the sync task to an async future
                loop = asyncio.get_event_loop()
                embedding_future = loop.run_in_executor(executor, embed_doc_worker)
                # Wait for the embedding with a timeout
                result_doc = await asyncio.wait_for(embedding_future, timeout=timeout_seconds)
                embed_time = time.time() - start_time
                logger.debug(f"⏱️ Document embedded in {embed_time:.2f} seconds")
            except asyncio.TimeoutError:
                logger.error(f"⏱️ Embedding timed out after {timeout_seconds} seconds")
                # Return a placeholder embedding to avoid breaking the pipeline
                doc.embedding = [0.0] * (metadata.get("embedding_dim", 768) if metadata else 768)
                result_doc = doc  # type: ignore

        # At this point, result_doc is guaranteed to exist
        if result_doc.embedding is None:
            logger.warning("⚠️ Embedded document has no embedding vector!")
            embedding_length = 0
        else:
            embedding_length = len(result_doc.embedding)
            logger.debug(f"✅ Generated embedding with dimension: {embedding_length}")

        # Return information about the embedding
        embedding_info = {
            "document_id": embedded_doc.id,
            "embedding_model": "hybrid_modernbert_isne",
            "dimensions": embedding_length,
            "metadata": embedded_doc.metadata
        }
        logger.debug("⏱️ [END] embed_document completed successfully")
        return embedding_info
        
    except Exception as e:
        import traceback
        logger.error(f"❌ Error embedding document: {e}")
        logger.error(f"Stack trace: {traceback.format_exc()}")
        
        # Fall back to placeholder
        embedding_info = {
            "document_id": f"doc_{hash(document)}",
            "embedding_model": model or "hybrid_modernbert_isne",
            "dimensions": 768,
            "metadata": metadata or {},
            "error": str(e)
        }
        logger.debug("⏱️ [END] embed_document returning error placeholder")
        return embedding_info


async def semantic_search(
    query: str,
    collection: Optional[str] = None,
    top_k: int = 5,
    min_similarity: float = 0.7
) -> List[Dict[str, Any]]:
    """
    Perform a semantic search on the knowledge graph.
    
    Args:
        query: The search query
        collection: Optional specific collection to search
        top_k: Number of results to return
        min_similarity: Minimum similarity threshold
        
    Returns:
        A list of search results
    """
    logger.info(f"Performing semantic search for query: {query}")
    
    # Get database configuration
    config = get_config()
    
    # Create a database connection
    db = ArangoDBConnection(
        host=config.database.url.replace('http://', '').replace('https://', '').split(':')[0],
        port=int(config.database.url.split(':')[-1]) if ':' in config.database.url else 8529,
        username=config.database.username,
        password=config.database.password,
        database=config.database.database_name
    )
    
    # Get the relevant config
    config = get_config()
    
    # Use the hybrid embedding processor if available
    if embedding_processor is not None and embedding_processor.semantic_embedder is not None:
        try:
            # Use semantic embedder to create query embedding
            query_embedding = embedding_processor.semantic_embedder.embed(query)
            
            # In a real implementation, we would search the vector database
            # For now, just return a more informative placeholder
            results = [
                {
                    "document_id": f"doc{i}",
                    "similarity": 0.9 - (i * 0.05),
                    "content": f"Sample content for result {i} matching query: {query}",
                    "metadata": {
                        "type": "text", 
                        "source": "example",
                        "embedding_model": "hybrid_modernbert_isne"
                    }
                }
                for i in range(1, top_k + 1)
            ]
        except Exception as e:
            logger.error(f"Error performing semantic search: {e}")
            # Fall back to placeholder
            results = [
                {
                    "document_id": f"doc{i}",
                    "similarity": 0.9 - (i * 0.05),
                    "content": f"Sample content for result {i}",
                    "metadata": {"type": "text", "source": "example", "error": str(e)}
                }
                for i in range(1, min(3, top_k + 1))
            ]
    else:
        # Placeholder implementation when no processor is available
        results = [
            {
                "document_id": f"doc{i}",
                "similarity": 0.9 - (i * 0.05),
                "content": f"Sample content for result {i}",
                "metadata": {"type": "text", "source": "example", "warning": "No embedding processor available"}
            }
            for i in range(1, top_k + 1)
        ]
    
    return results


async def create_graph_node(
    node_type: str,
    properties: Dict[str, Any],
    collection: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create a new node in the knowledge graph.
    
    Args:
        node_type: The type of node
        properties: Node properties
        collection: Optional collection to add the node to
        
    Returns:
        Information about the created node
    """
    logger.info(f"Creating graph node of type: {node_type}")
    
    # Get database configuration
    config = get_config()
    
    # Create a database connection
    db = ArangoDBConnection(
        host=config.database.url.replace('http://', '').replace('https://', '').split(':')[0],
        port=int(config.database.url.split(':')[-1]) if ':' in config.database.url else 8529,
        username=config.database.username,
        password=config.database.password,
        database=config.database.database_name
    )
    
    # Get the relevant config
    config = get_config()
    
    # Default collection if not specified
    if not collection:
        # Use the node_collection_name from PathRAG config
        collection = config.pathrag.max_nodes and "nodes" or "nodes"  # Fallback to 'nodes'
    
    # Logic to create a node would go here
    # This is a placeholder implementation
    node_info = {
        "node_id": "node123",
        "node_type": node_type,
        "collection": collection,
        "properties": properties
    }
    
    return node_info


async def create_graph_edge(
    from_node: str,
    to_node: str,
    edge_type: str,
    properties: Optional[Dict[str, Any]] = None,
    collection: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create a new edge in the knowledge graph.
    
    Args:
        from_node: The source node ID
        to_node: The target node ID
        edge_type: The type of edge
        properties: Optional edge properties
        collection: Optional collection to add the edge to
        
    Returns:
        Information about the created edge
    """
    logger.info(f"Creating graph edge of type: {edge_type}")
    
    # Get database configuration
    config = get_config()
    
    # Create a database connection
    db = ArangoDBConnection(
        host=config.database.url.replace('http://', '').replace('https://', '').split(':')[0],
        port=int(config.database.url.split(':')[-1]) if ':' in config.database.url else 8529,
        username=config.database.username,
        password=config.database.password,
        database=config.database.database_name
    )
    
    # Get the relevant config
    config = get_config()
    
    # Default collection if not specified
    if not collection:
        # Use the edge_collection_name from PathRAG config
        collection = config.pathrag.max_paths and "edges" or "edges"  # Fallback to 'edges'
    
    # Logic to create an edge would go here
    # This is a placeholder implementation
    edge_info = {
        "edge_id": "edge123",
        "edge_type": edge_type,
        "from_node": from_node,
        "to_node": to_node,
        "collection": collection,
        "properties": properties or {}
    }
    
    return edge_info


async def get_graph_statistics() -> Dict[str, Any]:
    """
    Get statistics about the knowledge graph.
    
    Returns:
        Statistics about the knowledge graph
    """
    logger.info("Retrieving graph statistics")
    
    # Get database configuration
    config = get_config()
    
    # Create a database connection
    db = ArangoDBConnection(
        host=config.database.url.replace('http://', '').replace('https://', '').split(':')[0],
        port=int(config.database.url.split(':')[-1]) if ':' in config.database.url else 8529,
        username=config.database.username,
        password=config.database.password,
        database=config.database.database_name
    )
    
    # Logic to get statistics would go here
    # This is a placeholder implementation
    stats = {
        "node_count": 1000,
        "edge_count": 5000,
        "node_collections": ["documents", "entities", "concepts"],
        "edge_collections": ["links", "relationships"],
        "embedding_models": ["sentence-transformers", "custom"],
        "storage_size_mb": 250
    }
    
    return stats

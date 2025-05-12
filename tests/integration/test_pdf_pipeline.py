"""
Integration test for PDF processing in the full pipeline.

This test validates that the full pipeline can process PDF files correctly,
including document processing, chunking, embedding generation, and storage.
"""

import os
import sys
import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.pipelines.ingest.orchestrator.ingestor import RepositoryIngestor
from src.docproc.manager import DocumentProcessorManager
from src.embedding.base import get_adapter
from src.storage.arango.connection import ArangoConnection

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_pdf_pipeline():
    """Test the full pipeline with PDF files."""
    # Create a temporary directory with the PDF files
    temp_dir = Path("temp_pdf_test")
    temp_dir.mkdir(exist_ok=True)
    
    # Copy the PDF files to the temporary directory
    pdf_files = [
        Path("docs/PathRAG_paper.pdf"),
        Path("test-data/ISNE_paper.pdf")
    ]
    
    for pdf_file in pdf_files:
        if not pdf_file.exists():
            logger.error(f"Test PDF not found: {pdf_file}")
            continue
        
        # Create a copy in the temp directory
        target_path = temp_dir / pdf_file.name
        with open(pdf_file, "rb") as src, open(target_path, "wb") as dst:
            dst.write(src.read())
        
        logger.info(f"Copied {pdf_file} to {target_path}")
    
    # Initialize the pipeline components
    doc_processor = DocumentProcessorManager()
    embedding_adapter = get_adapter("vllm")  # This will use the default adapter if vllm is not available
    
    # Create a connection to ArangoDB with a unique database name for testing
    connection = ArangoConnection(
        database="pdf_test_db",
        host=os.environ.get("ARANGO_HOST", "http://localhost:8529"),
        username=os.environ.get("ARANGO_USERNAME", "root"),
        password=os.environ.get("ARANGO_PASSWORD", ""),
    )
    
    # Initialize the ingestor with our components
    ingestor = RepositoryIngestor(
        connection=connection,
        doc_processor=doc_processor,
        embedding_adapter=embedding_adapter,
        initialize_db=True,  # Create the database structure
        batch_size=2,
        max_concurrency=2,
    )
    
    try:
        # Run the ingestion process
        logger.info("Starting PDF pipeline test...")
        stats = await ingestor.ingest(
            repo_path=temp_dir,
            include_patterns=["**/*.pdf"],
            exclude_patterns=[],
        )
        
        # Log the results
        logger.info("PDF pipeline test completed")
        logger.info(f"Statistics: {stats.to_dict()}")
        
        # Validate the results
        assert stats.processed_files > 0, "No files were processed"
        assert stats.entities_created > 0, "No entities were created"
        assert stats.embeddings_created > 0, "No embeddings were created"
        assert stats.nodes_created > 0, "No nodes were stored in the database"
        
        # Check if we have document and chunk entities in the database
        doc_count = await connection.db.collection("nodes").count()
        logger.info(f"Total nodes in database: {doc_count}")
        
        # Query for document entities
        doc_cursor = await connection.db.aql.execute(
            "FOR doc IN nodes FILTER doc.type == 'document' RETURN doc"
        )
        documents = await doc_cursor.all()
        logger.info(f"Document entities: {len(documents)}")
        
        # Query for chunk entities
        chunk_cursor = await connection.db.aql.execute(
            "FOR chunk IN nodes FILTER chunk.type == 'chunk' RETURN chunk"
        )
        chunks = await chunk_cursor.all()
        logger.info(f"Chunk entities: {len(chunks)}")
        
        # Check if we have embeddings
        embed_cursor = await connection.db.aql.execute(
            "FOR embed IN embeddings RETURN embed"
        )
        embeddings = await embed_cursor.all()
        logger.info(f"Embeddings: {len(embeddings)}")
        
        # Check if we have relationships
        edge_cursor = await connection.db.aql.execute(
            "FOR edge IN edges RETURN edge"
        )
        edges = await edge_cursor.all()
        logger.info(f"Edges: {len(edges)}")
        
        return {
            "stats": stats.to_dict(),
            "documents": len(documents),
            "chunks": len(chunks),
            "embeddings": len(embeddings),
            "edges": len(edges),
        }
    
    finally:
        # Clean up the temporary directory
        for file in temp_dir.glob("*"):
            file.unlink()
        temp_dir.rmdir()
        
        # Drop the test database
        try:
            await connection.sys_db.delete_database("pdf_test_db")
            logger.info("Cleaned up test database")
        except Exception as e:
            logger.warning(f"Error cleaning up test database: {e}")


async def main():
    """Run the PDF pipeline test."""
    try:
        results = await test_pdf_pipeline()
        logger.info("Test completed successfully")
        logger.info(f"Results: {results}")
    except Exception as e:
        logger.error(f"Test failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())

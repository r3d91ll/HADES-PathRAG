#!/usr/bin/env python
"""
Startup script for HADES-PathRAG MCP server with proper model initialization.

This script initializes the hybrid embedding system that combines ModernBERT semantic
embeddings with ISNE structural embeddings before starting the MCP server.
"""
import os
import sys
import logging
import argparse
from pathlib import Path
from typing import Optional, Dict, Any

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add project root to path if needed
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import model configuration
from model_config import setup_models


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Start MCP server with hybrid embedding system")
    parser.add_argument(
        "--host", 
        type=str,
        default="127.0.0.1",
        help="Host to bind the server to"
    )
    parser.add_argument(
        "--port", 
        type=int,
        default=8000,
        help="Port to bind the server to"
    )
    parser.add_argument(
        "--debug", 
        action="store_true",
        help="Run in debug mode"
    )
    parser.add_argument(
        "--log-level", 
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Logging level"
    )
    parser.add_argument(
        "--semantic-weight",
        type=float,
        default=0.5,
        help="Weight of semantic embeddings (0.0-1.0)"
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda", "mps"],
        help="Device to run models on (defaults to cuda if available)"
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    # Parse command line arguments
    args = parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    logger.info("Initializing hybrid embedding system...")
    try:
        # Initialize the hybrid embedding processor
        processor = setup_models(
            semantic_weight=args.semantic_weight,
            device=args.device
        )
        logger.info("Hybrid embedding system initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize hybrid embedding system: {e}")
        sys.exit(1)
    
    # Register the processor with the appropriate components
    try:
        # Store the processor in a global variable for use by MCP tools
        import hades_pathrag.mcp_server.handlers.pathrag_tools as pathrag_tools
        pathrag_tools.embedding_processor = processor
        logger.info("Registered processor with MCP tools")
    except Exception as e:
        logger.error(f"Failed to register processor with MCP tools: {e}")
        sys.exit(1)
    
    # Import the MCP server app
    from hades_pathrag.mcp_server.mcp_standalone import app, get_config
    
    # Configure the server
    config = get_config()
    config.server.host = args.host
    config.server.port = args.port
    config.server.debug = args.debug
    config.server.log_level = args.log_level
    
    # Import uvicorn for running the server
    import uvicorn
    
    logger.info(f"Starting HADES-PathRAG MCP Server (Model Context Protocol)")
    logger.info(f"Server running at http://{config.server.host}:{config.server.port}")
    
    # Run the server
    uvicorn.run(
        app,
        host=config.server.host,
        port=config.server.port,
        log_level=config.server.log_level.lower()
    )


if __name__ == "__main__":
    main()

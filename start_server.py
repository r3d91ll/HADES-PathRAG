#!/usr/bin/env python3
"""
HADES-PathRAG Server Launcher.

This script launches the HADES-PathRAG MCP server with Ollama integration,
providing a web interface for XnX-enhanced PathRAG operations.
"""

import os
import sys
import argparse
import logging
import subprocess
import time
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('hades_pathrag_server.log')
    ]
)
logger = logging.getLogger("HADES-PathRAG")

def check_ollama():
    """Check if Ollama is running, and attempt to start it if not."""
    try:
        import httpx
        async def check_ollama_service():
            async with httpx.AsyncClient() as client:
                try:
                    response = await client.get("http://localhost:11434/api/version", timeout=2.0)
                    if response.status_code == 200:
                        version = response.json().get("version", "unknown")
                        logger.info(f"✅ Ollama is running (version {version})")
                        return True
                except Exception as e:
                    logger.warning(f"❌ Ollama is not running: {str(e)}")
                    return False
        
        import asyncio
        ollama_running = asyncio.run(check_ollama_service())
        
        if not ollama_running:
            logger.info("🚀 Attempting to start Ollama...")
            try:
                # Start Ollama in the background
                subprocess.Popen(
                    ["ollama", "serve"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                time.sleep(5)  # Give Ollama time to start
                
                # Check again
                ollama_running = asyncio.run(check_ollama_service())
                if not ollama_running:
                    logger.warning("⚠️ Failed to start Ollama. Please start it manually.")
            except Exception as e:
                logger.warning(f"⚠️ Error starting Ollama: {str(e)}")
                logger.warning("⚠️ Please start Ollama manually with 'ollama serve'")
    
    except ImportError:
        logger.warning("⚠️ httpx not installed. Cannot check Ollama status.")

def check_required_models(model_name="llama3"):
    """Check if the required Ollama models are available and pull them if not."""
    try:
        result = subprocess.run(
            ["ollama", "list"], 
            capture_output=True, 
            text=True
        )
        if model_name not in result.stdout:
            logger.info(f"🔄 Pulling {model_name} model (this may take a while)...")
            subprocess.run(["ollama", "pull", model_name])
            logger.info(f"✅ Successfully pulled {model_name}")
        else:
            logger.info(f"✅ {model_name} model is already available")
    except Exception as e:
        logger.warning(f"⚠️ Error checking/pulling models: {str(e)}")

def check_arango():
    """Check if ArangoDB is running."""
    try:
        import httpx
        async def check_arango_service():
            async with httpx.AsyncClient() as client:
                try:
                    response = await client.get("http://localhost:8529/_api/version", timeout=2.0)
                    if response.status_code == 200:
                        version = response.json().get("version", "unknown")
                        logger.info(f"✅ ArangoDB is running (version {version})")
                        return True
                except Exception as e:
                    logger.warning(f"❌ ArangoDB is not running: {str(e)}")
                    logger.warning("⚠️ Please start ArangoDB manually")
                    return False
        
        import asyncio
        return asyncio.run(check_arango_service())
    
    except ImportError:
        logger.warning("⚠️ httpx not installed. Cannot check ArangoDB status.")
        return False

def start_server(host="0.0.0.0", port=8000, debug=False):
    """Start the HADES-PathRAG MCP server."""
    from src.mcp.server import start_server
    logger.info(f"🚀 Starting HADES-PathRAG MCP server on {host}:{port}")
    
    # Create path_cache directory if it doesn't exist
    Path("./path_cache").mkdir(exist_ok=True)
    
    # Start the server
    start_server(host=host, port=port)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start HADES-PathRAG MCP server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", default=8000, type=int, help="Port to bind to")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--skip-checks", action="store_true", help="Skip dependency checks")
    
    args = parser.parse_args()
    
    # ASCII Art Banner
    banner = """
    ╔═══════════════════════════════════════════════════════════╗
    ║                                                           ║
    ║  ██╗  ██╗ █████╗ ██████╗ ███████╗███████╗                ║
    ║  ██║  ██║██╔══██╗██╔══██╗██╔════╝██╔════╝                ║
    ║  ███████║███████║██║  ██║█████╗  ███████╗                ║
    ║  ██╔══██║██╔══██║██║  ██║██╔══╝  ╚════██║                ║
    ║  ██║  ██║██║  ██║██████╔╝███████╗███████║                ║
    ║  ╚═╝  ╚═╝╚═╝  ╚═╝╚═════╝ ╚══════╝╚══════╝                ║
    ║                                                           ║
    ║       XnX-enhanced PathRAG with Ollama Integration       ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝
    """
    print(banner)
    
    # Run checks if not skipped
    if not args.skip_checks:
        logger.info("🔍 Running system checks...")
        check_ollama()
        check_required_models()
        check_arango()
    
    # Start the server
    try:
        start_server(host=args.host, port=args.port, debug=args.debug)
    except KeyboardInterrupt:
        logger.info("👋 Server stopped by user")
    except Exception as e:
        logger.error(f"❌ Error starting server: {str(e)}")
        sys.exit(1)

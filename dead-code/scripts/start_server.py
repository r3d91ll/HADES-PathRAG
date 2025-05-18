#!/usr/bin/env python3
"""
HADES-PathRAG Server Launcher.

This script launches the HADES-PathRAG FastAPI server with vLLM integration,
providing a web interface for PathRAG operations.
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

def check_vllm():
    """Check if vLLM is installed and available."""
    try:
        import vllm
        logger.info(f"✅ vLLM is installed (version {vllm.__version__})")
        return True
    except ImportError:
        logger.warning("⚠️ vLLM is not installed. Some functionality may be limited.")
        logger.warning("⚠️ Install vLLM with: pip install vllm")
        return False
    except Exception as e:
        logger.warning(f"⚠️ Error importing vLLM: {str(e)}")
        logger.warning("⚠️ This may be due to compatibility issues between PyTorch and torchvision.")
        logger.warning("⚠️ The server will continue without vLLM functionality.")
        return False

# TODO: we should be using the model name from the config file
def initialize_vllm_server(model_name="unsloth/Qwen3-14B"):
    """Initialize vLLM server for embedding and model inference."""
    try:
        # First check if vLLM is available
        try:
            import vllm
        except (ImportError, Exception) as e:
            logger.warning(f"⚠️ Cannot initialize vLLM server: {str(e)}")
            logger.warning("⚠️ The server will continue without vLLM functionality.")
            return False
            
        from src.model_engine.server_manager import ServerManager
        from src.types.vllm_types import VLLMConfig, VLLMModelConfig, VLLMServerConfig
        
        logger.info(f"🚀 Initializing vLLM server with model: {model_name}")
        
        # Create basic configuration
        server_config = VLLMServerConfig(
            host="localhost",
            port=8080,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.9
        )
        
        # Create model configuration
        model_config = VLLMModelConfig(
            model_id=model_name
        )
        
        # Initialize server manager with default config path
        # The server_manager expects a path to config, not a config object
        config_path = os.path.join(os.path.dirname(__file__), "src/config/vllm_config.yaml")
        # Create server manager with different server port to avoid conflicts
        manager = ServerManager(config_path=config_path, host="localhost", base_port=8123)
        
        try:
            # The actual server manager API uses async methods
            import asyncio
            
            # Create an event loop or use existing one
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            # Start server in background (non-blocking)
            # Use 'general' model from config instead of 'default'
            model_alias = "general"  # Available: general, code, fast in vllm_config.yaml
            success = loop.run_until_complete(
                manager.ensure_server_running(model_alias, "inference")
            )
            
            if success:
                logger.info(f"✅ vLLM server initialization started for {model_name}")
                return True
            else:
                logger.warning("⚠️ Failed to initialize vLLM server")
                return False
        except Exception as e:
            logger.warning(f"⚠️ Error starting vLLM server: {str(e)}")
            return False
    
    except Exception as e:
        logger.warning(f"⚠️ Error initializing vLLM: {str(e)}")
        logger.warning("⚠️ The server will continue without vLLM functionality.")
        return False

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
                    logger.warning("⚠️ Make sure ArangoDB is installed and running")
                    return False
        
        import asyncio
        return asyncio.run(check_arango_service())
    
    except ImportError:
        logger.warning("⚠️ httpx not installed. Cannot check ArangoDB status.")
        return False

def start_server(host="0.0.0.0", port=8000, debug=False):
    """Start the HADES-PathRAG FastAPI server."""
    from src.api.server import app
    import uvicorn
    logger.info(f"🚀 Starting HADES-PathRAG FastAPI server on {host}:{port}")
    
    # Create path_cache directory if it doesn't exist
    Path("./path_cache").mkdir(exist_ok=True)
    
    # Start the server
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start HADES-PathRAG FastAPI server")
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
    ║         PathRAG with vLLM Integration                    ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝
    """
    print(banner)
    
    # Run checks if not skipped
    if not args.skip_checks:
        logger.info("🔍 Running system checks...")
        try:
            vllm_available = check_vllm()
            if vllm_available:
                initialize_vllm_server()
        except Exception as e:
            logger.warning(f"⚠️ Error during vLLM checks: {str(e)}")
            logger.warning("⚠️ Continuing without vLLM functionality")
        check_arango()
    
    # Start the server
    try:
        start_server(host=args.host, port=args.port, debug=args.debug)
    except KeyboardInterrupt:
        logger.info("👋 Server stopped by user")
    except Exception as e:
        logger.error(f"❌ Error starting server: {str(e)}")
        sys.exit(1)

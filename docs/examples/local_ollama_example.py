#!/usr/bin/env python3
"""
Local Ollama Integration Example for HADES-PathRAG

This example demonstrates how to use HADES-PathRAG with a locally installed
Ollama service that's running as a systemd service on your machine.

It showcases:
1. Text generation with locally installed Ollama
2. Embedding generation with locally installed Ollama
3. Full PathRAG integration with knowledge graph creation and querying
"""

import os
import sys
import asyncio
import logging
from typing import List

# Add the parent directory to sys.path to import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pathrag.PathRAG import PathRAG
from src.pathrag.llm import ollama_model_complete, ollama_embed
from src.config.llm_config import get_ollama_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Example knowledge for our RAG system
EXAMPLE_KNOWLEDGE = [
    "PathRAG is a graph-based retrieval augmented generation system.",
    "HADES-PathRAG extends PathRAG with XnX notation and other features.",
    "Ollama is an open-source framework for running LLMs locally.",
    "Knowledge graphs store data as nodes connected by relationships.",
    "Local LLM inference means running AI models on your own machine.",
    "ArangoDB is a multi-model database that supports graph operations.",
    "The systemd service manager is used to run Ollama as a background service.",
    "HADES is designed to be a recursive AI system capable of self-improvement.",
]

async def check_ollama_connection():
    """Test connection to locally installed Ollama service"""
    try:
        # Get Ollama configuration (reads from .env file or environment variables)
        ollama_config = get_ollama_config()
        
        # Default host if not found in configuration
        host = ollama_config.get("host", "http://localhost:11434")
        
        # Check connection with a simple curl
        import subprocess
        result = subprocess.run(
            ["curl", f"{host}/api/tags"], 
            capture_output=True, 
            text=True
        )
        
        if result.returncode == 0 and "models" in result.stdout:
            logger.info(f"Successfully connected to local Ollama service at {host}")
            return True
        else:
            logger.error(f"Failed to connect to Ollama: {result.stderr}")
            return False
    except Exception as e:
        logger.error(f"Error checking Ollama connection: {e}")
        return False

async def test_text_generation():
    """Test text generation with locally installed Ollama"""
    try:
        # Get configuration
        ollama_config = get_ollama_config()
        
        prompt = "Explain how knowledge graphs can improve AI systems"
        system_prompt = "You are a helpful AI assistant specialized in explaining AI concepts."
        
        logger.info(f"Generating text with Ollama model: {ollama_config['model']}")
        
        # Generate text using the local Ollama service
        response = await ollama_model_complete(
            prompt=prompt,
            system_prompt=system_prompt,
            hashing_kv={"global_config": {"llm_model_name": ollama_config["model"]}},
            host=ollama_config["host"],
            timeout=ollama_config.get("timeout", 60),
            **ollama_config.get("parameters", {})
        )
        
        logger.info(f"Ollama response ({len(response)} chars):\n{response[:500]}...")
        return response
    except Exception as e:
        logger.error(f"Text generation failed: {e}")
        return None

async def test_embeddings():
    """Test embedding generation with locally installed Ollama"""
    try:
        # Get configuration
        ollama_config = get_ollama_config()
        
        # Sample texts to embed
        texts = [
            "Knowledge graphs in AI systems",
            "Local LLM inference with Ollama",
            "HADES recursive architecture"
        ]
        
        logger.info(f"Generating embeddings with model: {ollama_config['embed_model']}")
        
        # Generate embeddings using the local Ollama service
        embeddings = await ollama_embed(
            texts=texts,
            embed_model=ollama_config["embed_model"],
            host=ollama_config["host"],
            timeout=ollama_config.get("timeout", 60)
        )
        
        logger.info(f"Generated {len(embeddings)} embeddings with dimension {len(embeddings[0])}")
        return embeddings
    except Exception as e:
        logger.error(f"Embedding generation failed: {e}")
        return None

async def test_pathrag_with_local_ollama():
    """Test PathRAG with locally installed Ollama"""
    try:
        # Get configuration
        ollama_config = get_ollama_config()
        
        # Create a working directory for PathRAG if it doesn't exist
        working_dir = "./pathrag_local_example"
        os.makedirs(working_dir, exist_ok=True)
        
        logger.info("Initializing PathRAG with local Ollama")
        
        # Initialize PathRAG
        pathrag = PathRAG(
            working_dir=working_dir,
            embedding_cache_config={"enabled": True},
            kv_storage="JsonKVStorage",
            vector_storage="NanoVectorDBStorage",
            graph_storage="NetworkXStorage"
        )
        
        # Configure PathRAG to use local Ollama
        pathrag.llm_model_func = lambda prompt, **kwargs: ollama_model_complete(
            prompt=prompt,
            hashing_kv={"global_config": {"llm_model_name": ollama_config["model"]}},
            host=ollama_config["host"],
            timeout=ollama_config.get("timeout", 60),
            **ollama_config.get("parameters", {})
        )
        
        # Insert example knowledge
        logger.info(f"Inserting {len(EXAMPLE_KNOWLEDGE)} knowledge items")
        pathrag.insert(EXAMPLE_KNOWLEDGE)
        
        # Query the knowledge graph
        query = "How does HADES-PathRAG use Ollama for local inference?"
        logger.info(f"Querying PathRAG: '{query}'")
        
        response = await pathrag.query(query)
        
        logger.info(f"PathRAG response:\n{response}")
        return response
    except Exception as e:
        logger.error(f"PathRAG integration test failed: {e}")
        return None

async def main():
    """Run all tests"""
    logger.info("Starting local Ollama integration example")
    
    # First check if Ollama is properly connected
    if not await check_ollama_connection():
        logger.error("Failed to connect to local Ollama service. Make sure it's running.")
        logger.info("You can check the status with: systemctl status ollama")
        return
    
    # Test each component
    logger.info("\n=== Testing Text Generation ===")
    await test_text_generation()
    
    logger.info("\n=== Testing Embeddings ===")
    await test_embeddings()
    
    logger.info("\n=== Testing Full PathRAG Integration ===")
    await test_pathrag_with_local_ollama()
    
    logger.info("\nAll examples completed!")

if __name__ == "__main__":
    asyncio.run(main())

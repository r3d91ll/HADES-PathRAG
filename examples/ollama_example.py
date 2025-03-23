#!/usr/bin/env python3
"""
Example script demonstrating the use of Ollama as the model engine in HADES-PathRAG.

This script shows how to:
1. Configure Ollama as the LLM provider
2. Generate text using Ollama
3. Generate embeddings using Ollama
4. Use Ollama with PathRAG for knowledge retrieval
"""

import asyncio
import os
from dotenv import load_dotenv
import sys
import logging

# Ensure the src directory is in the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config.llm_config import get_llm_config, get_ollama_config
from src.pathrag.llm import ollama_model_complete, ollama_embed
from src.pathrag.PathRAG import PathRAG
from src.xnx.xnx_pathrag import XnXPathRAG
from src.xnx.xnx_params import XnXParams

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

async def test_ollama_text_generation():
    """Test Ollama text generation capabilities."""
    ollama_config = get_ollama_config()
    
    # Basic text generation example
    prompt = "Explain the concept of a knowledge graph in simple terms"
    system_prompt = "You are a helpful AI assistant specialized in explaining complex concepts simply."
    
    logger.info(f"Generating text with Ollama model: {ollama_config['model']}")
    response = await ollama_model_complete(
        prompt=prompt,
        system_prompt=system_prompt,
        hashing_kv={"global_config": {"llm_model_name": ollama_config['model']}},
        host=ollama_config['host'],
        timeout=ollama_config['timeout'],
        **ollama_config['parameters']
    )
    
    logger.info(f"Ollama response: {response}")
    return response

async def test_ollama_embeddings():
    """Test Ollama embedding capabilities."""
    ollama_config = get_ollama_config()
    
    # Text to generate embeddings for
    texts = [
        "Knowledge graphs store interconnected data",
        "HADES uses PathRAG for knowledge retrieval",
        "XnX notation enhances relationship management in knowledge graphs"
    ]
    
    logger.info(f"Generating embeddings with Ollama model: {ollama_config['embed_model']}")
    embeddings = await ollama_embed(
        texts=texts,
        embed_model=ollama_config['embed_model'],
        host=ollama_config['host'],
        timeout=ollama_config['timeout']
    )
    
    logger.info(f"Generated {len(embeddings)} embeddings, each with dimension {len(embeddings[0])}")
    return embeddings

async def test_ollama_with_pathrag():
    """Test Ollama integration with PathRAG."""
    ollama_config = get_ollama_config()
    
    # Initialize PathRAG with Ollama
    logger.info("Initializing PathRAG with Ollama integration")
    
    # Create a basic PathRAG instance
    # Note: In a real application, you would connect to a database
    pathrag = PathRAG()
    
    # Insert some knowledge
    knowledge = [
        "HADES is a recursive AI system that can improve itself.",
        "PathRAG uses path-based retrieval in knowledge graphs.",
        "XnX notation provides weighted relationships in knowledge graphs."
    ]
    pathrag.insert(knowledge)
    
    # Set up Ollama as the model provider for PathRAG
    pathrag.llm_model_func = lambda prompt, **kwargs: ollama_model_complete(
        prompt=prompt,
        hashing_kv={"global_config": {"llm_model_name": ollama_config['model']}},
        host=ollama_config['host'],
        timeout=ollama_config['timeout'],
        **ollama_config['parameters']
    )
    
    # Query using PathRAG with Ollama
    query = "How does HADES improve itself?"
    logger.info(f"Querying PathRAG with: '{query}'")
    
    result = await pathrag.query(query)
    logger.info(f"PathRAG result: {result}")
    
    return result

async def main():
    """Main function to run the examples."""
    logger.info("Starting Ollama integration examples")
    
    # Make sure Ollama server is running
    ollama_config = get_ollama_config()
    logger.info(f"Using Ollama server at: {ollama_config['host']}")
    logger.info(f"Using Ollama model: {ollama_config['model']}")
    
    # Run the examples
    try:
        # Test text generation
        await test_ollama_text_generation()
        
        # Test embeddings
        await test_ollama_embeddings()
        
        # Test PathRAG integration
        await test_ollama_with_pathrag()
        
        logger.info("All examples completed successfully!")
        
    except Exception as e:
        logger.error(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()
        
        # Provide troubleshooting tips
        logger.info("""
        Troubleshooting tips:
        1. Make sure Ollama is installed and running:
           - Install: curl -fsSL https://ollama.com/install.sh | sh
           - Run: ollama serve
        
        2. Make sure you have the required models:
           - ollama pull llama3
           - ollama pull nomic-embed-text
           
        3. Check connection to Ollama server:
           - Default URL is http://localhost:11434
           - Verify with: curl http://localhost:11434/api/tags
        """)

if __name__ == "__main__":
    asyncio.run(main())

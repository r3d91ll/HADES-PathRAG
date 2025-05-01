#!/usr/bin/env python3
"""
Test script to validate vLLM integration for HADES-PathRAG
"""
import asyncio
import os
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from loguru import logger
from src.config.vllm_config import VLLMConfig
from src.pathrag.vllm_server import VLLMServerManager
from src.pathrag.vllm_adapter import vllm_model_complete, vllm_embed


async def test_model_complete():
    """Test model completion with vLLM"""
    logger.info("Testing vLLM model completion...")
    
    # Test with default model
    prompt = "What is PathRAG and how does it work?"
    response = await vllm_model_complete(
        prompt=prompt,
        model_alias="default"
    )
    
    logger.info(f"Prompt: {prompt}")
    logger.info(f"Response: {response}")
    
    # Test with code model
    code_prompt = "Write a Python function to calculate the Fibonacci sequence"
    code_response = await vllm_model_complete(
        prompt=code_prompt,
        model_alias="code"
    )
    
    logger.info(f"Code Prompt: {code_prompt}")
    logger.info(f"Code Response: {code_response}")
    
    return True


async def test_embeddings():
    """Test embedding generation with vLLM"""
    logger.info("Testing vLLM embeddings...")
    
    texts = [
        "PathRAG is a path-aware retrieval augmented generation system.",
        "ISNE provides graph-based document understanding.",
        "Chonky enables semantic chunking for better document processing."
    ]
    
    embeddings = await vllm_embed(texts=texts, model_alias="embedding")
    
    logger.info(f"Generated {len(embeddings)} embeddings")
    logger.info(f"Embedding shape: {embeddings.shape}")
    
    return True


async def test_server_manager():
    """Test vLLM server manager capabilities"""
    logger.info("Testing vLLM server manager...")
    
    config = VLLMConfig()
    manager = VLLMServerManager(config)
    
    # Test server startup
    logger.info("Starting server for general model...")
    await manager.ensure_server_running("general")
    
    # Check server status
    status = await manager.check_server_status()
    logger.info(f"Server status: {status}")
    
    # Test server switching to a different model
    logger.info("Switching to code model...")
    await manager.ensure_server_running("code")
    
    return True


async def main():
    """Run all tests"""
    logger.info("Starting vLLM integration tests")
    
    try:
        logger.info("===== Testing Server Manager =====")
        await test_server_manager()
        
        logger.info("\n===== Testing Model Completion =====")
        await test_model_complete()
        
        logger.info("\n===== Testing Embeddings =====")
        await test_embeddings()
        
        logger.success("All tests completed successfully!")
    except Exception as e:
        logger.error(f"Test failed: {e}")
        raise
    

if __name__ == "__main__":
    asyncio.run(main())

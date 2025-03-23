#!/usr/bin/env python3
"""
Comprehensive test script for the Ollama integration in HADES-PathRAG.

This script tests:
1. Configuration loading
2. Ollama connectivity
3. Text generation
4. Embedding generation
5. PathRAG integration
6. XnX-PathRAG integration

Run this script to verify that your Ollama integration is working correctly.
"""

import os
import sys
import asyncio
import logging
from typing import Dict, List, Any, Optional
import unittest

# Ensure the src directory is in the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config.llm_config import get_llm_config, get_ollama_config
from src.config.mcp_config import get_mcp_config, get_pathrag_config
from src.pathrag.llm import ollama_model_complete, ollama_embed
from src.pathrag.PathRAG import PathRAG
from src.xnx.xnx_pathrag import XnXPathRAG
from src.xnx.xnx_params import XnXParams

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OllamaIntegrationTest(unittest.TestCase):
    """Test suite for Ollama integration in HADES-PathRAG."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment."""
        cls.llm_config = get_llm_config()
        cls.ollama_config = get_ollama_config()
        cls.pathrag_config = get_pathrag_config()
        
        # Skip tests if Ollama is not the configured provider
        if cls.llm_config["provider"] != "ollama":
            logger.warning("Ollama is not the configured provider. Tests will be skipped.")
            cls.skip_tests = True
        else:
            cls.skip_tests = False
            
        # Test data
        cls.test_prompt = "Explain the concept of a knowledge graph in simple terms"
        cls.test_system_prompt = "You are a helpful AI assistant specialized in explaining complex concepts simply."
        cls.test_embeddings_texts = [
            "Knowledge graphs store interconnected data",
            "HADES uses PathRAG for knowledge retrieval",
            "XnX notation enhances relationship management in knowledge graphs"
        ]
        cls.test_knowledge = [
            "HADES is a recursive AI system that can improve itself.",
            "PathRAG uses path-based retrieval in knowledge graphs.",
            "XnX notation provides weighted relationships in knowledge graphs.",
            "Ollama allows running large language models locally.",
            "ArangoDB is a multi-model database with graph capabilities."
        ]
        cls.test_query = "How does HADES improve itself?"

    async def async_test_ollama_connection(self):
        """Test connection to Ollama server."""
        if self.skip_tests:
            self.skipTest("Ollama provider not configured")
            
        import ollama
        
        # Test Ollama server connection
        client = ollama.Client(host=self.ollama_config["host"])
        try:
            models = client.list()
            logger.info(f"Connected to Ollama server. Available models: {[m['name'] for m in models['models']]}")
            
            # Check if required models are available
            required_models = [self.ollama_config["model"], self.ollama_config["embed_model"]]
            available_models = [m["name"] for m in models["models"]]
            
            for model in required_models:
                self.assertIn(model, available_models, f"Required model '{model}' not found in Ollama")
                
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Ollama server: {e}")
            self.fail(f"Failed to connect to Ollama server: {e}")
            return False

    async def async_test_text_generation(self):
        """Test text generation with Ollama."""
        if self.skip_tests:
            self.skipTest("Ollama provider not configured")
            
        # Test basic text generation
        logger.info(f"Testing text generation with Ollama model: {self.ollama_config['model']}")
        
        try:
            response = await ollama_model_complete(
                prompt=self.test_prompt,
                system_prompt=self.test_system_prompt,
                hashing_kv={"global_config": {"llm_model_name": self.ollama_config['model']}},
                host=self.ollama_config['host'],
                timeout=self.ollama_config['timeout'],
                **self.ollama_config['parameters']
            )
            
            logger.info(f"Received response of length: {len(response)}")
            self.assertIsInstance(response, str, "Response should be a string")
            self.assertTrue(len(response) > 0, "Response should not be empty")
            
            return response
        except Exception as e:
            logger.error(f"Text generation failed: {e}")
            self.fail(f"Text generation failed: {e}")
            return None

    async def async_test_embeddings(self):
        """Test embedding generation with Ollama."""
        if self.skip_tests:
            self.skipTest("Ollama provider not configured")
            
        # Test embedding generation
        logger.info(f"Testing embeddings with Ollama model: {self.ollama_config['embed_model']}")
        
        try:
            embeddings = await ollama_embed(
                texts=self.test_embeddings_texts,
                embed_model=self.ollama_config['embed_model'],
                host=self.ollama_config['host'],
                timeout=self.ollama_config['timeout']
            )
            
            logger.info(f"Generated {len(embeddings)} embeddings, each with dimension {len(embeddings[0])}")
            
            self.assertEqual(len(embeddings), len(self.test_embeddings_texts), 
                            "Number of embeddings should match number of input texts")
            self.assertTrue(all(len(emb) > 0 for emb in embeddings), 
                           "All embeddings should have non-zero length")
            
            return embeddings
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            self.fail(f"Embedding generation failed: {e}")
            return None

    async def async_test_pathrag_integration(self):
        """Test PathRAG integration with Ollama."""
        if self.skip_tests:
            self.skipTest("Ollama provider not configured")
            
        # Create a temporary working directory for testing
        import tempfile
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Initialize PathRAG with Ollama
            logger.info("Testing PathRAG integration with Ollama")
            
            pathrag = PathRAG(
                working_dir=temp_dir,
                embedding_cache_config={"enabled": True},
                kv_storage="JsonKVStorage",
                vector_storage="NanoVectorDBStorage",
                graph_storage="NetworkXStorage"
            )
            
            # Configure PathRAG with Ollama
            ollama_config = self.ollama_config
            
            # Set up the model completion function
            pathrag.llm_model_func = lambda prompt, **kwargs: ollama_model_complete(
                prompt=prompt,
                hashing_kv={"global_config": {"llm_model_name": ollama_config['model']}},
                host=ollama_config['host'],
                timeout=ollama_config['timeout'],
                **ollama_config['parameters']
            )
            
            # Insert test knowledge
            pathrag.insert(self.test_knowledge)
            logger.info(f"Inserted {len(self.test_knowledge)} pieces of knowledge into PathRAG")
            
            # Query using PathRAG
            logger.info(f"Querying PathRAG with: '{self.test_query}'")
            response = await pathrag.query(self.test_query)
            
            logger.info(f"PathRAG response: {response}")
            self.assertIsInstance(response, str, "Response should be a string")
            self.assertTrue(len(response) > 0, "Response should not be empty")
            
            return response
        except Exception as e:
            logger.error(f"PathRAG integration test failed: {e}")
            self.fail(f"PathRAG integration test failed: {e}")
            return None
        finally:
            # Clean up the temporary directory
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_ollama_integration(self):
        """Run all Ollama integration tests."""
        # Run all tests in a single event loop
        loop = asyncio.get_event_loop()
        
        # We'll run the tests sequentially to make debugging easier
        tasks = [
            self.async_test_ollama_connection(),
            self.async_test_text_generation(),
            self.async_test_embeddings(),
            self.async_test_pathrag_integration()
        ]
        
        results = loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))
        
        # Check if any tests failed with exceptions
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.fail(f"Test {i} failed with exception: {result}")

def main():
    """Run the tests."""
    logger.info("Starting Ollama integration tests")
    unittest.main()

if __name__ == "__main__":
    main()

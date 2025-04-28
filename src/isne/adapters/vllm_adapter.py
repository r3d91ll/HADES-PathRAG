"""
vLLM adapter for accelerated embeddings.

This module provides integration with vLLM for accelerated embedding generation,
supporting both local and remote vLLM servers.
"""

import logging
import json
import requests
from typing import Dict, List, Any, Optional, Union, Tuple
import numpy as np
import time
import os
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VLLMAdapter:
    """
    Adapter for vLLM embedding generation.
    
    This class provides methods to connect to a vLLM server (either local or remote)
    and generate embeddings for documents or text content.
    """
    
    def __init__(
        self,
        model_name: str = "BAAI/bge-large-en-v1.5",
        server_url: Optional[str] = None,
        batch_size: int = 32,
        device: str = "cuda",
        normalize_embeddings: bool = True,
        max_retries: int = 3,
        timeout: int = 60,
        use_openai_api: bool = True
    ) -> None:
        """
        Initialize the vLLM adapter.
        
        Args:
            model_name: Name of the embedding model to use
            server_url: URL of the vLLM server (if None, will use localhost:8000)
            batch_size: Batch size for embedding generation
            device: Device to run the model on ('cuda' or 'cpu')
            normalize_embeddings: Whether to normalize embeddings
            max_retries: Maximum number of retries for failed requests
            timeout: Timeout for requests in seconds
            use_openai_api: Whether to use OpenAI-compatible API endpoints
        """
        self.model_name = model_name
        self.server_url = server_url or "http://localhost:8000"
        self.batch_size = batch_size
        self.device = device
        self.normalize_embeddings = normalize_embeddings
        self.max_retries = max_retries
        self.timeout = timeout
        self.use_openai_api = use_openai_api
        
        # If server URL ends with /, remove it
        if self.server_url.endswith("/"):
            self.server_url = self.server_url[:-1]
        
        # Configure API endpoints based on format
        if self.use_openai_api:
            self.embedding_endpoint = f"{self.server_url}/v1/embeddings"
            self.health_endpoint = f"{self.server_url}/v1/models"
        else:
            self.embedding_endpoint = f"{self.server_url}/v1/embeddings"
            self.health_endpoint = f"{self.server_url}/health"
        
        # Check if server is available
        self._check_server()
    
    def _check_server(self) -> bool:
        """
        Check if the vLLM server is available.
        
        Returns:
            True if server is available, False otherwise
        """
        try:
            response = requests.get(self.health_endpoint, timeout=self.timeout)
            if response.status_code == 200:
                logger.info(f"vLLM server at {self.server_url} is available")
                return True
            else:
                logger.warning(f"vLLM server at {self.server_url} returned status code {response.status_code}")
                return False
        except Exception as e:
            logger.warning(f"Could not connect to vLLM server at {self.server_url}: {e}")
            return False
    
    def _normalize(self, embeddings: List[List[float]]) -> List[List[float]]:
        """
        Normalize embeddings to unit length.
        
        Args:
            embeddings: List of embeddings to normalize
            
        Returns:
            Normalized embeddings
        """
        if not self.normalize_embeddings:
            return embeddings
        
        normalized = []
        for emb in embeddings:
            # Convert to numpy for efficient normalization
            emb_np = np.array(emb)
            norm = np.linalg.norm(emb_np)
            if norm > 0:
                emb_np = emb_np / norm
            normalized.append(emb_np.tolist())
        
        return normalized
    
    def _process_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Process a batch of texts to generate embeddings.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embeddings
        """
        # Retry mechanism for robustness
        for attempt in range(self.max_retries):
            try:
                if self.use_openai_api:
                    payload = {
                        "input": texts,
                        "model": self.model_name
                    }
                else:
                    payload = {
                        "texts": texts,
                        "model": self.model_name
                    }
                
                response = requests.post(
                    self.embedding_endpoint,
                    json=payload,
                    timeout=self.timeout
                )
                
                if response.status_code == 200:
                    data = response.json()
                    
                    if self.use_openai_api:
                        # OpenAI API format
                        embeddings = [item["embedding"] for item in data["data"]]
                    else:
                        # Custom API format
                        embeddings = data["embeddings"]
                    
                    return self._normalize(embeddings)
                else:
                    logger.warning(f"vLLM server returned status code {response.status_code}: {response.text}")
                    if attempt < self.max_retries - 1:
                        backoff = 2 ** attempt
                        logger.info(f"Retrying in {backoff} seconds...")
                        time.sleep(backoff)
                    else:
                        raise RuntimeError(f"Failed to generate embeddings after {self.max_retries} attempts")
            
            except Exception as e:
                logger.warning(f"Error in embedding generation: {e}")
                if attempt < self.max_retries - 1:
                    backoff = 2 ** attempt
                    logger.info(f"Retrying in {backoff} seconds...")
                    time.sleep(backoff)
                else:
                    raise RuntimeError(f"Failed to generate embeddings: {e}")
        
        # If we get here, all retries failed
        raise RuntimeError("Failed to generate embeddings")
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embeddings
        """
        if not texts:
            return []
        
        # Filter out empty texts
        non_empty_indices = []
        non_empty_texts = []
        for i, text in enumerate(texts):
            if text and text.strip():
                non_empty_indices.append(i)
                non_empty_texts.append(text)
        
        # Process in batches
        all_embeddings = []
        for i in range(0, len(non_empty_texts), self.batch_size):
            batch = non_empty_texts[i:i+self.batch_size]
            batch_embeddings = self._process_batch(batch)
            all_embeddings.extend(batch_embeddings)
        
        # Reconstruct the full embedding list with empty placeholders
        result = []
        embedding_idx = 0
        for i in range(len(texts)):
            if i in non_empty_indices:
                result.append(all_embeddings[embedding_idx])
                embedding_idx += 1
            else:
                # For empty texts, use a zero vector of the same dimension as other embeddings
                if all_embeddings:
                    dim = len(all_embeddings[0])
                    result.append([0.0] * dim)
                else:
                    # If no embeddings were generated, use a default dimension
                    result.append([0.0] * 1024)
        
        return result


def start_vllm_server(
    model_name: str = "BAAI/bge-large-en-v1.5",
    port: int = 8000,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.8,
    use_openai_api: bool = True
) -> Optional[str]:
    """
    Start a vLLM server for embeddings.
    
    Args:
        model_name: Name of the embedding model to use
        port: Port for the server
        tensor_parallel_size: Number of GPUs to use in parallel
        gpu_memory_utilization: Fraction of GPU memory to use
        use_openai_api: Whether to use OpenAI-compatible API
        
    Returns:
        Command to run the server or None if startup fails
    """
    try:
        # Check if vLLM is available
        try:
            import vllm
            logger.info(f"vLLM version {vllm.__version__} is available")
        except ImportError:
            logger.error("vLLM is not installed. Please install it with 'pip install vllm'")
            return None
        
        # Build the command for starting the server
        api_flag = "--api-implementation openai" if use_openai_api else ""
        
        command = (
            f"python -m vllm.entrypoints.openai.api_server "
            f"--model {model_name} "
            f"--port {port} "
            f"--tensor-parallel-size {tensor_parallel_size} "
            f"--gpu-memory-utilization {gpu_memory_utilization} "
            f"{api_flag}"
        )
        
        logger.info(f"vLLM server command: {command}")
        return command
        
    except Exception as e:
        logger.error(f"Error preparing vLLM server: {e}")
        return None

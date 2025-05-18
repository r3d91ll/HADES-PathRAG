#!/usr/bin/env python
"""
Start a vLLM server for HADES-PathRAG using the configuration file.

This script reads the vllm_config.yaml and starts a vLLM server with the
specified models and configuration.
"""

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config.vllm_config import VLLMConfig
from src.model_engine.adapters.vllm_adapter import start_vllm_server

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    """Parse arguments and start the vLLM server."""
    parser = argparse.ArgumentParser(description="Start vLLM server for HADES-PathRAG")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to vllm_config.yaml file")
    parser.add_argument("--model-type", type=str, choices=["embedding", "chunking", "relationship"],
                        default="embedding", help="Which ingestion model to use")
    parser.add_argument("--tensor-parallel", type=int, default=None,
                        help="Override tensor parallel size from config")
    parser.add_argument("--port", type=int, default=None,
                        help="Override port from config")
    parser.add_argument("--gpu-mem", type=float, default=None,
                        help="Override GPU memory utilization from config")
    args = parser.parse_args()

    # Load configuration
    config = VLLMConfig.load_from_yaml(args.config)
    logger.info(f"Loaded configuration from {args.config if args.config else 'default location'}")
    
    # Get ingestion model configuration
    if args.model_type not in config.get_ingestion_models():
        available_models = ", ".join(config.get_ingestion_models().keys())
        logger.error(f"Model type '{args.model_type}' not found in configuration.")
        logger.error(f"Available models: {available_models}")
        sys.exit(1)
    
    model_config = config.get_ingestion_models()[args.model_type]
    model_id = model_config.model_id
    logger.info(f"Using model: {model_id}")
    
    # Override configuration with command line arguments
    server_config = config.server
    port = args.port if args.port is not None else server_config.port
    tensor_parallel_size = args.tensor_parallel if args.tensor_parallel is not None else server_config.tensor_parallel_size
    gpu_memory_utilization = args.gpu_mem if args.gpu_mem is not None else server_config.gpu_memory_utilization
    
    # Start server
    command = start_vllm_server(
        model_name=model_id,
        port=port,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        use_openai_api=True
    )
    
    logger.info(f"Starting vLLM server with command: {command}")
    try:
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to start vLLM server: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Shutting down vLLM server...")


if __name__ == "__main__":
    main()

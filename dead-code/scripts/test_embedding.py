#!/usr/bin/env python3
"""
Test script for the embedding module.

This script reads chunked documents from test-output/chunking,
processes them using the embedding module, and saves the results
to test-output/embedding.
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Union

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import embedding module
from src.embedding.processors import process_chunked_documents_batch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("test_embedding")


async def main_async():
    """Main async function."""
    parser = argparse.ArgumentParser(description="Test the embedding module")
    parser.add_argument("--input", "-i", default="./test-output/chunking", help="Input directory with chunked documents")
    parser.add_argument("--output", "-o", default="./test-output/embedding", help="Output directory for embedded documents")
    parser.add_argument("--adapter", "-a", default="cpu", help="Embedding adapter to use")
    parser.add_argument("--model", "-m", default="all-MiniLM-L6-v2", help="Model name for embedding")
    parser.add_argument("--max-length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for processing")
    args = parser.parse_args()
    
    # Set up directories
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not input_dir.exists():
        logger.error(f"Input directory {input_dir} does not exist")
        return
    
    # Find chunked document files
    chunked_files = list(input_dir.glob("*_chunked.json"))
    logger.info(f"Found {len(chunked_files)} chunked documents in {input_dir}")
    
    if not chunked_files:
        logger.error(f"No chunked documents found in {input_dir}")
        return
    
    # Configure adapter options
    adapter_options = {
        "model_name": args.model,
        "max_length": args.max_length,
        "batch_size": args.batch_size
    }
    
    # Process documents with embeddings
    logger.info(f"Processing documents with {args.adapter} adapter and model {args.model}")
    stats = await process_chunked_documents_batch(
        file_paths=chunked_files,
        output_dir=output_dir,
        adapter_name=args.adapter,
        adapter_options=adapter_options
    )
    
    # Print statistics
    logger.info(f"Embedding processing complete")
    logger.info(f"Processed {stats['total']} documents, {stats['successful']} successful, {stats['failed']} failed")
    
    # Save a summary of the results
    summary_file = output_dir / "embedding_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(
            {
                "adapter": args.adapter,
                "model": args.model,
                "statistics": stats
            }, 
            f, 
            indent=2
        )
    
    # Print path to a sample embedding file for inspection
    if stats['successful'] > 0:
        sample_file = next((d['output_file'] for d in stats['details'] if d['success']), None)
        if sample_file:
            logger.info(f"Sample embedded document: {sample_file}")


def main():
    """Main entry point."""
    asyncio.run(main_async())


if __name__ == "__main__":
    main()

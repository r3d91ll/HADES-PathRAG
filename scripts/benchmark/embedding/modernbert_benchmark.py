#!/usr/bin/env python
"""
Performance benchmark for ModernBERT embedding adapter.

This script measures performance metrics for the ModernBERT embedding adapter
with various chunk sizes to evaluate its effectiveness with academic papers.
"""

import asyncio
import time
import argparse
import logging
import json
from pathlib import Path
from typing import Dict, List, Any, Optional

from src.embedding.base import get_adapter
from src.types.common import EmbeddingVector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("modernbert_benchmark")

async def benchmark_embedding(
    texts: List[str],
    batch_size: int = 8,
    adapter_name: str = "modernbert",
    pooling_strategy: str = "cls"
) -> Dict[str, Any]:
    """
    Benchmark the embedding generation process for a list of texts.
    
    Args:
        texts: List of texts to embed
        batch_size: Size of batches for processing
        adapter_name: Name of the embedding adapter to use
        pooling_strategy: Pooling strategy for embeddings (cls, mean, max)
        
    Returns:
        Dictionary with benchmark metrics
    """
    # Initialize performance metrics
    metrics: Dict[str, Any] = {
        "total_texts": len(texts),
        "total_tokens": sum(len(text.split()) for text in texts),
        "adapter": adapter_name,
        "pooling": pooling_strategy,
        "batch_size": batch_size,
        "text_lengths": [],
        "embedding_times": [],
        "embeddings_per_second": 0,
        "tokens_per_second": 0,
        "total_time_seconds": 0,
        "avg_time_per_text_ms": 0,
        "embedding_dimensions": 0
    }
    
    # Get text lengths for analysis
    metrics["text_lengths"] = [len(text.split()) for text in texts]
    
    # Initialize adapter
    logger.info(f"Initializing {adapter_name} adapter...")
    adapter = get_adapter(adapter_name, pooling_strategy=pooling_strategy)
    
    # Measure total time
    start_time = time.time()
    
    # Process texts in batches
    embeddings: List[EmbeddingVector] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch_start = time.time()
        
        # Generate embeddings for the batch
        batch_embeddings = await adapter.embed(batch)
        embeddings.extend(batch_embeddings)
        
        # Record batch time
        batch_time = time.time() - batch_start
        metrics["embedding_times"].append(batch_time)
        
        logger.info(f"Processed batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size} "
                   f"in {batch_time:.2f}s ({len(batch)/batch_time:.2f} texts/s)")
    
    # Calculate final metrics
    total_time = time.time() - start_time
    metrics["total_time_seconds"] = total_time
    metrics["avg_time_per_text_ms"] = (total_time / len(texts)) * 1000
    metrics["embeddings_per_second"] = len(texts) / total_time
    metrics["tokens_per_second"] = metrics["total_tokens"] / total_time
    
    # Record embedding dimensions if embeddings were generated
    if embeddings and len(embeddings) > 0:
        if isinstance(embeddings[0], list):
            metrics["embedding_dimensions"] = len(embeddings[0])
        else:
            metrics["embedding_dimensions"] = embeddings[0].shape[0]
    
    return metrics

async def run_benchmark(
    input_file: str,
    max_texts: Optional[int] = None,
    batch_size: int = 8,
    output_file: Optional[str] = None
) -> None:
    """
    Run the benchmark on texts from an input file.
    
    Args:
        input_file: Path to the input file with texts
        max_texts: Maximum number of texts to process
        batch_size: Batch size for processing
        output_file: Path to write the results
    """
    # Load the input file
    input_path = Path(input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    # Read texts based on file extension
    texts: List[str] = []
    if input_path.suffix.lower() == ".json":
        with open(input_path, "r") as f:
            data = json.load(f)
            if isinstance(data, list):
                # Try to extract text from a list of dictionaries
                if all(isinstance(item, dict) for item in data):
                    for item in data:
                        if "content" in item:
                            texts.append(item["content"])
                        elif "text" in item:
                            texts.append(item["text"])
                        elif "chunk" in item:
                            texts.append(item["chunk"])
                else:
                    # Assume it's a list of strings
                    texts = [str(item) for item in data]
            elif isinstance(data, dict):
                # Try to extract chunks from a document structure
                if "chunks" in data and isinstance(data["chunks"], list):
                    for chunk in data["chunks"]:
                        if isinstance(chunk, dict) and "content" in chunk:
                            texts.append(chunk["content"])
    else:
        # Plain text file, one document per line
        with open(input_path, "r") as f:
            texts = [line.strip() for line in f if line.strip()]
    
    # Limit the number of texts if specified
    if max_texts is not None and max_texts > 0:
        texts = texts[:max_texts]
    
    if not texts:
        logger.error("No texts found in input file")
        return
    
    logger.info(f"Loaded {len(texts)} texts from {input_file}")
    logger.info(f"Average text length: {sum(len(t.split()) for t in texts) / len(texts):.1f} tokens")
    
    # Run the benchmark
    adapter_metrics = {}
    
    # Test ModernBERT with different pooling strategies
    for pooling in ["cls", "mean", "max"]:
        logger.info(f"Benchmarking ModernBERT with {pooling} pooling...")
        metrics = await benchmark_embedding(
            texts=texts,
            batch_size=batch_size,
            adapter_name="modernbert",
            pooling_strategy=pooling
        )
        adapter_metrics[f"modernbert_{pooling}"] = metrics
    
    # Also test the CPU adapter for comparison if available
    try:
        logger.info("Benchmarking CPU adapter for comparison...")
        metrics = await benchmark_embedding(
            texts=texts,
            batch_size=batch_size,
            adapter_name="cpu", 
            pooling_strategy="mean"
        )
        adapter_metrics["cpu"] = metrics
    except Exception as e:
        logger.warning(f"CPU adapter benchmark failed: {e}")
    
    # Compile results
    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "input_file": str(input_path),
        "num_texts": len(texts),
        "batch_size": batch_size,
        "metrics": adapter_metrics
    }
    
    # Print summary
    print("\nBenchmark Results:")
    print(f"{'Adapter':<20} {'Pooling':<8} {'Texts/s':<10} {'Tokens/s':<12} {'Avg Time (ms)':<15} {'Dimensions':<10}")
    print("-" * 80)
    
    for name, metrics in adapter_metrics.items():
        adapter = metrics["adapter"]
        pooling = metrics["pooling"]
        texts_per_second = metrics["embeddings_per_second"]
        tokens_per_second = metrics["tokens_per_second"]
        avg_time_ms = metrics["avg_time_per_text_ms"]
        dims = metrics["embedding_dimensions"]
        
        print(f"{adapter:<20} {pooling:<8} {texts_per_second:<10.2f} {tokens_per_second:<12.2f} {avg_time_ms:<15.2f} {dims:<10}")
    
    # Save results if output file specified
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark ModernBERT embedding adapter")
    parser.add_argument("--input", "-i", type=str, required=True, 
                        help="Input file with texts to embed (JSON or text)")
    parser.add_argument("--max-texts", "-m", type=int, default=None,
                        help="Maximum number of texts to process")
    parser.add_argument("--batch-size", "-b", type=int, default=8,
                        help="Batch size for processing")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output file for results (JSON)")
    
    args = parser.parse_args()
    
    # Run the benchmark
    asyncio.run(run_benchmark(
        input_file=args.input,
        max_texts=args.max_texts,
        batch_size=args.batch_size,
        output_file=args.output
    ))

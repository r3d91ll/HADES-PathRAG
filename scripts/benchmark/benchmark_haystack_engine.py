"""
Benchmark script for comparing CPU vs GPU performance of the Haystack model engine
with ModnerBERT models used in the document processing and chunking pipeline.
"""

import time
import argparse
import logging
import numpy as np
from typing import List, Dict, Any, Tuple
import os

# Setup logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("benchmark")

# Import model engine
from src.model_engine.engines.haystack import HaystackModelEngine

def run_embedding_benchmark(engine: HaystackModelEngine, 
                           texts: List[str],
                           model_id: str,
                           batch_sizes: List[int],
                           runs_per_batch: int = 5) -> Dict[int, Dict[str, float]]:
    """
    Run embedding benchmark with various batch sizes.
    
    Args:
        engine: The model engine to benchmark
        texts: List of texts to embed
        model_id: Model ID to use
        batch_sizes: List of batch sizes to test
        runs_per_batch: Number of runs for each batch size
        
    Returns:
        Dictionary mapping batch size to timing statistics
    """
    results = {}
    
    # Ensure model is loaded
    logger.info(f"Loading model {model_id}")
    engine.load_model(model_id)
    
    for batch_size in batch_sizes:
        logger.info(f"Testing batch size: {batch_size}")
        batch_times = []
        
        for run in range(runs_per_batch):
            start_time = time.time()
            
            # Process in batches
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                # Generate embeddings for the batch
                _ = engine.generate_embeddings(batch, model_id)
            
            end_time = time.time()
            batch_time = end_time - start_time
            batch_times.append(batch_time)
            logger.info(f"  Run {run+1}/{runs_per_batch}: {batch_time:.4f}s")
        
        # Calculate statistics
        results[batch_size] = {
            "mean": np.mean(batch_times),
            "median": np.median(batch_times),
            "min": np.min(batch_times),
            "max": np.max(batch_times),
            "std": np.std(batch_times)
        }
        
    return results

def generate_test_data(num_samples: int = 100, 
                      min_length: int = 50, 
                      max_length: int = 500) -> List[str]:
    """
    Generate test data for benchmarking.
    
    Args:
        num_samples: Number of text samples to generate
        min_length: Minimum length of each sample in words
        max_length: Maximum length of each sample in words
        
    Returns:
        List of text strings
    """
    import random
    
    # Sample words to create random texts
    words = ["the", "be", "to", "of", "and", "a", "in", "that", "have", "I", 
            "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
            "this", "but", "his", "by", "from", "they", "we", "say", "her", "she",
            "or", "an", "will", "my", "one", "all", "would", "there", "their", "what",
            "so", "up", "out", "if", "about", "who", "get", "which", "go", "me",
            "document", "code", "function", "class", "method", "variable", "algorithm",
            "data", "structure", "implementation", "software", "development", "programming",
            "model", "neural", "network", "learning", "artificial", "intelligence", "machine"]
    
    texts = []
    for _ in range(num_samples):
        length = random.randint(min_length, max_length)
        text = " ".join(random.choices(words, k=length))
        texts.append(text)
    
    return texts

def main():
    parser = argparse.ArgumentParser(description="Benchmark CPU vs GPU performance for Haystack model engine")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], 
                       help="Device to run benchmark on (cpu or cuda)")
    parser.add_argument("--model", type=str, default="bert-base-uncased",
                       help="Model ID to benchmark")
    parser.add_argument("--num-samples", type=int, default=100,
                       help="Number of text samples to use")
    parser.add_argument("--batch-sizes", type=str, default="1,2,4,8,16,32",
                       help="Comma-separated list of batch sizes to test")
    parser.add_argument("--runs-per-batch", type=int, default=3,
                       help="Number of runs for each batch size")
    
    args = parser.parse_args()
    
    # Parse batch sizes
    batch_sizes = [int(bs) for bs in args.batch_sizes.split(",")]
    
    # Set up environment for device
    if args.device == "cuda":
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use first GPU
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Disable GPU
    
    # Log system info
    import platform
    import torch
    
    logger.info("System information:")
    logger.info(f"  Platform: {platform.platform()}")
    logger.info(f"  Python: {platform.python_version()}")
    logger.info(f"  PyTorch: {torch.__version__}")
    logger.info(f"  Device: {args.device}")
    if args.device == "cuda" and torch.cuda.is_available():
        logger.info(f"  GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"  CUDA: {torch.version.cuda}")
    
    # Initialize model engine
    logger.info(f"Initializing Haystack model engine")
    engine = HaystackModelEngine(device=args.device)
    
    # Generate test data
    logger.info(f"Generating {args.num_samples} test samples")
    test_data = generate_test_data(args.num_samples)
    
    # Run benchmarks
    logger.info(f"Starting benchmark")
    results = run_embedding_benchmark(
        engine=engine,
        texts=test_data,
        model_id=args.model,
        batch_sizes=batch_sizes,
        runs_per_batch=args.runs_per_batch
    )
    
    # Display results
    logger.info("Benchmark results:")
    logger.info(f"{'Batch Size':<10} {'Mean (s)':<10} {'Median (s)':<10} {'Min (s)':<10} {'Max (s)':<10} {'Std Dev':<10}")
    logger.info("-" * 60)
    for batch_size, stats in sorted(results.items()):
        logger.info(f"{batch_size:<10} {stats['mean']:<10.4f} {stats['median']:<10.4f} {stats['min']:<10.4f} {stats['max']:<10.4f} {stats['std']:<10.4f}")

if __name__ == "__main__":
    main()

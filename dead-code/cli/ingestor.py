"""
Command-line interface for the ingestion pipeline.

This module provides a command-line interface for the ingestion pipeline,
including the GPU-orchestrated batch engine.
"""

import argparse
import asyncio
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Union

import yaml

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.engine.orchestrator import run_pipeline


def setup_logging(verbose: bool = False) -> None:
    """
    Set up logging.
    
    Args:
        verbose: Whether to enable verbose logging
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )


async def orchestrate_command(args: argparse.Namespace) -> None:
    """
    Run the orchestrator command.
    
    Args:
        args: Command-line arguments
    """
    # Set up logging
    setup_logging(args.verbose)
    
    # Run the pipeline
    metrics = await run_pipeline(
        input_dir=args.input,
        config_path=args.config,
        batch_size=args.batch_size,
        queue_depth=args.queue_depth,
        recursive=not args.no_recursive
    )
    
    # Print metrics
    if args.output:
        # Write metrics to a file
        with open(args.output, 'w') as f:
            yaml.dump(metrics, f, default_flow_style=False)
    else:
        # Print metrics to console
        print("\nPipeline Metrics:")
        print(f"  Total batches: {metrics['total_batches']}")
        print(f"  Total documents: {metrics['total_documents']}")
        print(f"  Total chunks: {metrics['total_chunks']}")
        print(f"  Errors: {metrics['errors']}")
        print(f"  Elapsed time: {metrics['elapsed_seconds']:.2f}s")
        
        # Print stage metrics
        print("\nStage Metrics:")
        for stage_name, stage_metrics in metrics['stage_metrics'].items():
            print(f"  {stage_name}:")
            for key, value in stage_metrics.items():
                if isinstance(value, float):
                    print(f"    {key}: {value:.2f}")
                else:
                    print(f"    {key}: {value}")


def main() -> None:
    """
    Main entry point for the command-line interface.
    """
    parser = argparse.ArgumentParser(description="HADES-PathRAG Ingestion Pipeline")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Orchestrate command
    orchestrate_parser = subparsers.add_parser(
        "orchestrate", help="Run the GPU-orchestrated batch engine"
    )
    orchestrate_parser.add_argument(
        "--input", "-i", required=True, help="Directory containing input files"
    )
    orchestrate_parser.add_argument(
        "--config", "-c", help="Path to the engine configuration file"
    )
    orchestrate_parser.add_argument(
        "--batch-size", "-b", type=int, help="Override the batch size from the configuration"
    )
    orchestrate_parser.add_argument(
        "--queue-depth", "-q", type=int, help="Override the queue depth from the configuration"
    )
    orchestrate_parser.add_argument(
        "--no-recursive", "-n", action="store_true", help="Do not process subdirectories"
    )
    orchestrate_parser.add_argument(
        "--output", "-o", help="Path to write metrics to (YAML format)"
    )
    orchestrate_parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Run the appropriate command
    if args.command == "orchestrate":
        asyncio.run(orchestrate_command(args))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

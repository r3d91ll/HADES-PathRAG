"""
Command-line interface for the HADES-PathRAG ingestion pipeline.

This module provides a CLI for running the ingestion pipeline to 
load data, compute ISNE embeddings, and store it in ArangoDB.
"""
import argparse
import logging
import sys
from pathlib import Path
import json

from hades_pathrag.ingestion.pipeline import create_pipeline_from_yaml_config


def setup_logging(verbose: bool = False) -> None:
    """Set up logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler()],
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="HADES-PathRAG Data Ingestion Pipeline")
    
    parser.add_argument(
        "source",
        help="Source path for data ingestion (directory, JSON, or CSV)"
    )
    
    parser.add_argument(
        "--loader",
        choices=["text_directory", "json", "csv"],
        help="Loader type to use (if not specified, auto-detect based on source)"
    )
    
    parser.add_argument(
        "--dataset-name",
        help="Name for the dataset (defaults to source path basename)"
    )
    
    parser.add_argument(
        "--skip-embeddings",
        action="store_true",
        help="Skip computing ISNE embeddings"
    )
    
    parser.add_argument(
        "--skip-storage",
        action="store_true",
        help="Skip storing data in ArangoDB"
    )
    
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--output",
        help="Path to output file for processed dataset (JSON format)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--file-extensions",
        nargs="+",
        help="File extensions to include when using text_directory loader"
    )
    
    parser.add_argument(
        "--chunk-size",
        type=int,
        help="Maximum chunk size for documents"
    )
    
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=0,
        help="Overlap between chunks when splitting documents"
    )
    
    return parser.parse_args()


def main() -> int:
    """Main entry point for the CLI.
    
    Returns:
        Exit code (0 for success, 1 for error)
    """
    args = parse_args()
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    try:
        # Create pipeline from YAML config
        pipeline = create_pipeline_from_yaml_config(args.config)

        # Run the pipeline (all loader and embedding settings come from config)
        logger.info(f"Starting ingestion from {args.source}")
        dataset, stats = pipeline.ingest(
            args.source,
            loader_type=None,  # Loader type and kwargs are now set by config
            skip_embeddings=args.skip_embeddings,
            skip_storage=args.skip_storage,
        )
        
        # Print results
        logger.info(f"Ingestion complete. Processed {len(dataset.documents)} documents and "
                    f"{len(dataset.relationships)} relationships.")
        logger.info(f"Statistics: {json.dumps(stats, indent=2)}")
        
        # Save output if requested
        if args.output:
            output_path = Path(args.output)
            output_data = {
                "name": dataset.name,
                "metadata": dataset.metadata,
                "document_count": len(dataset.documents),
                "relationship_count": len(dataset.relationships),
                "stats": stats
            }
            
            with open(output_path, "w") as f:
                json.dump(output_data, f, indent=2)
            logger.info(f"Output saved to {output_path}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=args.verbose)
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

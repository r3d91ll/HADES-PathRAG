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

from hades_pathrag.mcp_server.config.settings import load_config
from hades_pathrag.ingestion.pipeline import create_pipeline_from_config


def setup_logging(verbose: bool = False):
    """Set up logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler()],
    )


def parse_args():
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


def main():
    """Main entry point for the CLI."""
    args = parse_args()
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    try:
        # Load configuration
        config = load_config(args.config)
        
        # Create pipeline
        pipeline = create_pipeline_from_config(config)
        
        # Configure the loader if needed
        loader_kwargs = {
            "dataset_name": args.dataset_name
        }
        
        if args.loader == "text_directory" and args.file_extensions:
            loader_kwargs["file_extensions"] = args.file_extensions
            
        if args.chunk_size:
            loader_kwargs["chunk_size"] = args.chunk_size
            loader_kwargs["chunk_overlap"] = args.chunk_overlap
        
        # Run the pipeline
        logger.info(f"Starting ingestion from {args.source}")
        dataset, stats = pipeline.ingest(
            args.source,
            loader_type=args.loader,
            skip_embeddings=args.skip_embeddings,
            skip_storage=args.skip_storage,
            **loader_kwargs
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
    sys.exit(main())

#!/usr/bin/env python3
"""
HTML Documentation ingestion script for testing HADES-PathRAG text processing.

This is a specialized wrapper that ensures HTML files are properly processed
using the Docling preprocessor.
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.ingest.pre_processor import PRE_PROCESSOR_REGISTRY, DoclingPreProcessor
from scripts.ingest_documentation import ingest_documentation

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def register_html_processor():
    """
    Ensure proper registration of HTML processors.
    
    This function patches the PRE_PROCESSOR_REGISTRY to make sure
    all HTML related extensions are properly mapped.
    """
    # Ensure HTML processors are registered
    PRE_PROCESSOR_REGISTRY['html'] = DoclingPreProcessor
    PRE_PROCESSOR_REGISTRY['.html'] = DoclingPreProcessor
    PRE_PROCESSOR_REGISTRY['.htm'] = DoclingPreProcessor
    PRE_PROCESSOR_REGISTRY['htm'] = DoclingPreProcessor
    
    # Register other document types for completeness
    PRE_PROCESSOR_REGISTRY['docling'] = DoclingPreProcessor
    
    logger.info("HTML preprocessors registered successfully")

def main():
    parser = argparse.ArgumentParser(description="Ingest HTML documentation into HADES-PathRAG")
    parser.add_argument("--docs-dir", type=str, required=True, help="Directory containing HTML documentation files")
    parser.add_argument("--dataset-name", type=str, help="Optional name for the dataset")
    parser.add_argument("--db-mode", type=str, choices=["create", "append"], default="append",
                       help="Database mode: 'create' to initialize new collections, 'append' to add to existing ones")
    parser.add_argument("--db-name", type=str, default="pathrag_docs", 
                       help="Name of the ArangoDB database to use")
    parser.add_argument("--force", action="store_true", 
                       help="Force creation of collections even if they already exist (only with --db-mode=create)")
    args = parser.parse_args()
    
    # Convert to Path
    docs_dir = Path(args.docs_dir)
    
    # Verify documentation directory exists
    if not docs_dir.exists() or not docs_dir.is_dir():
        logger.error(f"Documentation directory does not exist: {docs_dir}")
        return 1
    
    # Patch the preprocessor registry
    register_html_processor()
    
    # Ingest documentation
    logger.info(f"Starting HTML documentation ingestion from {docs_dir}")
    results = ingest_documentation(
        docs_dir, 
        dataset_name=args.dataset_name,
        db_name=args.db_name,
        db_mode=args.db_mode,
        force=args.force
    )
    
    logger.info(f"HTML ingestion completed. Stats: {results}")
    return 0

if __name__ == "__main__":
    sys.exit(main())

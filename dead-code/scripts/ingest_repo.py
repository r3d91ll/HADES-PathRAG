#!/usr/bin/env python3
"""
CLI script for ingesting GitHub repositories into PathRAG.

Usage:
    python ingest_repo.py https://github.com/username/repo [--name custom_name]
"""
import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime
import json

# Add the project root to the Python path
repo_root = Path(__file__).parent.parent
sys.path.append(str(repo_root))

from src.ingest.ingestor import RepositoryIngestor
from src.config import settings

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(repo_root / 'logs' / f'ingest_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger(__name__)

def main():
    """Main entry point for repository ingestion."""
    parser = argparse.ArgumentParser(description='Ingest a GitHub repository into PathRAG')
    parser.add_argument('repo_url', help='URL of the GitHub repository to ingest')
    parser.add_argument('--name', help='Custom name for the repository directory', default=None)
    parser.add_argument('--base-dir', help='Base directory to clone repositories into', default='/home/todd/ML-Lab')
    parser.add_argument('--host', help='ArangoDB host', default='localhost')
    parser.add_argument('--port', help='ArangoDB port', type=int, default=8529)
    parser.add_argument('--database', help='ArangoDB database name', default='pathrag')
    parser.add_argument('--username', help='ArangoDB username', default='root')
    parser.add_argument('--password', help='ArangoDB password', default='')
    parser.add_argument('--output', help='Output file for ingestion stats', default=None)
    
    args = parser.parse_args()
    
    # Create logs directory if it doesn't exist
    logs_dir = repo_root / 'logs'
    logs_dir.mkdir(exist_ok=True)
    
    # Initialize ingestor
    ingestor = RepositoryIngestor(
        database=args.database,
        host=args.host,
        port=args.port,
        username=args.username,
        password=args.password
    )
    
    # Start ingestion
    logger.info(f"Starting ingestion of repository: {args.repo_url}")
    print(f"Starting ingestion of repository: {args.repo_url}")
    print(f"This may take a while depending on the repository size...")
    
    start_time = datetime.now()
    success, message, stats = ingestor.ingest_repository(
        repo_url=args.repo_url,
        repo_name=args.name,
        base_dir=args.base_dir
    )
    
    end_time = datetime.now()
    execution_time = (end_time - start_time).total_seconds()
    
    # Add execution time to stats
    stats["execution_time_seconds"] = execution_time
    
    # Output results
    if success:
        print("\n✅ Repository ingestion completed successfully!")
        logger.info("Repository ingestion completed successfully")
    else:
        print("\n❌ Repository ingestion failed!")
        logger.error(f"Repository ingestion failed: {message}")
    
    print(f"\nExecution time: {execution_time:.2f} seconds")
    print(f"Nodes created: {stats['nodes_created']}")
    print(f"Edges created: {stats['edges_created']}")
    print(f"Files processed: {stats['files_processed']}")
    
    if stats.get('errors'):
        print("\nErrors encountered:")
        for error in stats['errors']:
            print(f"  - {error}")
    
    # Save stats to output file if specified
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"\nDetailed stats saved to: {args.output}")
    
    return 0 if success else 1

if __name__ == '__main__':
    sys.exit(main())

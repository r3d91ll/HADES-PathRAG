#!/usr/bin/env python3
"""
Script to compare traditional chunking with Chonky semantic chunking.

This script loads sample documents and processes them with both the 
traditional ChunkingProcessor and the new ChonkyProcessor, providing
metrics and visualizations to compare the results.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import json
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.isne.types.models import IngestDocument, DocumentRelation
from src.isne.processors.chunking_processor import ChunkingProcessor
from src.isne.processors.chonking_processor import ChonkyProcessor
from src.isne.processors.base_processor import ProcessorConfig

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_sample_documents(directory: Path, num_docs: int = 5) -> List[IngestDocument]:
    """Create sample documents from text files in the specified directory."""
    documents = []
    
    if not directory.exists():
        logger.error(f"Directory {directory} does not exist")
        return documents
    
    # Get text files from the directory
    text_files = list(directory.glob("*.txt")) + list(directory.glob("*.md"))
    if len(text_files) == 0:
        logger.error(f"No text files found in {directory}")
        return documents
    
    # Limit to requested number
    files_to_process = text_files[:num_docs]
    
    for file_path in files_to_process:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            doc = IngestDocument(
                id=str(file_path.stem),
                content=content,
                source=str(file_path),
                document_type="text" if file_path.suffix == ".txt" else "markdown",
                title=file_path.name,
                created_at=datetime.now()
            )
            documents.append(doc)
            logger.info(f"Loaded document: {doc.title} ({len(content)} chars)")
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
    
    return documents


def process_with_chunking(documents: List[IngestDocument], splitting_strategy: str = "paragraph") -> Dict[str, Any]:
    """Process documents with traditional chunking."""
    processor = ChunkingProcessor(
        processor_config=ProcessorConfig(),
        chunk_size=1000,
        chunk_overlap=200,
        splitting_strategy=splitting_strategy,
        preserve_metadata=True,
        create_relationships=True
    )
    
    start_time = datetime.now()
    result = processor.process(documents)
    end_time = datetime.now()
    
    stats = {
        "original_docs": len(documents),
        "processed_docs": len(result.documents),
        "chunks_created": len(result.documents) - len(documents),
        "relations_created": len(result.relations),
        "processing_time": (end_time - start_time).total_seconds(),
        "chunks_per_doc": (len(result.documents) - len(documents)) / max(1, len(documents)),
        "documents": result.documents,
        "relations": result.relations
    }
    
    return stats


def process_with_chonky(documents: List[IngestDocument], model_id: str = "mirth/chonky_distilbert_uncased_1") -> Dict[str, Any]:
    """Process documents with Chonky semantic chunking."""
    try:
        processor = ChonkyProcessor(
            processor_config=ProcessorConfig(),
            model_id=model_id,
            device="cuda" if torch_available else "cpu",
            preserve_metadata=True,
            create_relationships=True
        )
        
        if not processor.splitter:
            logger.error("Chonky splitter initialization failed. Check installation.")
            return {"error": "Chonky initialization failed"}
        
        start_time = datetime.now()
        result = processor.process(documents)
        end_time = datetime.now()
        
        stats = {
            "original_docs": len(documents),
            "processed_docs": len(result.documents),
            "chunks_created": len(result.documents) - len(documents),
            "relations_created": len(result.relations),
            "processing_time": (end_time - start_time).total_seconds(),
            "chunks_per_doc": (len(result.documents) - len(documents)) / max(1, len(documents)),
            "documents": result.documents,
            "relations": result.relations
        }
        
        return stats
    except Exception as e:
        logger.error(f"Error in Chonky processing: {e}")
        return {"error": str(e)}


def analyze_chunk_quality(traditional_stats: Dict[str, Any], chonky_stats: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze the quality of chunks produced by each method."""
    if "error" in chonky_stats:
        return {"error": chonky_stats["error"]}
    
    traditional_chunks = []
    chonky_chunks = []
    
    # Extract chunks (excluding original documents)
    orig_ids = set(doc.id for doc in traditional_stats["documents"][:traditional_stats["original_docs"]])
    
    for doc in traditional_stats["documents"]:
        if doc.id not in orig_ids:
            traditional_chunks.append(doc)
    
    orig_ids = set(doc.id for doc in chonky_stats["documents"][:chonky_stats["original_docs"]])
    for doc in chonky_stats["documents"]:
        if doc.id not in orig_ids:
            chonky_chunks.append(doc)
    
    # Analyze chunk length distribution
    trad_lengths = [len(chunk.content) for chunk in traditional_chunks]
    chonky_lengths = [len(chunk.content) for chunk in chonky_chunks]
    
    # Calculate statistics
    analysis = {
        "traditional": {
            "chunk_count": len(traditional_chunks),
            "min_length": min(trad_lengths) if trad_lengths else 0,
            "max_length": max(trad_lengths) if trad_lengths else 0,
            "avg_length": sum(trad_lengths) / len(trad_lengths) if trad_lengths else 0,
            "std_dev": np.std(trad_lengths) if trad_lengths else 0
        },
        "chonky": {
            "chunk_count": len(chonky_chunks),
            "min_length": min(chonky_lengths) if chonky_lengths else 0,
            "max_length": max(chonky_lengths) if chonky_lengths else 0,
            "avg_length": sum(chonky_lengths) / len(chonky_lengths) if chonky_lengths else 0,
            "std_dev": np.std(chonky_lengths) if chonky_lengths else 0
        }
    }
    
    return analysis


def visualize_comparison(traditional_stats: Dict[str, Any], chonky_stats: Dict[str, Any], 
                        analysis: Dict[str, Any], output_dir: Path) -> None:
    """Create visualizations comparing the two chunking methods."""
    if "error" in chonky_stats or "error" in analysis:
        logger.error("Cannot create visualizations due to processing errors")
        return
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Bar chart comparing chunk counts
    plt.figure(figsize=(10, 6))
    methods = ["Traditional", "Chonky"]
    counts = [traditional_stats["chunks_created"], chonky_stats["chunks_created"]]
    plt.bar(methods, counts, color=['blue', 'green'])
    plt.title('Number of Chunks Created')
    plt.ylabel('Chunk Count')
    plt.savefig(output_dir / 'chunk_count_comparison.png')
    
    # 2. Bar chart comparing processing time
    plt.figure(figsize=(10, 6))
    times = [traditional_stats["processing_time"], chonky_stats["processing_time"]]
    plt.bar(methods, times, color=['blue', 'green'])
    plt.title('Processing Time Comparison')
    plt.ylabel('Time (seconds)')
    plt.savefig(output_dir / 'processing_time_comparison.png')
    
    # 3. Box plot of chunk lengths
    plt.figure(figsize=(12, 7))
    
    # Get chunk lengths
    orig_ids = set(doc.id for doc in traditional_stats["documents"][:traditional_stats["original_docs"]])
    trad_lengths = [len(doc.content) for doc in traditional_stats["documents"] if doc.id not in orig_ids]
    
    orig_ids = set(doc.id for doc in chonky_stats["documents"][:chonky_stats["original_docs"]])
    chonky_lengths = [len(doc.content) for doc in chonky_stats["documents"] if doc.id not in orig_ids]
    
    plt.boxplot([trad_lengths, chonky_lengths], labels=methods)
    plt.title('Distribution of Chunk Lengths')
    plt.ylabel('Characters')
    plt.savefig(output_dir / 'chunk_length_distribution.png')
    
    # 4. Save summary table as text
    summary_data = [
        ["Metric", "Traditional", "Chonky"],
        ["Total Chunks", traditional_stats["chunks_created"], chonky_stats["chunks_created"]],
        ["Processing Time (s)", round(traditional_stats["processing_time"], 2), round(chonky_stats["processing_time"], 2)],
        ["Chunks per Document", round(traditional_stats["chunks_per_doc"], 2), round(chonky_stats["chunks_per_doc"], 2)],
        ["Min Chunk Length", analysis["traditional"]["min_length"], analysis["chonky"]["min_length"]],
        ["Max Chunk Length", analysis["traditional"]["max_length"], analysis["chonky"]["max_length"]],
        ["Avg Chunk Length", round(analysis["traditional"]["avg_length"], 2), round(analysis["chonky"]["avg_length"], 2)],
        ["Std Dev", round(analysis["traditional"]["std_dev"], 2), round(analysis["chonky"]["std_dev"], 2)]
    ]
    
    with open(output_dir / 'summary_metrics.txt', 'w') as f:
        f.write(tabulate(summary_data, headers="firstrow", tablefmt="grid"))
    
    logger.info(f"Visualizations and summary saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Compare traditional chunking with Chonky semantic chunking")
    parser.add_argument("--input-dir", type=str, required=True, help="Directory containing text files to process")
    parser.add_argument("--output-dir", type=str, default="./chunking_comparison", help="Directory to save results")
    parser.add_argument("--num-docs", type=int, default=5, help="Number of documents to process")
    parser.add_argument("--model-id", type=str, default="mirth/chonky_distilbert_uncased_1", help="Chonky model ID")
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    # Check if Chonky is installed
    global torch_available
    try:
        import torch
        torch_available = torch.cuda.is_available()
        logger.info(f"PyTorch CUDA available: {torch_available}")
    except ImportError:
        torch_available = False
        logger.warning("PyTorch not installed, will use CPU for processing")
    
    try:
        import chonky
        logger.info(f"Chonky version: {chonky.__version__}")
    except ImportError:
        logger.error("Chonky not installed. Please install with: pip install chonky")
        return 1
    
    # Create sample documents
    documents = create_sample_documents(input_dir, args.num_docs)
    if not documents:
        logger.error("No documents loaded. Exiting.")
        return 1
    
    logger.info(f"Loaded {len(documents)} documents for processing")
    
    # Process with traditional chunking
    logger.info("Processing with traditional chunking...")
    traditional_stats = process_with_chunking(documents)
    logger.info(f"Traditional chunking created {traditional_stats['chunks_created']} chunks "
               f"in {traditional_stats['processing_time']:.2f} seconds")
    
    # Process with Chonky
    logger.info("Processing with Chonky semantic chunking...")
    chonky_stats = process_with_chonky(documents, args.model_id)
    if "error" in chonky_stats:
        logger.error(f"Chonky processing failed: {chonky_stats['error']}")
    else:
        logger.info(f"Chonky chunking created {chonky_stats['chunks_created']} chunks "
                   f"in {chonky_stats['processing_time']:.2f} seconds")
    
    # Analyze results
    analysis = analyze_chunk_quality(traditional_stats, chonky_stats)
    if "error" in analysis:
        logger.error(f"Analysis failed: {analysis['error']}")
    else:
        logger.info("Analysis complete")
        
        # Print summary table
        summary_data = [
            ["Metric", "Traditional", "Chonky"],
            ["Total Chunks", traditional_stats["chunks_created"], chonky_stats["chunks_created"]],
            ["Processing Time (s)", round(traditional_stats["processing_time"], 2), round(chonky_stats["processing_time"], 2)],
            ["Chunks per Document", round(traditional_stats["chunks_per_doc"], 2), round(chonky_stats["chunks_per_doc"], 2)],
            ["Min Chunk Length", analysis["traditional"]["min_length"], analysis["chonky"]["min_length"]],
            ["Max Chunk Length", analysis["traditional"]["max_length"], analysis["chonky"]["max_length"]],
            ["Avg Chunk Length", round(analysis["traditional"]["avg_length"], 2), round(analysis["chonky"]["avg_length"], 2)],
            ["Std Dev", round(analysis["traditional"]["std_dev"], 2), round(analysis["chonky"]["std_dev"], 2)]
        ]
        
        print("\nChunking Comparison Summary:")
        print(tabulate(summary_data, headers="firstrow", tablefmt="grid"))
    
    # Create visualizations
    visualize_comparison(traditional_stats, chonky_stats, analysis, output_dir)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

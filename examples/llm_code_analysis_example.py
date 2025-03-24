#!/usr/bin/env python3
"""
Example script demonstrating LLM-powered code analysis for repository ingestion.

This script shows how to use the Qwen2.5 model with 128K context window to analyze
code files and extract relationships between components.
"""
import os
import sys
import logging
from pathlib import Path

# Add the project root to the Python path
repo_root = Path(__file__).parent.parent
sys.path.append(str(repo_root))

from src.ingest.llm_code_analyzer import LLMCodeAnalyzer
from src.ingest.git_operations import GitOperations
from src.config import settings

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def analyze_repository_file(repo_path: Path, file_path: Path) -> None:
    """
    Analyze a single file from a repository using the LLM-powered code analyzer.
    
    Args:
        repo_path: Path to the repository
        file_path: Path to the file to analyze
    """
    # Initialize the LLM code analyzer with the Qwen2.5 model (128K context)
    analyzer = LLMCodeAnalyzer(model_name="qwen2.5-128k")
    
    # Analyze the file
    logger.info(f"Analyzing file: {file_path}")
    analysis = analyzer.analyze_code_file(file_path)
    
    if analysis:
        # Print a summary of what was found
        print(f"\n🔎 Analysis of {file_path.relative_to(repo_path)}:")
        print(f"  Classes: {len(analysis.get('classes', []))}")
        print(f"  Functions: {len(analysis.get('functions', []))}")
        print(f"  Imports: {len(analysis.get('imports', []))}")
        
        # Print class details
        if 'classes' in analysis and analysis['classes']:
            print("\n📦 Classes:")
            for cls in analysis['classes']:
                print(f"  • {cls['name']}")
                if 'inherits_from' in cls and cls['inherits_from']:
                    print(f"    ↳ Inherits from: {', '.join(cls['inherits_from'])}")
                if 'methods' in cls and cls['methods']:
                    print(f"    ↳ Methods: {len(cls['methods'])}")
        
        # Print function details
        if 'functions' in analysis and analysis['functions']:
            print("\n🔧 Functions:")
            for func in analysis['functions']:
                print(f"  • {func['name']}")
                if 'description' in func and func['description']:
                    print(f"    ↳ {func['description']}")
    else:
        print(f"❌ Failed to analyze {file_path}")

def find_python_files(directory: Path) -> list[Path]:
    """Find all Python files in a directory recursively."""
    return list(directory.glob('**/*.py'))

def main():
    """Main entry point for the example."""
    # Example repository to analyze (can be changed to any GitHub repository)
    repo_url = "https://github.com/r3d91ll/HADES-PathRAG"
    repo_name = "HADES-PathRAG-example"
    
    # Clone the repository
    git_ops = GitOperations()
    success, message, repo_path = git_ops.clone_repository(repo_url, repo_name)
    
    if not success:
        logger.error(f"Failed to clone repository: {message}")
        return
    
    logger.info(f"Successfully cloned repository to {repo_path}")
    
    # Find Python files to analyze
    python_files = find_python_files(repo_path)
    logger.info(f"Found {len(python_files)} Python files")
    
    # Analyze a few key files as examples
    # You can modify this to analyze all files if desired
    key_files = [
        # Core files (adjust paths as needed for the repository)
        repo_path / "src" / "ingest" / "ingestor.py",
        repo_path / "src" / "xnx" / "arango_adapter.py",
    ]
    
    # Ensure the files exist, otherwise find alternative files
    files_to_analyze = []
    for file in key_files:
        if file.exists():
            files_to_analyze.append(file)
        else:
            # If specific file doesn't exist, find a suitable alternative
            logger.warning(f"File {file} not found, looking for alternatives")
            alternatives = [f for f in python_files if f.name.endswith("adapter.py") or f.name.endswith("ingestor.py")]
            if alternatives:
                files_to_analyze.append(alternatives[0])
                logger.info(f"Using alternative file: {alternatives[0]}")
    
    # If we couldn't find the specified files, use the first few Python files
    if not files_to_analyze and python_files:
        files_to_analyze = python_files[:2]  # Just analyze the first 2 files as examples
    
    # Analyze each file
    for file_path in files_to_analyze:
        analyze_repository_file(repo_path, file_path)
        print("\n" + "-" * 80 + "\n")
    
    # Demonstrate relationship extraction between two files
    if len(files_to_analyze) >= 2:
        analyzer = LLMCodeAnalyzer(model_name="qwen2.5-128k")
        
        print("\n🔄 Analyzing relationships between files:")
        print(f"File 1: {files_to_analyze[0].relative_to(repo_path)}")
        print(f"File 2: {files_to_analyze[1].relative_to(repo_path)}")
        
        # Get the analysis for each file
        analysis1 = analyzer.analyze_code_file(files_to_analyze[0])
        analysis2 = analyzer.analyze_code_file(files_to_analyze[1])
        
        if analysis1 and analysis2:
            # Extract relationships
            relationships = analyzer.extract_relationships(analysis1, analysis2)
            
            print("\n🔗 Identified relationships:")
            if relationships:
                for rel in relationships:
                    print(f"  • {rel['type']}: {rel['from_element']} → {rel['to_element']}")
                    print(f"    ↳ {rel['description']}")
                    print(f"    ↳ Confidence: {rel['confidence']:.2f}")
            else:
                print("  No relationships found between these files")
    
    print("\n✅ Example completed!")

if __name__ == "__main__":
    main()

"""
File processor for HADES-PathRAG.

This module handles file discovery, filtering, and batching for ingestion,
with type-safe interfaces and implementations.
"""

import logging
import os
from typing import Dict, List, Set, Optional, Union, Pattern
from pathlib import Path
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.types.common import PreProcessorConfig

# Set up logging
logger = logging.getLogger(__name__)


class FileProcessor:
    """
    File processor for discovering and batching files for ingestion.
    
    This class provides type-safe methods for file discovery, filtering by patterns,
    and batching for efficient processing.
    """
    
    DEFAULT_MAX_WORKERS = 4
    DEFAULT_BATCH_SIZE = 10
    
    def __init__(self, config: Optional[PreProcessorConfig] = None):
        """
        Initialize the file processor.
        
        Args:
            config: Optional pre-processor configuration
        """
        self.config = config or {}
        self.exclude_patterns = config.get('exclude_patterns', []) if config else []
        self.max_workers = config.get('max_workers', self.DEFAULT_MAX_WORKERS) if config else self.DEFAULT_MAX_WORKERS
        self.recursive = config.get('recursive', True) if config else True
        self.file_type_map = config.get('file_type_map', {}) if config else {}
        
        # Compile exclude patterns for efficiency
        self.exclude_regex: List[Pattern] = []
        for pattern in self.exclude_patterns:
            try:
                self.exclude_regex.append(re.compile(pattern))
            except re.error as e:
                logger.warning(f"Invalid exclude pattern '{pattern}': {e}")
    
    def should_exclude(self, file_path: Union[str, Path]) -> bool:
        """
        Check if a file should be excluded based on patterns.
        
        Args:
            file_path: Path to the file to check
            
        Returns:
            True if the file should be excluded, False otherwise
        """
        # Convert to string for pattern matching
        path_str = str(file_path)
        
        # Check against exclude patterns
        for pattern in self.exclude_regex:
            if pattern.search(path_str):
                return True
        
        return False
    
    def get_file_type(self, file_path: Union[str, Path]) -> str:
        """
        Determine the file type based on extension.
        
        Args:
            file_path: Path to the file
            
        Returns:
            File type string
        """
        # Normalize to Path object
        if isinstance(file_path, str):
            file_path = Path(file_path)
        
        # Get the extension without the leading dot
        ext = file_path.suffix.lstrip('.')
        
        # Check file_type_map for custom mappings
        for file_type, extensions in self.file_type_map.items():
            if ext.lower() in [e.lower().lstrip('.') for e in extensions]:
                return file_type
        
        # Default to the extension itself
        return ext.lower() or 'unknown'
    
    def collect_files(self, root_dir: Union[str, Path]) -> Dict[str, List[Path]]:
        """
        Collect files from a directory, sorted by file type.
        
        Args:
            root_dir: Root directory to collect files from
            
        Returns:
            Dictionary mapping file types to lists of file paths
        """
        # Normalize to Path object
        if isinstance(root_dir, str):
            root_dir = Path(root_dir)
        
        if not root_dir.exists() or not root_dir.is_dir():
            logger.error(f"Directory does not exist or is not a directory: {root_dir}")
            return {}
        
        # Store files by type
        files_by_type: Dict[str, List[Path]] = {}
        
        # Walk directory
        for current_dir, dirs, files in os.walk(root_dir):
            # Skip excluded directories
            current_path = Path(current_dir)
            if self.should_exclude(current_path):
                continue
            
            # Process files in this directory
            for file_name in files:
                file_path = current_path / file_name
                
                # Skip excluded files
                if self.should_exclude(file_path):
                    continue
                
                # Get file type
                file_type = self.get_file_type(file_path)
                
                # Add to the appropriate list
                if file_type not in files_by_type:
                    files_by_type[file_type] = []
                
                files_by_type[file_type].append(file_path)
            
            # If not recursive, break after the top directory
            if not self.recursive:
                break
        
        return files_by_type
    
    def create_batches(self, files_by_type: Dict[str, List[Path]], 
                      batch_size: Optional[int] = None) -> List[Dict[str, List[Path]]]:
        """
        Create batches of files for processing.
        
        Args:
            files_by_type: Dictionary mapping file types to lists of file paths
            batch_size: Number of files per batch (default: 10)
            
        Returns:
            List of batches, where each batch is a dictionary of file types to lists of file paths
        """
        if batch_size is None:
            batch_size = self.DEFAULT_BATCH_SIZE
        
        batches: List[Dict[str, List[Path]]] = []
        
        # Handle each file type separately to ensure balanced batches
        for file_type, files in files_by_type.items():
            # Create batches for this file type
            file_batches = [files[i:i + batch_size] for i in range(0, len(files), batch_size)]
            
            # Add to the overall batches
            for i, file_batch in enumerate(file_batches):
                # Extend batches list if needed
                while i >= len(batches):
                    batches.append({})
                
                # Add this file type's batch to the current batch
                batches[i][file_type] = file_batch
        
        return batches
    
    def process_directory(self, directory: Union[str, Path], 
                         batch_size: Optional[int] = None) -> List[Dict[str, List[Path]]]:
        """
        Process a directory to collect and batch files.
        
        Args:
            directory: Directory to process
            batch_size: Optional batch size
            
        Returns:
            List of batches for processing
        """
        # Collect files
        files_by_type = self.collect_files(directory)
        
        # Create batches
        return self.create_batches(files_by_type, batch_size)
    
    def process_directories(self, directories: List[Union[str, Path]], 
                           batch_size: Optional[int] = None) -> List[Dict[str, List[Path]]]:
        """
        Process multiple directories to collect and batch files.
        
        Args:
            directories: List of directories to process
            batch_size: Optional batch size
            
        Returns:
            List of batches for processing
        """
        # Collect files from all directories
        all_files_by_type: Dict[str, List[Path]] = {}
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all directory processing tasks
            future_to_dir = {
                executor.submit(self.collect_files, directory): directory
                for directory in directories
            }
            
            # Process results as they complete
            for future in as_completed(future_to_dir):
                directory = future_to_dir[future]
                try:
                    files_by_type = future.result()
                    
                    # Merge with overall results
                    for file_type, files in files_by_type.items():
                        if file_type not in all_files_by_type:
                            all_files_by_type[file_type] = []
                        
                        all_files_by_type[file_type].extend(files)
                
                except Exception as e:
                    logger.error(f"Error processing directory {directory}: {e}")
        
        # Create batches from all collected files
        return self.create_batches(all_files_by_type, batch_size)

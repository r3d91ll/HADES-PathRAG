"""
Base pre-processor interface for the ingestion pipeline.

Defines the common interface that all file-type specific pre-processors must implement.
"""

from abc import ABC, abstractmethod
import logging
import os
import traceback
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, TypedDict

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProcessedFile(TypedDict, total=False):
    """Type definition for processed file output."""
    id: str                    # Unique identifier for the document
    path: str                  # Path to the source file
    type: str                  # Type of document (e.g., 'python', 'markdown')
    content: str               # Textual content of the document
    relationships: List[Dict[str, Any]]  # Relationships to other documents
    metadata: Dict[str, Any]   # Additional metadata about the document
    functions: List[Dict[str, Any]]  # Functions found in the document (for code)
    classes: List[Dict[str, Any]]    # Classes found in the document (for code)
    imports: List[Dict[str, Any]]    # Imports found in the document (for code)


class ProcessError(TypedDict):
    """Type definition for processing errors."""
    file_path: str
    error: str
    traceback: str


class BasePreProcessor(ABC):
    """Base interface for all file-type specific pre-processors."""
    
    def __init__(self) -> None:
        """Initialize the base pre-processor."""
        self.errors: List[ProcessError] = []
    
    @abstractmethod
    def process_file(self, file_path: str) -> Optional[Dict[str, Any]]:
        """
        Process a single file.
        
        Args:
            file_path: Path to the file to process
            
        Returns:
            Processed document data
        """
        pass
    
    def process_batch(self, file_paths: List[str]) -> List[Dict[str, Any]]:
        """
        Process a batch of files with error handling and logging.
        
        Args:
            file_paths: List of paths to files to process
            
        Returns:
            List of processed document data
        """
        results = []
        self.errors = []
        
        for file_path in file_paths:
            try:
                logger.info(f"Processing file: {file_path}")
                
                # Check if file exists
                if not os.path.exists(file_path):
                    logger.warning(f"File does not exist: {file_path}")
                    continue
                    
                # Process the file
                processed = self.process_file(file_path)
                if processed is not None:
                    results.append(processed)
                else:
                    logger.warning(f"Skipping file due to preprocessing failure: {file_path}")
                logger.info(f"Successfully processed: {file_path}")
                
            except Exception as e:
                error_traceback = traceback.format_exc()
                logger.error(f"Error processing {file_path}: {e}", exc_info=True)
                
                # Record error details
                self.errors.append({
                    "file_path": file_path,
                    "error": str(e),
                    "traceback": error_traceback
                })
                
        if self.errors:
            logger.warning(f"Completed with {len(self.errors)} errors out of {len(file_paths)} files")
        else:
            logger.info(f"Successfully processed all {len(file_paths)} files")
            
        return results
        
    def get_errors(self) -> List[ProcessError]:
        """
        Get list of errors from the last batch processing.
        
        Returns:
            List of processing errors
        """
        return self.errors

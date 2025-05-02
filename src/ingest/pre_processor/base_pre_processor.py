"""
Base pre-processor interface for the ingestion pipeline.

Defines the common interface that all file-type specific pre-processors must implement.
This module now serves as an adapter layer over the new docproc module, 
maintaining backward compatibility while leveraging the new document processing
capabilities.
"""

from abc import ABC, abstractmethod
import logging
import os
import traceback
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, TypedDict

# Import new docproc module
from src.docproc.core import process_document, process_text, detect_format

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
            
        return results
        
    def get_errors(self) -> List[ProcessError]:
        """Return the list of errors encountered during processing."""
        return self.errors
        
class DocProcAdapter(BasePreProcessor):
    """
    Adapter class that wraps the new docproc module to maintain 
    backward compatibility with the BasePreProcessor interface.
    """
    
    def __init__(self, format_override: Optional[str] = None) -> None:
        """
        Initialize the docproc adapter.
        
        Args:
            format_override: Optional format to override auto-detection
        """
        super().__init__()
        self.format_override = format_override
        
    def process_file(self, file_path: str) -> Optional[Dict[str, Any]]:
        """
        Process a file using the new docproc module.
        
        Args:
            file_path: Path to the file to process
            
        Returns:
            Processed document data in the legacy format
        """
        try:
            # Process the document using the new docproc module
            result = process_document(file_path)
            
            # Convert to the legacy format expected by callers
            processed = self._convert_to_legacy_format(result, file_path)
            return processed
        except Exception as e:
            logger.error(f"Error processing {file_path} with docproc: {e}", exc_info=True)
            return None
            
    def _convert_to_legacy_format(self, result: Dict[str, Any], file_path: str) -> Dict[str, Any]:
        """
        Convert the docproc result to the legacy format.
        
        Args:
            result: Result from docproc module
            file_path: Original file path
            
        Returns:
            Data in the legacy ProcessedFile format
        """
        # Create a basic legacy format
        legacy_result: Dict[str, Any] = {
            "id": result.get("id", os.path.basename(file_path)),
            "path": file_path,
            "type": result.get("format", "unknown"),
            "content": result.get("content", ""),
            "metadata": result.get("metadata", {})
        }
        
        # Add relationships if present
        if "relationships" in result:
            legacy_result["relationships"] = result["relationships"]
            
        # Map entities to legacy fields based on type
        if "entities" in result:
            functions = []
            classes = []
            imports = []
            
            for entity in result.get("entities", []):
                entity_type = entity.get("type", "")
                if entity_type == "function":
                    functions.append(entity)
                elif entity_type == "class":
                    classes.append(entity)
                elif entity_type == "import":
                    imports.append(entity)
            
            if functions:
                legacy_result["functions"] = functions
            if classes:
                legacy_result["classes"] = classes
            if imports:
                legacy_result["imports"] = imports
                
        # Map symbols if present (from code files)
        if "symbols" in result:
            # The docproc module already separates symbols by type
            for symbol in result.get("symbols", []):
                symbol_type = symbol.get("type", "")
                if symbol_type == "function":
                    if "functions" not in legacy_result:
                        legacy_result["functions"] = []
                    legacy_result["functions"].append(symbol)
                elif symbol_type == "class":
                    if "classes" not in legacy_result:
                        legacy_result["classes"] = []
                    legacy_result["classes"].append(symbol)
                elif symbol_type == "import":
                    if "imports" not in legacy_result:
                        legacy_result["imports"] = []
                    legacy_result["imports"].append(symbol)
                        
        return legacy_result
    
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

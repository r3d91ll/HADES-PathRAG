"""
Preprocessor manager for HADES-PathRAG.

This module provides a type-safe manager to coordinate preprocessing for different file types,
delegating to appropriate preprocessors based on file extension and configuration.
"""

import logging
from typing import Dict, List, Any, Optional, Union, Set, cast, TypedDict
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.types.common import PreProcessorConfig
from src.ingest.pre_processor import get_pre_processor
from src.ingest.pre_processor.base_pre_processor import BasePreProcessor

# Set up logging
logger = logging.getLogger(__name__)

# Type alias for preprocessing results
PreprocessingResult = Dict[str, Any]


class PreprocessorManager:
    """
    Type-safe manager for preprocessing operations across different file types.
    
    This class coordinates the preprocessing of files by delegating to the
    appropriate preprocessors based on file type and handling parallelization.
    """
    
    # Default number of worker threads for parallel processing
    DEFAULT_MAX_WORKERS = 4
    
    # Standard file extension mappings
    STANDARD_MAPPINGS = {
        "py": "python",
        "python": "python",
        "md": "markdown",
        "markdown": "markdown"
    }
    
    def __init__(self, config: Optional[PreProcessorConfig] = None):
        """
        Initialize the preprocessor manager.
        
        Args:
            config: Optional preprocessor configuration
        """
        self.config = config or {}
        self.preprocessors: Dict[str, BasePreProcessor] = {}
        
        # Get the maximum number of worker threads
        self.max_workers = self.DEFAULT_MAX_WORKERS
        if self.config and 'max_workers' in self.config:
            max_workers_config = self.config.get('max_workers')
            if isinstance(max_workers_config, int) and max_workers_config > 0:
                self.max_workers = max_workers_config
        
        # Initialize standard preprocessors
        self._initialize_preprocessors()
    
    def _initialize_preprocessors(self) -> None:
        """Initialize the standard preprocessors and apply mappings."""
        # Initialize core preprocessors
        self._create_core_preprocessors()
        
        # Apply standard and custom mappings
        self._apply_mappings()
    
    def _create_core_preprocessors(self) -> None:
        """Create the core set of preprocessors."""
        for processor_type in ["python", "markdown", "docling"]:
            try:
                # Create the preprocessor
                preprocessor = get_pre_processor(processor_type)
                if preprocessor:
                    self.preprocessors[processor_type] = preprocessor
                    logger.info(f"Initialized {processor_type} preprocessor")
            except Exception as e:
                logger.error(f"Failed to initialize {processor_type} preprocessor: {e}")
    
    def _apply_mappings(self) -> None:
        """Apply standard and custom mappings for file extensions."""
        # Apply standard mappings first
        for ext, processor_type in self.STANDARD_MAPPINGS.items():
            if processor_type in self.preprocessors:
                self.preprocessors[ext] = self.preprocessors[processor_type]
        
        # Apply custom mappings if available
        if not self.config:
            return
        
        file_type_map = self.config.get('file_type_map')
        if not isinstance(file_type_map, dict):
            return
        
        # Process each custom mapping
        for ext_key in file_type_map.keys():
            # Safe extension key
            ext = ext_key
            processor_info: Any = file_type_map[ext] # Explicitly type as Any initially
            
            # Handle different processor type formats
            if isinstance(processor_info, str):
                # Handle string processor type
                p_str = processor_info
                self._map_extension(ext, p_str)
            elif isinstance(processor_info, list):
                # Handle list of processor types - try each until one works
                self._handle_processor_list(ext, processor_info)
    
    def _handle_processor_list(self, ext: str, processor_list: List[Any]) -> None:
        """Process a list of potential processor types for an extension."""
        for item in processor_list:
            if isinstance(item, str):
                if self._map_extension(ext, item):
                    # Stop after first successful mapping
                    return
    
    def _map_extension(self, extension: str, processor_type: str) -> bool:
        """
        Map a file extension to a processor type.
        
        Args:
            extension: File extension
            processor_type: Processor type
            
        Returns:
            True if mapping succeeded, False otherwise
        """
        # If processor exists, map directly
        if processor_type in self.preprocessors:
            self.preprocessors[extension] = self.preprocessors[processor_type]
            return True
        
        # Try to create the processor if it doesn't exist
        try:
            processor = get_pre_processor(processor_type)
            if processor:
                self.preprocessors[processor_type] = processor
                self.preprocessors[extension] = processor
                return True
        except Exception as e:
            logger.error(f"Failed to create processor {processor_type} for extension {extension}: {e}")
        
        return False
    
    def get_preprocessor(self, file_type: str) -> Optional[BasePreProcessor]:
        """
        Get the appropriate preprocessor for a file type.
        
        Args:
            file_type: The file type (extension or preprocessor name)
            
        Returns:
            The preprocessor if available, None otherwise
        """
        # Direct match
        if file_type in self.preprocessors:
            return self.preprocessors[file_type]
        
        # Try lowercase version
        file_type_lower = file_type.lower()
        if file_type_lower in self.preprocessors:
            return self.preprocessors[file_type_lower]
        
        # Common fallbacks
        if file_type_lower in ['py', 'python']:
            return self.preprocessors.get('python')
        elif file_type_lower in ['md', 'markdown']:
            return self.preprocessors.get('markdown')
        
        logger.warning(f"No preprocessor available for file type: {file_type}")
        return None
    
    def preprocess_file(self, file_path: Union[str, Path], 
                       file_type: Optional[str] = None) -> Optional[PreprocessingResult]:
        """
        Preprocess a single file.
        
        Args:
            file_path: Path to the file
            file_type: Optional file type (if not provided, inferred from extension)
            
        Returns:
            Preprocessing result if successful, None otherwise
        """
        # Normalize to Path
        path_obj = Path(file_path) if isinstance(file_path, str) else file_path
        
        # Infer file type if not provided
        if file_type is None:
            ext = path_obj.suffix.lstrip('.')
            file_type = ext.lower() or 'unknown'
        
        # Get the appropriate preprocessor
        preprocessor = self.get_preprocessor(file_type)
        if not preprocessor:
            logger.warning(f"No preprocessor available for file: {path_obj}")
            return None
        
        try:
            # Preprocess the file
            result = preprocessor.process_file(str(path_obj))
            return result
        except Exception as e:
            logger.error(f"Error preprocessing file {path_obj}: {e}")
            return None
    
    def preprocess_batch(self, files_by_type: Dict[str, List[Path]]) -> Dict[str, List[PreprocessingResult]]:
        """
        Preprocess a batch of files.
        
        Args:
            files_by_type: Dictionary mapping file types to lists of file paths
            
        Returns:
            Dictionary mapping file types to lists of preprocessing results
        """
        results: Dict[str, List[PreprocessingResult]] = {}
        
        # Process each file type
        for file_type, files in files_by_type.items():
            # Skip if no files
            if not files:
                continue
                
            # Get the appropriate preprocessor
            preprocessor = self.get_preprocessor(file_type)
            if not preprocessor:
                logger.warning(f"No preprocessor available for file type: {file_type}")
                continue
            
            # Initialize results list for this file type
            results[file_type] = []
            
            # Process files in parallel
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all file processing tasks
                future_to_file = {
                    executor.submit(self.preprocess_file, file_path, file_type): file_path
                    for file_path in files
                }
                
                # Process results as they complete
                for future in as_completed(future_to_file):
                    file_path = future_to_file[future]
                    try:
                        result = future.result()
                        if result:
                            results[file_type].append(result)
                    except Exception as e:
                        logger.error(f"Error processing file {file_path}: {e}")
        
        return results
    
    def extract_entities_and_relationships(self, preprocessing_results: Dict[str, List[PreprocessingResult]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Extract entities and relationships from preprocessing results.
        
        Args:
            preprocessing_results: Dictionary mapping file types to lists of preprocessing results
            
        Returns:
            Dictionary with entities and relationships
        """
        entities: List[Dict[str, Any]] = []
        relationships: List[Dict[str, Any]] = []
        
        # Process each file type's results
        for file_type, results_list in preprocessing_results.items():
            for result in results_list:
                # Extract entities
                if 'entities' in result and isinstance(result['entities'], list):
                    entities.extend(result['entities'])
                
                # Extract relationships
                if 'relationships' in result and isinstance(result['relationships'], list):
                    relationships.extend(result['relationships'])
        
        return {
            'entities': entities,
            'relationships': relationships
        }

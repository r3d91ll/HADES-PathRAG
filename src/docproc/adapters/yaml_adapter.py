"""
YAML adapter for document processing.

This module provides functionality to process YAML documents.
"""

import hashlib
import re
from pathlib import Path
from typing import Dict, Any, Optional, List, Union

# Import YAML library if available
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

from .base import BaseAdapter
from .registry import register_adapter


class YAMLAdapter(BaseAdapter):
    """Adapter for processing YAML documents."""
    
    def __init__(self, options: Optional[Dict[str, Any]] = None):
        """
        Initialize the YAML adapter.
        
        Args:
            options: Optional configuration options
        """
        self.options = options or {}
        if not YAML_AVAILABLE:
            raise ImportError("PyYAML is required for YAML processing. Please install PyYAML.")
    
    def process(self, file_path: Path, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process a YAML file.
        
        Args:
            file_path: Path to the YAML file
            options: Optional processing options
            
        Returns:
            Dictionary with processed content and metadata
        """
        process_options = {**self.options, **(options or {})}
        
        # Verify the file exists
        if not file_path.exists():
            raise FileNotFoundError(f"YAML file not found: {file_path}")
        
        # Generate a stable document ID
        file_path_str = str(file_path)
        file_name = file_path.name
        file_name_sanitized = re.sub(r"[^A-Za-z0-9_:\-@\.\(\)\+\,=;\$!\*'%]+", "_", file_name)
        doc_id = f"yaml_{hashlib.md5(file_path_str.encode()).hexdigest()[:8]}_{file_name_sanitized}"
        
        try:
            # Read the YAML file
            with open(file_path, 'r', encoding='utf-8') as f:
                yaml_content = f.read()
                yaml_data = yaml.safe_load(yaml_content)
            
            # Also store the original content
            original_content = yaml_content
            
            # Convert to markdown for readability
            markdown_content = self.to_markdown(yaml_data)
            
            # Extract metadata
            metadata = self.extract_metadata(yaml_data)
            
            # Extract entities
            entities = self.extract_entities(yaml_data)
            
            return {
                "id": doc_id,
                "source": str(file_path),
                "content": markdown_content,
                "content_type": "markdown",
                "format": "yaml",
                "metadata": metadata,
                "entities": entities,
                "raw_data": yaml_data,  # Store the parsed data
                "original_content": original_content  # Store the original YAML text
            }
            
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in file {file_path}: {e}")
        except Exception as e:
            raise ValueError(f"Error processing YAML file {file_path}: {e}")
    
    def process_text(self, text: str, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process YAML text content.
        
        Args:
            text: YAML text content to process
            options: Optional processing options
            
        Returns:
            Dictionary with processed content and metadata
        """
        process_options = {**self.options, **(options or {})}
        
        # Generate a stable document ID
        doc_id = f"yaml_text_{hashlib.md5(text[:100].encode()).hexdigest()[:8]}"
        
        try:
            # Parse the YAML
            yaml_data = yaml.safe_load(text)
            
            # Convert to markdown for readability
            markdown_content = self.to_markdown(yaml_data)
            
            # Extract metadata
            metadata = self.extract_metadata(yaml_data)
            
            # Extract entities
            entities = self.extract_entities(yaml_data)
            
            return {
                "id": doc_id,
                "source": "text",
                "content": markdown_content,
                "content_type": "markdown",
                "format": "yaml",
                "metadata": metadata,
                "entities": entities,
                "raw_data": yaml_data,  # Store the parsed data
                "original_content": text  # Store the original YAML text
            }
            
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML: {e}")
        except Exception as e:
            raise ValueError(f"Error processing YAML text: {e}")
    
    def extract_entities(self, content: Union[str, Dict[str, Any], Any]) -> List[Dict[str, Any]]:
        """
        Extract entities from YAML content.
        
        Args:
            content: YAML content as string, dict, or parsed YAML
            
        Returns:
            List of extracted entities with metadata
        """
        entities = []
        
        # Handle different content types
        if isinstance(content, str):
            try:
                # Try to parse as YAML
                yaml_data = yaml.safe_load(content)
            except yaml.YAMLError:
                # If not valid YAML, return empty entities
                return entities
        else:
            # Assume it's already parsed YAML data
            yaml_data = content
        
        # Process based on data type
        if isinstance(yaml_data, dict):
            # Extract keys as entities
            for key, value in yaml_data.items():
                # Add the key as an entity
                entity_type = self._infer_entity_type(value)
                entities.append({
                    "type": "yaml_key",
                    "value": key,
                    "related_type": entity_type,
                    "confidence": 1.0
                })
                
                # Recursively process nested structures
                if isinstance(value, (dict, list)):
                    nested_entities = self.extract_entities(value)
                    for entity in nested_entities:
                        # Add parent key context
                        entity["parent_key"] = key
                        entities.append(entity)
                
                # Look for special types of values
                elif isinstance(value, str) and len(value) > 3:
                    detected_type = self._detect_string_entity_type(value)
                    if detected_type:
                        entities.append({
                            "type": detected_type,
                            "value": value,
                            "parent_key": key,
                            "confidence": 0.8
                        })
        
        # Process list items
        elif isinstance(yaml_data, list):
            for idx, item in enumerate(yaml_data):
                if isinstance(item, (dict, list)):
                    nested_entities = self.extract_entities(item)
                    for entity in nested_entities:
                        # Add array index context
                        entity["array_index"] = idx
                        entities.append(entity)
                elif isinstance(item, str) and len(item) > 3:
                    detected_type = self._detect_string_entity_type(item)
                    if detected_type:
                        entities.append({
                            "type": detected_type,
                            "value": item,
                            "array_index": idx,
                            "confidence": 0.7
                        })
        
        return entities
    
    def extract_metadata(self, content: Union[str, Dict[str, Any], Any]) -> Dict[str, Any]:
        """
        Extract metadata from YAML content.
        
        Args:
            content: YAML content as string, dict, or parsed YAML
            
        Returns:
            Dictionary of metadata
        """
        metadata = {
            "format": "yaml",
            "content_type": "structured"
        }
        
        # Parse content if it's a string
        if isinstance(content, str):
            try:
                yaml_data = yaml.safe_load(content)
            except yaml.YAMLError:
                # If not valid YAML, return basic metadata
                return metadata
        else:
            # Assume it's already parsed YAML data
            yaml_data = content
        
        # Extract basic structure information
        if isinstance(yaml_data, dict):
            metadata["top_level_type"] = "mapping"
            metadata["key_count"] = len(yaml_data)
            metadata["keys"] = list(yaml_data.keys())[:20]  # Limit to first 20 keys
            
            # Check for common config metadata in YAML files
            common_meta_keys = ["name", "version", "description", "author", 
                              "type", "environment", "title", "api_version"]
            for meta_key in common_meta_keys:
                if meta_key in yaml_data and isinstance(yaml_data[meta_key], (str, int, float, bool)):
                    metadata[meta_key] = yaml_data[meta_key]
            
        elif isinstance(yaml_data, list):
            metadata["top_level_type"] = "sequence"
            metadata["item_count"] = len(yaml_data)
            
            # Analyze sequence items
            if len(yaml_data) > 0:
                first_item_type = type(yaml_data[0]).__name__ if yaml_data else None
                all_same_type = all(type(item).__name__ == first_item_type for item in yaml_data)
                metadata["homogeneous_sequence"] = all_same_type
                metadata["item_type"] = first_item_type
        
        # Calculate structure complexity
        complexity = self._calculate_complexity(yaml_data)
        metadata["complexity"] = complexity
        
        return metadata
    
    def convert_to_markdown(self, content: Union[str, Dict[str, Any]]) -> str:
        """
        Convert YAML content to markdown format.
        
        Args:
            content: YAML content as string or dictionary
            
        Returns:
            Markdown representation of the YAML content
        """
        if isinstance(content, dict):
            if "content" in content:
                return content["content"]
            elif "raw_data" in content:
                return self.to_markdown(content["raw_data"])
            elif "original_content" in content:
                return f"```yaml\n{content['original_content']}\n```"
        elif isinstance(content, str):
            return f"```yaml\n{content}\n```"
                
        return self.to_markdown(content)
    
    def convert_to_text(self, content: Union[str, Dict[str, Any]]) -> str:
        """
        Convert YAML content to plain text.
        
        Args:
            content: YAML content as string or dictionary
            
        Returns:
            Plain text representation of the YAML content
        """
        if isinstance(content, dict):
            if "content" in content:
                return content["content"]
            elif "raw_data" in content:
                return yaml.dump(content["raw_data"], default_flow_style=False, sort_keys=False)
            elif "original_content" in content:
                return content["original_content"]
        elif isinstance(content, str):
            return content
                
        # Handle direct objects
        if isinstance(content, (dict, list)):
            return yaml.dump(content, default_flow_style=False, sort_keys=False)
            
        return str(content)
    
    def to_markdown(self, content: Any, indent: int = 0) -> str:
        """
        Convert YAML content to markdown format.
        
        Args:
            content: YAML content to convert
            indent: Current indentation level
            
        Returns:
            Markdown representation of the YAML content
        """
        if content is None:
            return "null"
        
        if isinstance(content, (str, int, float, bool)):
            # Format special types differently
            if isinstance(content, bool):
                return str(content).lower()
            return str(content)
        
        result = []
        spacing = "  " * indent
        
        if isinstance(content, dict):
            for key, value in content.items():
                if isinstance(value, dict):
                    result.append(f"{spacing}- **{key}**:")
                    result.append(self.to_markdown(value, indent + 1))
                elif isinstance(value, list):
                    result.append(f"{spacing}- **{key}**:")
                    result.append(self.to_markdown(value, indent + 1))
                else:
                    result.append(f"{spacing}- **{key}**: {value}")
        
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, dict):
                    # Check if the dict has a "name" or similar field for better representation
                    name_field = next((item.get(k) for k in ["name", "id", "key", "title"] if k in item), None)
                    if name_field:
                        result.append(f"{spacing}- Item `{name_field}`:")
                    else:
                        result.append(f"{spacing}- Item:")
                    result.append(self.to_markdown(item, indent + 1))
                elif isinstance(item, list):
                    result.append(f"{spacing}- Sequence:")
                    result.append(self.to_markdown(item, indent + 1))
                else:
                    result.append(f"{spacing}- {item}")
        
        return "\n".join(result)
    
    def _infer_entity_type(self, value: Any) -> str:
        """Infer the entity type from a YAML value."""
        if isinstance(value, dict):
            return "mapping"
        elif isinstance(value, list):
            return "sequence"
        elif isinstance(value, str):
            return "string"
        elif isinstance(value, bool):
            return "boolean"
        elif isinstance(value, (int, float)):
            return "number"
        elif value is None:
            return "null"
        else:
            return "unknown"
    
    def _detect_string_entity_type(self, value: str) -> Optional[str]:
        """
        Detect if a string value represents a specific entity type.
        
        Args:
            value: String value to analyze
            
        Returns:
            Detected entity type or None
        """
        # Convert to string if not already
        if not isinstance(value, str):
            value = str(value)
        
        # Check for common patterns
        value = value.strip()
        
        # Empty string
        if not value:
            return None
            
        # Email pattern
        if re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', value):
            return "email"
            
        # URL pattern
        if re.match(r'^(https?|ftp)://[^\s/$.?#].[^\s]*$', value):
            return "url"
            
        # Path pattern (Unix or Windows)
        if re.match(r'^(/[^/\s]+)+/?$', value) or re.match(r'^([A-Za-z]:\\[^\\/:*?"<>|\r\n]+\\?)+$', value):
            return "file_path"
            
        # Environment variable reference
        if re.match(r'^\${[A-Za-z0-9_]+}$', value) or re.match(r'^%[A-Za-z0-9_]+%$', value):
            return "env_var"
            
        # Version string
        if re.match(r'^v?\d+(\.\d+)+(-[a-zA-Z0-9.-]+)?$', value):
            return "version"
            
        # Date pattern
        if re.match(r'^\d{4}-\d{2}-\d{2}$', value) or re.match(r'^\d{2}/\d{2}/\d{4}$', value):
            return "date"
            
        # Connection string or password (heuristic)
        if ":" in value and "@" in value and "/" in value and len(value) > 20:
            return "connection_string"
            
        # No specific pattern detected
        return None
    
    def _calculate_complexity(self, content: Any, depth: int = 0, max_depth: int = 10) -> int:
        """
        Calculate the complexity of a YAML structure.
        
        Args:
            content: YAML content to analyze
            depth: Current depth in the structure
            max_depth: Maximum recursion depth
            
        Returns:
            Complexity score
        """
        # Prevent excessive recursion
        if depth >= max_depth:
            return 1
            
        if isinstance(content, dict):
            # Complexity increases with number of keys and depth
            return sum(self._calculate_complexity(value, depth + 1, max_depth) 
                      for value in content.values()) + len(content)
        elif isinstance(content, list):
            # Complexity increases with number of items and depth
            return sum(self._calculate_complexity(item, depth + 1, max_depth) 
                      for item in content) + 1
        else:
            # Base values have complexity 1
            return 1


# Register the adapter
register_adapter('yaml', YAMLAdapter)

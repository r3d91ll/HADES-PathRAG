"""
JSON adapter for document processing.

This module provides functionality to process JSON documents.
"""

import hashlib
import json
import re
from pathlib import Path
from typing import Dict, Any, Optional, List, Union

from .base import BaseAdapter
from .registry import register_adapter


class JSONAdapter(BaseAdapter):
    """Adapter for processing JSON documents."""
    
    def __init__(self, options: Optional[Dict[str, Any]] = None):
        """
        Initialize the JSON adapter.
        
        Args:
            options: Optional configuration options
        """
        self.options = options or {}
    
    def process(self, file_path: Path, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process a JSON file.
        
        Args:
            file_path: Path to the JSON file
            options: Optional processing options
            
        Returns:
            Dictionary with processed content and metadata
        """
        process_options = {**self.options, **(options or {})}
        
        # Verify the file exists
        if not file_path.exists():
            raise FileNotFoundError(f"JSON file not found: {file_path}")
        
        # Generate a stable document ID
        file_path_str = str(file_path)
        file_name = file_path.name
        file_name_sanitized = re.sub(r"[^A-Za-z0-9_:\-@\.\(\)\+\,=;\$!\*'%]+", "_", file_name)
        doc_id = f"json_{hashlib.md5(file_path_str.encode()).hexdigest()[:8]}_{file_name_sanitized}"
        
        try:
            # Read the JSON file
            with open(file_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            
            # Convert to markdown for readability
            markdown_content = self.to_markdown(json_data)
            
            # Extract metadata from the document
            metadata = self.extract_metadata(json_data)
            
            # Extract entities
            entities = self.extract_entities(json_data)
            
            return {
                "id": doc_id,
                "source": str(file_path),
                "content": markdown_content,
                "content_type": "markdown",
                "format": "json",
                "metadata": metadata,
                "entities": entities,
                "raw_data": json_data  # Store the original data for further processing
            }
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in file {file_path}: {e}")
        except Exception as e:
            raise ValueError(f"Error processing JSON file {file_path}: {e}")
    
    def process_text(self, text: str, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process JSON text content.
        
        Args:
            text: JSON text content to process
            options: Optional processing options
            
        Returns:
            Dictionary with processed content and metadata
        """
        process_options = {**self.options, **(options or {})}
        
        # Generate a stable document ID
        doc_id = f"json_text_{hashlib.md5(text[:100].encode()).hexdigest()[:8]}"
        
        try:
            # Parse the JSON
            json_data = json.loads(text)
            
            # Convert to markdown for readability
            markdown_content = self.to_markdown(json_data)
            
            # Extract metadata
            metadata = self.extract_metadata(json_data)
            
            # Extract entities
            entities = self.extract_entities(json_data)
            
            return {
                "id": doc_id,
                "source": "text",
                "content": markdown_content,
                "content_type": "markdown",
                "format": "json",
                "metadata": metadata,
                "entities": entities,
                "raw_data": json_data  # Store the original data for further processing
            }
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON: {e}")
        except Exception as e:
            raise ValueError(f"Error processing JSON text: {e}")
    
    def extract_entities(self, content: Union[str, Dict[str, Any], Any]) -> List[Dict[str, Any]]:
        """
        Extract entities from JSON content.
        
        Args:
            content: JSON content as string, dict, or parsed JSON
            
        Returns:
            List of extracted entities with metadata
        """
        entities = []
        
        # Handle different content types
        if isinstance(content, str):
            try:
                # Try to parse as JSON
                json_data = json.loads(content)
            except json.JSONDecodeError:
                # If not valid JSON, return empty entities
                return entities
        elif isinstance(content, dict) or isinstance(content, list):
            json_data = content
        else:
            # Unsupported content type
            return entities
        
        # Extract keys as entities (if dict)
        if isinstance(json_data, dict):
            for key, value in json_data.items():
                entity_type = self._infer_entity_type(value)
                entities.append({
                    "type": "json_key",
                    "value": key,
                    "related_type": entity_type,
                    "confidence": 1.0
                })
                
                # Recursively extract from nested objects
                if isinstance(value, dict) or isinstance(value, list):
                    nested_entities = self.extract_entities(value)
                    for entity in nested_entities:
                        # Add parent key context to nested entities
                        entity["parent_key"] = key
                        entities.append(entity)
                elif entity_type in ["string", "number"] and isinstance(value, (str, int, float)):
                    # Check for potential entity types in string values
                    if entity_type == "string" and len(str(value)) > 3:  # Only check non-trivial strings
                        detected_type = self._detect_string_entity_type(value)
                        if detected_type:
                            entities.append({
                                "type": detected_type,
                                "value": value,
                                "parent_key": key,
                                "confidence": 0.8
                            })
        
        # Extract values from arrays
        elif isinstance(json_data, list) and len(json_data) > 0:
            for idx, item in enumerate(json_data):
                if isinstance(item, dict) or isinstance(item, list):
                    nested_entities = self.extract_entities(item)
                    for entity in nested_entities:
                        # Add array index context
                        entity["array_index"] = idx
                        entities.append(entity)
                else:
                    entity_type = self._infer_entity_type(item)
                    if entity_type in ["string", "number"] and len(str(item)) > 3:
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
        Extract metadata from JSON content.
        
        Args:
            content: JSON content as string, dict, or parsed JSON
            
        Returns:
            Dictionary of metadata
        """
        metadata = {
            "format": "json",
            "content_type": "structured"
        }
        
        # Parse content if it's a string
        if isinstance(content, str):
            try:
                json_data = json.loads(content)
            except json.JSONDecodeError:
                # If not valid JSON, return basic metadata
                return metadata
        elif isinstance(content, dict) or isinstance(content, list):
            json_data = content
        else:
            # Unsupported content type
            return metadata
        
        # Extract basic structure information
        if isinstance(json_data, dict):
            metadata["top_level_type"] = "object"
            metadata["key_count"] = len(json_data)
            metadata["keys"] = list(json_data.keys())[:20]  # Limit to first 20 keys
            
            # Check for common metadata fields in the JSON
            for meta_key in ["title", "name", "description", "version", "author", "date", "type"]:
                if meta_key in json_data and isinstance(json_data[meta_key], (str, int, float, bool)):
                    metadata[meta_key] = json_data[meta_key]
        
        elif isinstance(json_data, list):
            metadata["top_level_type"] = "array"
            metadata["item_count"] = len(json_data)
            
            # Analyze array items if present
            if len(json_data) > 0:
                # Determine if array items are all the same type
                first_item_type = type(json_data[0]).__name__ if len(json_data) > 0 else None
                all_same_type = all(type(item).__name__ == first_item_type for item in json_data)
                metadata["homogeneous_array"] = all_same_type
                metadata["item_type"] = first_item_type
                
                # If items are dicts, extract common keys
                if all_same_type and first_item_type == "dict" and len(json_data) > 0:
                    common_keys = set(json_data[0].keys())
                    for item in json_data[1:]:
                        common_keys = common_keys.intersection(set(item.keys()))
                    metadata["common_keys"] = list(common_keys)
        
        # Calculate structure complexity
        complexity = self._calculate_complexity(json_data)
        metadata["complexity"] = complexity
        
        return metadata
    
    def convert_to_markdown(self, content: Union[str, Dict[str, Any]]) -> str:
        """
        Convert JSON content to markdown format.
        
        Args:
            content: JSON content as string or dictionary
            
        Returns:
            Markdown representation of the JSON content
        """
        if isinstance(content, dict):
            if "content" in content:
                return content["content"]
            elif "raw_data" in content:
                return self.to_markdown(content["raw_data"])
        elif isinstance(content, str):
            try:
                # Try to parse and format JSON
                parsed = json.loads(content)
                return f"```json\n{json.dumps(parsed, indent=2)}\n```"
            except json.JSONDecodeError:
                # If not valid JSON, return as is
                return f"```\n{content}\n```"
                
        return self.to_markdown(content)
    
    def convert_to_text(self, content: Union[str, Dict[str, Any]]) -> str:
        """
        Convert JSON content to plain text.
        
        Args:
            content: JSON content as string or dictionary
            
        Returns:
            Plain text representation of the JSON content
        """
        if isinstance(content, dict):
            if "content" in content:
                return content["content"]
            elif "raw_data" in content:
                return json.dumps(content["raw_data"], indent=2)
        elif isinstance(content, str):
            try:
                # Try to parse and format JSON
                parsed = json.loads(content)
                return json.dumps(parsed, indent=2)
            except json.JSONDecodeError:
                # If not valid JSON, return as is
                return content
                
        # Handle direct objects
        if isinstance(content, (dict, list)):
            return json.dumps(content, indent=2)
            
        return str(content)
    
    def to_markdown(self, content: Any, indent: int = 0) -> str:
        """
        Convert JSON content to markdown format.
        
        Args:
            content: JSON content to convert
            indent: Current indentation level
            
        Returns:
            Markdown representation of the JSON content
        """
        if content is None:
            return "null"
        
        if isinstance(content, (str, int, float, bool)):
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
                    result.append(f"{spacing}- " + "Object:")
                    result.append(self.to_markdown(item, indent + 1))
                elif isinstance(item, list):
                    result.append(f"{spacing}- " + "Array:")
                    result.append(self.to_markdown(item, indent + 1))
                else:
                    result.append(f"{spacing}- {item}")
        
        return "\n".join(result)
    
    def _infer_entity_type(self, value: Any) -> str:
        """Infer the entity type from a JSON value."""
        if isinstance(value, dict):
            return "object"
        elif isinstance(value, list):
            return "array"
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
            
        # Date pattern (simple)
        if re.match(r'^\d{4}-\d{2}-\d{2}$', value) or re.match(r'^\d{2}/\d{2}/\d{4}$', value):
            return "date"
            
        # Person name heuristic (simplified)
        if re.match(r'^[A-Z][a-z]+ [A-Z][a-z]+$', value):
            return "person_name"
            
        # Phone number pattern (simplified)
        if re.match(r'^\+?[\d\s\(\)-]{7,20}$', value) and any(c.isdigit() for c in value):
            return "phone_number"
            
        # ID/UUID pattern
        if re.match(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', value, re.I):
            return "uuid"
            
        # No specific pattern detected
        return None
    
    def _calculate_complexity(self, content: Any, depth: int = 0) -> int:
        """
        Calculate the complexity of a JSON structure.
        
        Args:
            content: JSON content to analyze
            depth: Current depth in the structure
            
        Returns:
            Complexity score
        """
        if isinstance(content, dict):
            # Complexity increases with number of keys and depth
            return sum(self._calculate_complexity(value, depth + 1) for value in content.values()) + len(content)
        elif isinstance(content, list):
            # Complexity increases with number of items and depth
            return sum(self._calculate_complexity(item, depth + 1) for item in content) + 1
        else:
            # Base values have complexity 1
            return 1


# Register the adapter
register_adapter('json', JSONAdapter)

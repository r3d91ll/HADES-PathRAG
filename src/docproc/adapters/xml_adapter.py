"""
XML adapter for document processing.

This module provides functionality to process XML documents.
"""

import hashlib
import re
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
import xml.etree.ElementTree as ET
from xml.dom import minidom

from .base import BaseAdapter
from .registry import register_adapter


class XMLAdapter(BaseAdapter):
    """Adapter for processing XML documents."""
    
    def __init__(self, options: Optional[Dict[str, Any]] = None):
        """
        Initialize the XML adapter.
        
        Args:
            options: Optional configuration options
        """
        self.options = options or {}
    
    def process(self, file_path: Path, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process an XML file.
        
        Args:
            file_path: Path to the XML file
            options: Optional processing options
            
        Returns:
            Dictionary with processed content and metadata
        """
        process_options = {**self.options, **(options or {})}
        
        # Verify the file exists
        if not file_path.exists():
            raise FileNotFoundError(f"XML file not found: {file_path}")
        
        # Generate a stable document ID
        file_path_str = str(file_path)
        file_name = file_path.name
        file_name_sanitized = re.sub(r"[^A-Za-z0-9_:\-@\.\(\)\+\,=;\$!\*'%]+", "_", file_name)
        doc_id = f"xml_{hashlib.md5(file_path_str.encode()).hexdigest()[:8]}_{file_name_sanitized}"
        
        try:
            # Read the XML file as text first to preserve original formatting
            with open(file_path, 'r', encoding='utf-8') as f:
                xml_text = f.read()
            
            # Ensure there are no leading whitespaces before the XML declaration
            xml_text = xml_text.strip()
            
            # Parse the XML
            try:
                tree = ET.fromstring(xml_text)
            except ET.ParseError:
                # Try parsing with ElementTree.parse which can be more forgiving
                import io
                tree = ET.parse(io.StringIO(xml_text)).getroot()
            
            # Convert to markdown for readability
            markdown_content = self.to_markdown(tree)
            
            # Extract metadata
            metadata = self.extract_metadata(tree)
            
            # Extract entities
            entities = self.extract_entities(tree)
            
            return {
                "id": doc_id,
                "source": str(file_path),
                "content": markdown_content,
                "content_type": "markdown",
                "format": "xml",
                "metadata": metadata,
                "entities": entities,
                "original_content": xml_text  # Store the original XML
            }
            
        except ET.ParseError as e:
            raise ValueError(f"Invalid XML in file {file_path}: {e}")
        except Exception as e:
            raise ValueError(f"Error processing XML file {file_path}: {e}")
    
    def process_text(self, text: str, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process XML text content.
        
        Args:
            text: XML text content to process
            options: Optional processing options
            
        Returns:
            Dictionary with processed content and metadata
        """
        process_options = {**self.options, **(options or {})}
        
        # Generate a stable document ID
        doc_id = f"xml_text_{hashlib.md5(text[:100].encode()).hexdigest()[:8]}"
        
        try:
            # Parse the XML
            tree = ET.fromstring(text)
            
            # Convert to markdown for readability
            markdown_content = self.to_markdown(tree)
            
            # Extract metadata
            metadata = self.extract_metadata(tree)
            
            # Extract entities
            entities = self.extract_entities(tree)
            
            return {
                "id": doc_id,
                "source": "text",
                "content": markdown_content,
                "content_type": "markdown",
                "format": "xml",
                "metadata": metadata,
                "entities": entities,
                "original_content": text  # Store the original XML
            }
            
        except ET.ParseError as e:
            raise ValueError(f"Invalid XML: {e}")
        except Exception as e:
            raise ValueError(f"Error processing XML text: {e}")
    
    def extract_entities(self, content: Union[str, ET.Element, Any]) -> List[Dict[str, Any]]:
        """
        Extract entities from XML content.
        
        Args:
            content: XML content as string or Element
            
        Returns:
            List of extracted entities with metadata
        """
        entities = []
        
        # Parse content if needed
        root = None
        if isinstance(content, str):
            try:
                root = ET.fromstring(content)
            except ET.ParseError:
                return entities
        elif isinstance(content, ET.Element):
            root = content
        else:
            # Unsupported content type
            return entities
        
        # Process each element
        for elem_path, elem in self._iter_elements(root):
            # Create an entity for each element
            entity = {
                "type": "xml_element",
                "value": elem.tag,
                "xpath": elem_path,
                "confidence": 1.0
            }
            
            # Add attributes if present
            if elem.attrib:
                entity["attributes"] = dict(elem.attrib)
            
            entities.append(entity)
            
            # Check for special content or attributes
            if elem.text and elem.text.strip():
                text_content = elem.text.strip()
                # Only check significant text content
                if len(text_content) > 3:
                    entity_type = self._detect_string_entity_type(text_content)
                    if entity_type:
                        entities.append({
                            "type": entity_type,
                            "value": text_content,
                            "parent_element": elem.tag,
                            "xpath": elem_path,
                            "confidence": 0.8
                        })
            
            # Check attributes for special values
            for attr_name, attr_value in elem.attrib.items():
                if len(attr_value) > 3:  # Only check non-trivial values
                    entity_type = self._detect_string_entity_type(attr_value)
                    if entity_type:
                        entities.append({
                            "type": entity_type,
                            "value": attr_value,
                            "parent_element": elem.tag,
                            "attribute": attr_name,
                            "xpath": f"{elem_path}/@{attr_name}",
                            "confidence": 0.8
                        })
        
        return entities
    
    def extract_metadata(self, content: Union[str, ET.Element, Any]) -> Dict[str, Any]:
        """
        Extract metadata from XML content.
        
        Args:
            content: XML content as string or Element
            
        Returns:
            Dictionary of metadata
        """
        metadata = {
            "format": "xml",
            "content_type": "structured"
        }
        
        # Parse content if needed
        root = None
        if isinstance(content, str):
            try:
                root = ET.fromstring(content)
            except ET.ParseError:
                return metadata
        elif isinstance(content, ET.Element):
            root = content
        else:
            # Unsupported content type
            return metadata
        
        # Get the root element tag
        metadata["root_element"] = root.tag
        
        # Calculate basic structure statistics
        element_count = 0
        depth = 0
        max_depth = 0
        tag_counts = {}
        namespace_uris = set()
        
        # Extract namespaces if any
        namespaces = self._extract_namespaces(root)
        if namespaces:
            metadata["namespaces"] = namespaces
        
        # Analyze structure
        for elem_path, elem in self._iter_elements(root):
            element_count += 1
            
            # Track tag frequency
            if elem.tag in tag_counts:
                tag_counts[elem.tag] += 1
            else:
                tag_counts[elem.tag] = 1
            
            # Track depth
            current_depth = elem_path.count("/") + 1
            max_depth = max(max_depth, current_depth)
            
            # Check for namespaces
            if "}" in elem.tag:
                ns_uri = elem.tag.split("}")[0].strip("{")
                namespace_uris.add(ns_uri)
        
        metadata["element_count"] = element_count
        metadata["max_depth"] = max_depth
        metadata["common_elements"] = [tag for tag, count in sorted(
            tag_counts.items(), key=lambda x: x[1], reverse=True)[:10]]
        
        if namespace_uris:
            metadata["namespace_uris"] = list(namespace_uris)
        
        # Check for common metadata elements
        common_meta_elements = ["title", "description", "author", "version", "date", "name"]
        for meta_elem in common_meta_elements:
            matches = root.findall(f".//{meta_elem}")
            if matches and matches[0].text:
                metadata[meta_elem] = matches[0].text.strip()
        
        # Look for document-type specific metadata (e.g., RSS, SOAP, etc.)
        if root.tag.endswith("rss"):
            metadata["document_type"] = "rss"
            channel = root.find(".//channel")
            if channel is not None:
                for child in channel:
                    if child.tag in ["title", "description", "link"] and child.text:
                        metadata[f"rss_{child.tag}"] = child.text.strip()
        
        elif root.tag.endswith("html"):
            metadata["document_type"] = "html"
            head = root.find(".//head")
            if head is not None:
                title = head.find(".//title")
                if title is not None and title.text:
                    metadata["html_title"] = title.text.strip()
        
        elif "Envelope" in root.tag:
            metadata["document_type"] = "soap"
        
        return metadata
    
    def convert_to_markdown(self, content: Union[str, Dict[str, Any]]) -> str:
        """
        Convert XML content to markdown format.
        
        Args:
            content: XML content as string or dictionary
            
        Returns:
            Markdown representation of the XML content
        """
        # Handle different content types
        if isinstance(content, dict):
            if "content" in content:
                return content["content"]
            elif "original_content" in content:
                return f"```xml\n{content['original_content']}\n```"
        elif isinstance(content, str):
            return f"```xml\n{content}\n```"
        elif isinstance(content, ET.Element):
            return self.to_markdown(content, 0)
            
        return ""
    
    def convert_to_text(self, content: Union[str, Dict[str, Any]]) -> str:
        """
        Convert XML content to plain text.
        
        Args:
            content: XML content as string or dictionary
            
        Returns:
            Plain text representation of the XML content
        """
        if isinstance(content, dict):
            if "original_content" in content:
                root = ET.fromstring(content["original_content"])
                return self._get_text_from_element(root)
            elif "content" in content:
                return content["content"]
        elif isinstance(content, str):
            try:
                root = ET.fromstring(content)
                return self._get_text_from_element(root)
            except ET.ParseError:
                return content
                
        return ""
    
    def _get_text_from_element(self, element: ET.Element) -> str:
        """
        Extract plain text from an XML element and its children.
        
        Args:
            element: XML element
            
        Returns:
            Plain text content
        """
        text_parts = []
        if element.text and element.text.strip():
            text_parts.append(element.text.strip())
            
        for child in element:
            text_parts.append(self._get_text_from_element(child))
            if child.tail and child.tail.strip():
                text_parts.append(child.tail.strip())
                
        return " ".join(text_parts)
    
    def to_markdown(self, content: Union[ET.Element, str], indent: int = 0) -> str:
        """
        Convert XML content to markdown format.
        
        Args:
            content: XML content to convert (Element or string)
            indent: Current indentation level
            
        Returns:
            Markdown representation of the XML content
        """
        # Parse string content if needed
        root = None
        if isinstance(content, str):
            try:
                root = ET.fromstring(content)
            except ET.ParseError:
                return f"```xml\n{content}\n```"
        else:
            root = content
        
        result = []
        
        # Process the XML tree recursively
        def process_element(element, current_indent=0, path=""):
            spacing = "  " * current_indent
            current_path = f"{path}/{element.tag}" if path else element.tag
            
            # Build the element representation
            element_md = f"{spacing}- **{element.tag}**"
            
            # Add attributes if present
            if element.attrib:
                attrs = ", ".join([f"`{k}=\"{v}\"`" for k, v in element.attrib.items()])
                element_md += f" ({attrs})"
            
            # Add text content if present and not just whitespace
            if element.text and element.text.strip():
                # Truncate very long text
                text = element.text.strip()
                if len(text) > 100:
                    text = text[:97] + "..."
                element_md += f": {text}"
            
            result.append(element_md)
            
            # Process children
            for child in element:
                process_element(child, current_indent + 1, current_path)
                
                # Handle tail text (text after the element)
                if child.tail and child.tail.strip():
                    result.append(f"{spacing}  - TEXT: {child.tail.strip()}")
        
        # Start processing from the root
        process_element(root)
        
        return "\n".join(result)
    
    def _iter_elements(self, root: ET.Element) -> List[tuple]:
        """
        Iterate through all elements in the XML tree and yield (xpath, element) tuples.
        
        Args:
            root: Root element of the XML tree
            
        Yields:
            Tuple of (xpath, element)
        """
        result = []
        
        def _traverse(element, path=""):
            current_path = f"{path}/{element.tag}" if path else element.tag
            result.append((current_path, element))
            
            for child in element:
                _traverse(child, current_path)
        
        _traverse(root)
        return result
    
    def _extract_namespaces(self, root: ET.Element) -> Dict[str, str]:
        """
        Extract namespace information from an XML root element.
        
        Args:
            root: XML root element
            
        Returns:
            Dictionary mapping namespace prefixes to URIs
        """
        namespaces = {}
        
        # This is a bit of a hack since ElementTree doesn't preserve namespace info well
        # We'll try to extract from tag names
        for _, elem in self._iter_elements(root):
            if "}" in elem.tag:
                uri = elem.tag.split("}")[0].strip("{")
                # Use a synthetic prefix if we can't determine the real one
                prefix = f"ns{len(namespaces) + 1}"
                namespaces[prefix] = uri
                
            # Check attributes too
            for name in elem.attrib:
                if "}" in name:
                    uri = name.split("}")[0].strip("{")
                    prefix = f"ns{len(namespaces) + 1}"
                    namespaces[prefix] = uri
        
        return namespaces
    
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
            
        # Date pattern
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


# Register the adapter
register_adapter('xml', XMLAdapter)

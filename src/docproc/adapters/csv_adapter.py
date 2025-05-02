"""
CSV adapter for document processing.

This module provides functionality to process CSV documents.
"""

import csv
import hashlib
import re
from pathlib import Path
from typing import Dict, Any, Optional, List, Union

from .base import BaseAdapter
from .registry import register_adapter


class CSVAdapter(BaseAdapter):
    """Adapter for processing CSV documents."""
    
    def __init__(self, options: Optional[Dict[str, Any]] = None):
        """
        Initialize the CSV adapter.
        
        Args:
            options: Optional configuration options
        """
        self.options = options or {}
        self.sample_size = 5  # Number of lines to sample for delimiter detection
    
    def process(self, file_path: Path, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process a CSV file.
        
        Args:
            file_path: Path to the CSV file
            options: Optional processing options
            
        Returns:
            Dictionary with processed content and metadata
        """
        process_options = {**self.options, **(options or {})}
        
        # Verify the file exists
        if not file_path.exists():
            raise FileNotFoundError(f"CSV file not found: {file_path}")
        
        # Generate a stable document ID
        file_path_str = str(file_path)
        file_name = file_path.name
        file_name_sanitized = re.sub(r"[^A-Za-z0-9_:\-@\.\(\)\+\,=;\$!\*'%]+", "_", file_name)
        doc_id = f"csv_{hashlib.md5(file_path_str.encode()).hexdigest()[:8]}_{file_name_sanitized}"
        
        try:
            # Read the CSV file
            with open(file_path, 'r', encoding='utf-8', newline='') as f:
                csv_content = f.read().strip()  # Read and strip any leading/trailing whitespace
                
            # Try to detect the delimiter from the content
            try:
                delimiter = self._detect_delimiter_from_text(csv_content)
            except Exception:
                delimiter = ','  # Default to comma
            
            # Parse the CSV with the detected delimiter
            try:
                reader = csv.reader(csv_content.splitlines(), delimiter=delimiter)
                data = list(reader)
            except Exception:
                # Fallback to a simple line-by-line reading
                data = [line.strip().split(delimiter) for line in csv_content.splitlines() if line.strip()]
            
            # Ensure we have at least some data
            if not data:
                data = [[]]  # Empty data to avoid index errors
            
            # Get header row
            header = data[0] if data else []
            
            # Convert rows to dictionaries if we have headers
            records = []
            if header and len(data) > 1:
                for row in data[1:]:
                    # Ensure row has same length as header by padding or truncating
                    padded_row = row + [''] * (len(header) - len(row)) if len(row) < len(header) else row[:len(header)]
                    records.append(dict(zip(header, padded_row)))
            
            # Create structured representation
            structured_data = {
                "header": header,
                "rows": data[1:] if len(data) > 1 else [],
                "records": records
            }
            
            # Convert to markdown for readability
            markdown_content = self.convert_to_markdown({"structured_data": structured_data})
            
            # Extract metadata
            metadata = self.extract_metadata(structured_data)
            
            # Extract entities
            entities = self.extract_entities(structured_data)
            
            return {
                "id": doc_id,
                "source": str(file_path),
                "content": markdown_content,
                "content_type": "markdown",
                "format": "csv",
                "metadata": metadata,
                "entities": entities,
                "structured_data": structured_data  # Store the parsed data
            }
            
        except Exception as e:
            raise ValueError(f"Error processing CSV file {file_path}: {e}")
    
    def process_text(self, text: str, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process CSV content directly.
        
        Args:
            text: CSV content to process
            options: Optional processing options
            
        Returns:
            Dictionary with processed content and metadata
        """
        process_options = {**self.options, **(options or {})}
        
        # Generate a document ID from the content
        doc_id = f"csv_{hashlib.md5(text.encode()).hexdigest()[:12]}"
        
        try:
            # Try to detect delimiter
            delimiter = self._detect_delimiter_from_text(text) or ','
            
            # Parse the CSV
            reader = csv.reader(text.splitlines(), delimiter=delimiter)
            data = list(reader)
            
            # Get header row
            header = data[0] if len(data) > 0 else []
            
            # Convert rows to dictionaries if we have headers
            records = []
            if header and len(data) > 1:
                for row in data[1:]:
                    # Ensure row has same length as header
                    padded_row = row + [''] * (len(header) - len(row)) if len(row) < len(header) else row[:len(header)]
                    records.append(dict(zip(header, padded_row)))
            
            # Create structured representation
            structured_data = {
                "header": header,
                "rows": data[1:] if len(data) > 1 else [],
                "records": records
            }
            
            # Convert to markdown for readability
            markdown_content = self.convert_to_markdown({"structured_data": structured_data})
            
            # Extract metadata
            metadata = self.extract_metadata(structured_data)
            
            # Extract entities
            entities = self.extract_entities(structured_data)
            
            return {
                "id": doc_id,
                "source": "text",
                "content": markdown_content,
                "content_type": "markdown",
                "format": "csv",
                "metadata": metadata,
                "entities": entities,
                "structured_data": structured_data  # Store the parsed data
            }
            
        except Exception as e:
            raise ValueError(f"Error processing CSV text: {e}")
    
    def extract_entities(self, content: Union[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract entities from CSV content.
        
        Args:
            content: Document content as CSV string or structured data
            
        Returns:
            List of extracted entities
        """
        entities = []
        
        # Handle different content types
        if isinstance(content, dict):
            # Extract from structured data
            if "structured_data" in content:
                structured_data = content["structured_data"]
            elif "header" in content and "rows" in content:
                structured_data = content
            else:
                return entities  # Empty list for unsupported content
                
            header = structured_data.get("header", [])
            rows = structured_data.get("rows", [])
            
            # Extract entities from column-based patterns
            if header and rows:
                # Look for entity types in each column
                for col_idx, col_name in enumerate(header):
                    # Detect column type
                    col_type = self._detect_column_type(rows, col_idx)
                    
                    # Extract potential entities based on column type
                    if col_type in ["email", "url", "phone", "date"]:
                        for row_idx, row in enumerate(rows):
                            if col_idx < len(row) and row[col_idx].strip():
                                entity_type = self._detect_string_entity_type(row[col_idx])
                                if entity_type:
                                    entities.append({
                                        "type": entity_type,
                                        "value": row[col_idx],
                                        "confidence": 0.8,
                                        "column": col_name,
                                        "row": row_idx + 1  # +1 to account for 0-indexing vs. 1-indexing
                                    })
        
        elif isinstance(content, str):
            # Try to parse and extract from raw CSV
            try:
                reader = csv.reader(content.splitlines())
                data = list(reader)
                
                if data:
                    header = data[0]
                    rows = data[1:]
                    
                    # Call this method again with structured data
                    return self.extract_entities({
                        "header": header,
                        "rows": rows
                    })
            except:
                pass  # If parsing fails, return empty list
                
        return entities
    
    def extract_metadata(self, content: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Extract metadata from CSV content.
        
        Args:
            content: Document content as CSV string or structured data
            
        Returns:
            Dictionary of metadata
        """
        metadata = {}
        
        # Handle different content types
        if isinstance(content, dict):
            # Extract from structured data
            if "structured_data" in content:
                structured_data = content["structured_data"]
            elif "header" in content and "rows" in content:
                structured_data = content
            else:
                return metadata  # Empty dict for unsupported content
                
            header = structured_data.get("header", [])
            rows = structured_data.get("rows", [])
            
            # Extract basic metadata
            metadata["column_count"] = len(header)
            metadata["row_count"] = len(rows)
            metadata["column_names"] = header
            
            # Get column types
            if rows:
                column_types = {}
                for col_idx, col_name in enumerate(header):
                    if col_idx < len(header):
                        column_types[col_name] = self._detect_column_type(rows, col_idx)
                metadata["column_types"] = column_types
                
        elif isinstance(content, str):
            # Try to parse and extract from raw CSV
            try:
                reader = csv.reader(content.splitlines())
                data = list(reader)
                
                if data:
                    header = data[0]
                    rows = data[1:]
                    
                    # Call this method again with structured data
                    return self.extract_metadata({
                        "header": header,
                        "rows": rows
                    })
            except:
                pass  # If parsing fails, return empty dict
                
        return metadata
    
    def convert_to_markdown(self, content: Union[str, Dict[str, Any]]) -> str:
        """
        Convert CSV content to markdown format.
        
        Args:
            content: CSV content as string or dictionary
            
        Returns:
            Markdown representation of the CSV content
        """
        if isinstance(content, dict):
            if "content" in content:
                return content["content"]
            elif "structured_data" in content:
                structured_data = content["structured_data"]
                header = structured_data.get("header", [])
                rows = structured_data.get("rows", [])
            else:
                return "```\nUnsupported CSV format\n```"
        elif isinstance(content, str):
            try:
                # Try to parse as CSV
                lines = content.splitlines()
                reader = csv.reader(lines)
                data = list(reader)
                
                header = data[0] if len(data) > 0 else []
                rows = data[1:] if len(data) > 1 else []
            except:
                # If parsing fails, return the original text
                return f"```\n{content}\n```"
        else:
            return "```\nUnsupported CSV format\n```"
        
        # Generate markdown table
        result = []
        
        # Add header
        if header:
            result.append("| " + " | ".join(header) + " |")
            result.append("| " + " | ".join(["---"] * len(header)) + " |")
            
            # Add rows (limit to avoid excessively long tables)
            max_rows = 50
            for i, row in enumerate(rows):
                if i >= max_rows:
                    result.append(f"\n*... {len(rows) - max_rows} more rows omitted ...*")
                    break
                    
                # Ensure row has same length as header
                padded_row = row + [''] * (len(header) - len(row)) if len(row) < len(header) else row[:len(header)]
                # Escape pipe characters in cell values
                escaped_row = [str(cell).replace('|', '\\|') for cell in padded_row]
                result.append("| " + " | ".join(escaped_row) + " |")
        else:
            # No header, just show raw data
            result.append("```")
            for row in rows[:20]:  # Limit to 20 rows
                result.append(",".join(row))
            if len(rows) > 20:
                result.append(f"... {len(rows) - 20} more rows ...")
            result.append("```")
        
        return "\n".join(result)
    
    def convert_to_text(self, content: Union[str, Dict[str, Any]]) -> str:
        """
        Convert CSV content to plain text.
        
        Args:
            content: CSV content as string or dictionary
            
        Returns:
            Plain text representation of the CSV content
        """
        if isinstance(content, dict):
            if "content" in content:
                return content["content"]
            elif "structured_data" in content:
                # Convert structured data back to CSV
                output = []
                data = content["structured_data"]
                
                if "header" in data and data["header"]:
                    output.append(",".join(data["header"]))
                
                if "rows" in data:
                    for row in data["rows"]:
                        output.append(",".join(str(cell) for cell in row))
                
                return "\n".join(output)
        elif isinstance(content, str):
            return content
                
        return str(content)
    
    def _detect_delimiter(self, file_handle) -> str:
        """
        Detect the delimiter used in a CSV file.
        
        Args:
            file_handle: Open file handle to CSV file
            
        Returns:
            Detected delimiter
        """
        # Read first few lines to sample
        sample_lines = []
        for _ in range(self.sample_size):
            line = file_handle.readline()
            if not line:
                break
            # Strip whitespace to handle potential leading spaces
            sample_lines.append(line.strip())
            
        if not sample_lines:
            return ','  # Default to comma if no content
        
        # Check common delimiters
        delimiters = [',', '\t', ';', '|']
        counts = {}
        
        for delimiter in delimiters:
            for line in sample_lines:
                if delimiter in line:
                    counts[delimiter] = counts.get(delimiter, 0) + line.count(delimiter)
                    
        if not counts:
            return ','  # Default to comma if no delimiters found
            
        # Return the delimiter that appears most frequently
        return max(counts.items(), key=lambda x: x[1])[0]
    
    def _detect_delimiter_from_text(self, text: str) -> str:
        """
        Detect the delimiter used in CSV text.
        
        Args:
            text: CSV text content
            
        Returns:
            Detected delimiter character
        """
        # Common delimiters to check
        delimiters = [',', '\t', ';', '|', ':']
        
        # Check first few lines (up to 5)
        lines = text.splitlines()[:5]
        lines = [line.strip() for line in lines if line.strip()]
        
        if not lines:
            return ','  # Default to comma if no content
        
        counts = {d: 0 for d in delimiters}
        
        for line in lines:
            for d in delimiters:
                if d in line:
                    counts[d] += line.count(d)
        
        # Filter out delimiters with zero counts
        counts = {d: c for d, c in counts.items() if c > 0}
        
        if not counts:
            # Default to comma if no delimiters found
            return ','
            
        # Return the delimiter with the highest count
        return max(counts, key=counts.get)
    
    def _detect_column_type(self, rows: List[List[str]], col_idx: int) -> str:
        """
        Detect the data type of a CSV column.
        
        Args:
            rows: List of data rows
            col_idx: Column index to analyze
            
        Returns:
            Detected data type
        """
        # Collect non-empty values from the column
        values = []
        for row in rows:
            if col_idx < len(row) and row[col_idx].strip():
                values.append(row[col_idx].strip())
        
        if not values:
            return "empty"
            
        # Check if all values are numeric
        numeric_count = sum(1 for v in values if re.match(r'^-?\d+(\.\d+)?$', v))
        
        # Check if all values are dates (simple check)
        date_count = sum(1 for v in values if re.match(r'^\d{1,4}[-/]\d{1,2}[-/]\d{1,4}$', v) or 
                                               re.match(r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}', v))
        
        # Check if all values are boolean-like
        bool_count = sum(1 for v in values if v.lower() in ['true', 'false', 'yes', 'no', '0', '1', 'y', 'n'])
        
        # Determine type based on most common pattern
        total = len(values)
        
        if total == 0:
            return "empty"
        
        if numeric_count / total > 0.8:
            return "numeric"
        elif date_count / total > 0.8:
            return "date"
        elif bool_count / total > 0.8:
            return "boolean"
        else:
            return "text"
    
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
            
        # Date pattern (various formats)
        if re.match(r'^\d{1,4}[-/]\d{1,2}[-/]\d{1,4}$', value) or re.match(r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}', value):
            return "date"
            
        # Phone number patterns (various formats)
        if re.match(r'^\+?[\d\-\(\)\s]{7,}$', value) and re.search(r'\d{3}', value):
            return "phone"
            
        return None


# Register the adapter
register_adapter('csv', CSVAdapter)

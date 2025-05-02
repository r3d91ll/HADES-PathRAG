"""
Unit tests for document processing adapters.

This module tests the functionality of various document adapters for processing different file formats.
"""

import os
import tempfile
from pathlib import Path
import unittest
from typing import Dict, Any

from src.docproc.adapters import (
    get_adapter_for_format,
    get_supported_formats,
    PDFAdapter,
    HTMLAdapter,
    CodeAdapter,
    JSONAdapter,
    YAMLAdapter,
    XMLAdapter,
    CSVAdapter
)
from src.docproc.core import process_document, process_text, detect_format


class TestAdapters(unittest.TestCase):
    """Test suite for document processing adapters."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.base_path = Path(self.temp_dir.name)
        
        # Create sample files for testing
        self._create_sample_files()
    
    def tearDown(self):
        """Clean up after tests."""
        self.temp_dir.cleanup()
    
    def _create_sample_files(self):
        """Create sample files for testing different adapters."""
        # Python sample
        python_content = """
# Sample Python file
def hello_world():
    \"\"\"Say hello to the world.\"\"\"
    print("Hello, world!")
    
class Example:
    \"\"\"Example class for testing.\"\"\"
    def __init__(self, name):
        self.name = name
        
    def greet(self):
        \"\"\"Greet the user.\"\"\"
        return f"Hello, {self.name}!"
"""
        self._write_file("sample.py", python_content)
        
        # HTML sample
        html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>Sample HTML</title>
    <meta name="description" content="A sample HTML file for testing">
</head>
<body>
    <h1>Sample Document</h1>
    <p>This is a <strong>sample</strong> HTML document for testing the HTMLAdapter.</p>
    <ul>
        <li>Item 1</li>
        <li>Item 2</li>
        <li>Item 3</li>
    </ul>
    <a href="https://example.com">Example Link</a>
</body>
</html>
"""
        self._write_file("sample.html", html_content)
        
        # JSON sample
        json_content = """
{
    "name": "Sample Document",
    "type": "JSON",
    "version": "1.0",
    "items": [
        {
            "id": 1,
            "name": "Item 1",
            "value": 42
        },
        {
            "id": 2,
            "name": "Item 2",
            "value": 73
        }
    ],
    "metadata": {
        "author": "Test User",
        "email": "test@example.com",
        "created": "2023-01-01"
    }
}
"""
        self._write_file("sample.json", json_content)
        
        # YAML sample
        yaml_content = """
# Sample YAML config
name: Sample Document
type: YAML
version: 1.0
settings:
  debug: true
  log_level: info
  max_connections: 100

users:
  - username: user1
    email: user1@example.com
    roles:
      - admin
      - editor
  - username: user2
    email: user2@example.com
    roles:
      - viewer
"""
        self._write_file("sample.yaml", yaml_content)
        
        # XML sample
        xml_content = """
<?xml version="1.0" encoding="UTF-8"?>
<document type="sample">
    <metadata>
        <title>Sample XML Document</title>
        <author>Test User</author>
        <date>2023-01-01</date>
    </metadata>
    <content>
        <section id="intro">
            <heading>Introduction</heading>
            <paragraph>This is a sample XML document for testing.</paragraph>
        </section>
        <section id="details">
            <heading>Details</heading>
            <paragraph>It contains multiple sections and elements.</paragraph>
            <list>
                <item>First item</item>
                <item>Second item</item>
                <item>Third item</item>
            </list>
        </section>
    </content>
</document>
"""
        self._write_file("sample.xml", xml_content)
        
        # CSV sample
        csv_content = """
Name,Email,Age,JoinDate
John Doe,john@example.com,32,2020-01-15
Jane Smith,jane@example.com,28,2021-03-22
Bob Johnson,bob@example.com,45,2019-11-07
Alice Brown,alice@example.com,39,2022-05-30
"""
        self._write_file("sample.csv", csv_content)
    
    def _write_file(self, filename, content):
        """Helper to write a test file."""
        filepath = self.base_path / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
    
    def test_get_supported_formats(self):
        """Test retrieving all supported formats."""
        formats = get_supported_formats()
        
        # Ensure we have all the expected formats
        expected_formats = {'pdf', 'html', 'python', 'json', 'yaml', 'xml', 'csv'}
        for fmt in expected_formats:
            self.assertIn(fmt, formats, f"Expected format '{fmt}' not found in supported formats")
    
    def test_code_adapter(self):
        """Test the CodeAdapter with a Python file."""
        python_file = self.base_path / "sample.py"
        
        # Process using the adapter directly
        adapter = CodeAdapter()
        result = adapter.process(python_file)
        
        # Verify common fields
        self._verify_common_result_fields(result)
        
        # Verify code-specific fields
        self.assertEqual(result["format"], "python")
        self.assertIn("symbols", result)
        self.assertGreater(len(result["symbols"]), 0)
        
        # Test functions and classes extraction
        symbols = result["symbols"]
        function_names = [s["name"] for s in symbols if s["type"] == "function"]
        class_names = [s["name"] for s in symbols if s["type"] == "class"]
        
        self.assertIn("hello_world", function_names)
        self.assertIn("Example", class_names)
    
    def test_html_adapter(self):
        """Test the HTMLAdapter with an HTML file."""
        html_file = self.base_path / "sample.html"
        
        # Process using the adapter directly
        adapter = HTMLAdapter()
        result = adapter.process(html_file)
        
        # Verify common fields
        self._verify_common_result_fields(result)
        
        # Verify HTML-specific fields
        self.assertEqual(result["format"], "html")
        self.assertIn("metadata", result)
        
        # Test metadata extraction
        metadata = result["metadata"]
        self.assertIn("title", metadata)
        self.assertEqual(metadata["title"], "Sample HTML")
    
    def test_json_adapter(self):
        """Test the JSONAdapter with a JSON file."""
        json_file = self.base_path / "sample.json"
        
        # Process using the adapter directly
        adapter = JSONAdapter()
        result = adapter.process(json_file)
        
        # Verify common fields
        self._verify_common_result_fields(result)
        
        # Verify JSON-specific fields
        self.assertEqual(result["format"], "json")
        self.assertIn("raw_data", result)
        
        # Test content extraction
        raw_data = result["raw_data"]
        self.assertEqual(raw_data["name"], "Sample Document")
        self.assertEqual(raw_data["type"], "JSON")
        self.assertEqual(len(raw_data["items"]), 2)
    
    def test_yaml_adapter(self):
        """Test the YAMLAdapter with a YAML file."""
        yaml_file = self.base_path / "sample.yaml"
        
        # Process using the adapter directly
        adapter = YAMLAdapter()
        result = adapter.process(yaml_file)
        
        # Verify common fields
        self._verify_common_result_fields(result)
        
        # Verify YAML-specific fields
        self.assertEqual(result["format"], "yaml")
        self.assertIn("raw_data", result)
        
        # Test content extraction
        raw_data = result["raw_data"]
        self.assertEqual(raw_data["name"], "Sample Document")
        self.assertEqual(raw_data["type"], "YAML")
        self.assertEqual(len(raw_data["users"]), 2)
    
    def test_xml_adapter(self):
        """Test the XMLAdapter with an XML file."""
        xml_file = self.base_path / "sample.xml"
        
        # Process using the adapter directly
        adapter = XMLAdapter()
        result = adapter.process(xml_file)
        
        # Verify common fields
        self._verify_common_result_fields(result)
        
        # Verify XML-specific fields
        self.assertEqual(result["format"], "xml")
        self.assertIn("original_content", result)
        
        # Test metadata extraction
        metadata = result["metadata"]
        self.assertIn("root_element", metadata)
        self.assertIn("element_count", metadata)
    
    def test_csv_adapter(self):
        """Test the CSVAdapter with a CSV file."""
        csv_file = self.base_path / "sample.csv"
        
        # Process using the adapter directly
        adapter = CSVAdapter()
        result = adapter.process(csv_file)
        
        # Verify common fields
        self._verify_common_result_fields(result)
        
        # Verify CSV-specific fields
        self.assertEqual(result["format"], "csv")
        self.assertIn("structured_data", result)
        
        # Test content extraction
        structured_data = result["structured_data"]
        self.assertEqual(structured_data["header"][0], "Name")
        self.assertEqual(len(structured_data["rows"]), 4)  # 4 data rows
    
    def test_core_process_document(self):
        """Test the core process_document function with different file types."""
        for filename in ["sample.py", "sample.html", "sample.json", "sample.yaml", "sample.xml", "sample.csv"]:
            file_path = self.base_path / filename
            
            # Detect format
            detected_format = detect_format(file_path)
            self.assertIsNotNone(detected_format, f"Failed to detect format for {filename}")
            
            # Process document
            result = process_document(file_path)
            
            # Verify basic result structure
            self._verify_common_result_fields(result)
            self.assertEqual(result["format"], detected_format)
    
    def test_core_process_text(self):
        """Test the core process_text function with different content types."""
        # Test with HTML content
        html_file = self.base_path / "sample.html"
        with open(html_file, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        html_result = process_text(html_content, format="html")
        self.assertEqual(html_result["format"], "html")
        
        # Test with JSON content
        json_file = self.base_path / "sample.json"
        with open(json_file, 'r', encoding='utf-8') as f:
            json_content = f.read()
        
        json_result = process_text(json_content, format="json")
        self.assertEqual(json_result["format"], "json")
    
    def _verify_common_result_fields(self, result: Dict[str, Any]):
        """Helper to verify common fields in adapter processing results."""
        # Check for required fields
        required_fields = ["id", "source", "content", "content_type", "format", "metadata", "entities"]
        for field in required_fields:
            self.assertIn(field, result, f"Missing required field: {field}")
        
        # Content should be non-empty
        self.assertTrue(result["content"], "Content should not be empty")
        
        # ID should be a string
        self.assertIsInstance(result["id"], str)
        
        # Should have metadata dict
        self.assertIsInstance(result["metadata"], dict)
        
        # Should have entities list
        self.assertIsInstance(result["entities"], list)


if __name__ == "__main__":
    unittest.main()

"""
Tests for the JSON serializer module.
"""

import json
import os
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

from src.docproc.serializers.json_serializer import (
    serialize_to_json,
    save_to_json_file,
    _make_json_serializable
)


class TestJsonSerializer(unittest.TestCase):
    """Test cases for JSON serialization functions."""
    
    def test_serialize_to_json_basic(self):
        """Test basic serialization with minimal input."""
        result = {"content": "test content"}
        serialized = serialize_to_json(result)
        
        # Basic content should be preserved
        self.assertEqual(serialized["content"], "test content")
        
        # Version and timestamp should be added by default
        self.assertTrue("version" in serialized)
        self.assertTrue("timestamp" in serialized)
    
    def test_serialize_to_json_with_entities(self):
        """Test serialization with entities."""
        result = {
            "content": "test content",
            "entities": [
                {"name": "Entity1", "type": "Class", "line": 10},
                {"name": "Entity2", "type": "Function", "line": 20}
            ]
        }
        
        serialized = serialize_to_json(result)
        
        # Entities should be preserved
        self.assertEqual(len(serialized["entities"]), 2)
        self.assertEqual(serialized["entities"][0]["name"], "Entity1")
        self.assertEqual(serialized["entities"][1]["type"], "Function")
    
    def test_serialize_to_json_with_format(self):
        """Test serialization with format information."""
        result = {
            "content": "test content", 
            "format": "python"
        }
        
        serialized = serialize_to_json(result)
        
        # Format should be preserved
        self.assertEqual(serialized["format"], "python")
    
    def test_serialize_to_json_with_metadata(self):
        """Test serialization with metadata."""
        result = {
            "content": "test content",
            "metadata": {
                "filename": "test.py",
                "size": 1024,
                "author": "Test User"
            }
        }
        
        serialized = serialize_to_json(result)
        
        # Metadata should be preserved
        self.assertEqual(serialized["metadata"]["filename"], "test.py")
        self.assertEqual(serialized["metadata"]["size"], 1024)
    
    def test_serialize_to_json_with_processing_time(self):
        """Test serialization with processing time."""
        result = {
            "content": "test content",
            "processing_time": 1.23
        }
        
        serialized = serialize_to_json(result)
        
        # Processing time should be moved to metadata
        self.assertEqual(serialized["metadata"]["processing_time"], 1.23)
    
    def test_serialize_to_json_without_optional_fields(self):
        """Test serialization with optional fields disabled."""
        result = {"content": "test content"}
        
        serialized = serialize_to_json(
            result,
            include_metadata=False,
            include_timestamp=False,
            include_version=False
        )
        
        # Optional fields should be excluded
        self.assertNotIn("metadata", serialized)
        self.assertNotIn("timestamp", serialized)
        self.assertNotIn("version", serialized)
    
    def test_serialize_to_json_with_custom_version(self):
        """Test serialization with custom version."""
        result = {"content": "test content"}
        
        serialized = serialize_to_json(result, version="2.0.0")
        
        # Custom version should be used
        self.assertEqual(serialized["version"], "2.0.0")
    
    def test_serialize_to_json_with_extra_fields(self):
        """Test serialization with extra non-standard fields."""
        result = {
            "content": "test content",
            "extra_field1": "value1",
            "extra_field2": 42
        }
        
        serialized = serialize_to_json(result)
        
        # Extra fields should be preserved
        self.assertEqual(serialized["extra_field1"], "value1")
        self.assertEqual(serialized["extra_field2"], 42)
    
    def test_make_json_serializable_simple_types(self):
        """Test _make_json_serializable with simple types."""
        # Simple types should be unchanged
        self.assertEqual(_make_json_serializable("string"), "string")
        self.assertEqual(_make_json_serializable(42), 42)
        self.assertEqual(_make_json_serializable(3.14), 3.14)
        self.assertEqual(_make_json_serializable(True), True)
        self.assertEqual(_make_json_serializable(None), None)
    
    def test_make_json_serializable_collections(self):
        """Test _make_json_serializable with collection types."""
        # Lists should be processed recursively
        self.assertEqual(_make_json_serializable([1, 2, "three"]), [1, 2, "three"])
        
        # Tuples should be converted to lists
        self.assertEqual(_make_json_serializable((1, 2, "three")), [1, 2, "three"])
        
        # Sets should be converted to lists
        self.assertEqual(set(_make_json_serializable({1, 2, 3})), {1, 2, 3})
        
        # Dictionaries should be processed recursively
        self.assertEqual(
            _make_json_serializable({"a": 1, "b": [2, 3]}),
            {"a": 1, "b": [2, 3]}
        )
    
    def test_make_json_serializable_custom_objects(self):
        """Test _make_json_serializable with custom objects."""
        class TestObject:
            def __init__(self):
                self.a = 1
                self.b = "two"
        
        # Custom objects should be converted to dictionaries
        serialized = _make_json_serializable(TestObject())
        self.assertEqual(serialized["a"], 1)
        self.assertEqual(serialized["b"], "two")
    
    def test_make_json_serializable_complex_types(self):
        """Test _make_json_serializable with complex types."""
        # Non-serializable types should be converted to strings
        dt = datetime.now()
        self.assertTrue(isinstance(_make_json_serializable(dt), str))
    
    def test_save_to_json_file(self):
        """Test saving serialized results to a file."""
        result = {
            "content": "test content",
            "format": "python",
            "entities": [{"name": "TestEntity", "type": "Class"}]
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test_output.json"
            
            # Save to file
            saved_path = save_to_json_file(result, output_path)
            
            # File should exist
            self.assertTrue(Path(saved_path).exists())
            
            # Content should be valid JSON
            with open(saved_path, 'r', encoding='utf-8') as f:
                loaded = json.load(f)
                
            # Verify content
            self.assertEqual(loaded["content"], "test content")
            self.assertEqual(loaded["format"], "python")
            self.assertEqual(loaded["entities"][0]["name"], "TestEntity")
            
            # Optional fields should be included by default
            self.assertTrue("version" in loaded)
            self.assertTrue("timestamp" in loaded)
    
    def test_save_to_json_file_with_string_path(self):
        """Test saving with string path instead of Path object."""
        result = {"content": "test content"}
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "test_output.json")
            
            # Save to file using string path
            saved_path = save_to_json_file(result, output_path)
            
            # File should exist
            self.assertTrue(os.path.exists(saved_path))
    
    def test_save_to_json_file_without_pretty_print(self):
        """Test saving without pretty printing."""
        result = {"content": "test content"}
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test_output.json"
            
            # Save to file without pretty printing
            save_to_json_file(result, output_path, pretty_print=False)
            
            # Read raw content
            with open(output_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Without pretty printing, there should be no newlines in content
            # (except possibly at the very end)
            self.assertLessEqual(content.count('\n'), 1)
    
    def test_save_to_json_file_creates_directories(self):
        """Test that directories are created if they don't exist."""
        result = {"content": "test content"}
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a nested path that doesn't exist yet
            output_path = Path(temp_dir) / "nested" / "dir" / "test_output.json"
            
            # Save to file - should create directories
            saved_path = save_to_json_file(result, output_path)
            
            # File and parent directories should exist
            self.assertTrue(Path(saved_path).exists())
            self.assertTrue(Path(saved_path).parent.exists())


if __name__ == "__main__":
    unittest.main()

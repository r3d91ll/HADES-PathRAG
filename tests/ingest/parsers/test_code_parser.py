"""
Tests for the code parser module.
"""
import os
import unittest
from unittest.mock import patch, mock_open, MagicMock
import pytest
from typing import Dict, List, Any

from src.ingest.parsers.code_parser import CodeParser
from src.ingest.parsers.base_parser import BaseParser


class TestCodeParser(unittest.TestCase):
    """Test suite for CodeParser class."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.parser = CodeParser()
        self.mock_content = """
def hello_world():
    \"\"\"This is a docstring.\"\"\"
    print("Hello, world!")
    
class TestClass:
    \"\"\"A test class.\"\"\"
    def __init__(self):
        self.x = 1
        
    def method(self):
        \"\"\"A method.\"\"\"
        return self.x
"""

    @patch("builtins.open", new_callable=mock_open)
    def test_parse_python_file_successful(self, mock_file: MagicMock) -> None:
        """Test successful parsing of a Python file."""
        mock_file.return_value.__enter__.return_value.read.return_value = self.mock_content
        
        # Act
        result = self.parser.parse_python_file("test.py")
        
        # Assert
        self.assertIsNotNone(result)
        self.assertEqual(len(result.functions), 1)
        self.assertEqual(len(result.classes), 1)
        self.assertIn("hello_world", result.functions)
        self.assertIn("TestClass", result.classes)
        
    @patch("builtins.open", new_callable=mock_open)
    def test_parse_python_file_syntax_error(self, mock_file: MagicMock) -> None:
        """Test handling of syntax errors in Python files."""
        # Arrange
        mock_file.return_value.__enter__.return_value.read.return_value = "def broken_syntax("
        
        # Act & Assert
        with self.assertRaises(SyntaxError):
            self.parser.parse_python_file("broken.py")
            
    @patch("builtins.open")
    def test_parse_python_file_io_error(self, mock_file: MagicMock) -> None:
        """Test handling of IO errors when opening Python files."""
        # Arrange
        mock_file.side_effect = IOError("File not found")
        
        # Act & Assert
        with self.assertRaises(IOError):
            self.parser.parse_python_file("nonexistent.py")
            
    def test_parse_directory_with_valid_files(self) -> None:
        """Test parsing a directory with valid Python files."""
        # Arrange
        with patch("os.walk") as mock_walk, \
             patch.object(self.parser, "parse_python_file") as mock_parse:
            
            mock_walk.return_value = [
                ("/root", ["subdir"], ["file1.py", "file2.py"]),
                ("/root/subdir", [], ["file3.py"])
            ]
            
            mock_parse.return_value = MagicMock()
            
            # Act
            results = self.parser.parse_directory("/root")
            
            # Assert
            self.assertEqual(len(results), 3)
            mock_parse.assert_any_call("/root/file1.py")
            mock_parse.assert_any_call("/root/file2.py")
            mock_parse.assert_any_call("/root/subdir/file3.py")
            
    def test_parse_directory_skips_non_python_files(self) -> None:
        """Test that non-Python files are skipped during directory parsing."""
        # Arrange
        with patch("os.walk") as mock_walk, \
             patch.object(self.parser, "parse_python_file") as mock_parse:
            
            mock_walk.return_value = [
                ("/root", [], ["file1.py", "file2.txt", "file3.md"])
            ]
            
            mock_parse.return_value = MagicMock()
            
            # Act
            results = self.parser.parse_directory("/root")
            
            # Assert
            self.assertEqual(len(results), 1)
            mock_parse.assert_called_once_with("/root/file1.py")
            
    def test_parse_directory_handles_parse_errors(self) -> None:
        """Test that parsing errors in individual files don't stop directory parsing."""
        # Arrange
        with patch("os.walk") as mock_walk, \
             patch.object(self.parser, "parse_python_file") as mock_parse:
            
            mock_walk.return_value = [
                ("/root", [], ["file1.py", "file2.py"])
            ]
            
            # First call succeeds, second raises an exception
            mock_parse.side_effect = [MagicMock(), SyntaxError("Bad syntax")]
            
            # Act
            results = self.parser.parse_directory("/root")
            
            # Assert
            self.assertEqual(len(results), 1)  # Only one successful parse
            
    def test_is_subclass_of_base_parser(self) -> None:
        """Test that CodeParser is a subclass of BaseParser."""
        self.assertTrue(issubclass(CodeParser, BaseParser))


# Add pytest marker for categorization
pytestmark = pytest.mark.parsers

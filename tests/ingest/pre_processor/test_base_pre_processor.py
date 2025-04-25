"""
Tests for the base pre-processor module.
"""
import unittest
from unittest.mock import patch, MagicMock
import pytest
from typing import Dict, List, Any, Optional
import abc

from src.ingest.pre_processor.base_pre_processor import BasePreProcessor


class TestBasePreProcessor(unittest.TestCase):
    """Test suite for BasePreProcessor class."""
    
    def test_is_abstract_class(self) -> None:
        """Test that BasePreProcessor is an abstract base class."""
        self.assertTrue(issubclass(BasePreProcessor, abc.ABC))
    
    def test_cannot_instantiate_directly(self) -> None:
        """Test that BasePreProcessor cannot be instantiated directly."""
        with self.assertRaises(TypeError):
            BasePreProcessor()
    
    def test_required_abstract_methods(self) -> None:
        """Test that BasePreProcessor defines required abstract methods."""
        # Check that process_file is an abstract method
        self.assertTrue(hasattr(BasePreProcessor, 'process_file'))
        # In Python 3.11+, __isabstractmethod__ is a boolean, not an iterable
        is_abstract = getattr(BasePreProcessor.process_file, '__isabstractmethod__', False)
        self.assertTrue(is_abstract)
        
        # Check that process_batch is a method (but not abstract)
        self.assertTrue(hasattr(BasePreProcessor, 'process_batch'))


class ConcretePreProcessor(BasePreProcessor):
    """Concrete implementation of BasePreProcessor for testing."""
    
    def process_file(self, file_path: str) -> Dict[str, Any]:
        """Implement required abstract method."""
        return {"path": file_path, "processed": True}


class TestConcretePreProcessor(unittest.TestCase):
    """Test suite for a concrete implementation of BasePreProcessor."""
    
    def setUp(self) -> None:
        """Set up test fixtures."""
        self.processor = ConcretePreProcessor()
    
    def test_can_instantiate_concrete_subclass(self) -> None:
        """Test that a concrete subclass can be instantiated."""
        self.assertIsNotNone(self.processor)
        self.assertTrue(isinstance(self.processor, BasePreProcessor))
    
    def test_process_file_implementation(self) -> None:
        """Test the concrete implementation of process_file."""
        # Act
        result = self.processor.process_file("test_file.txt")
        
        # Assert
        self.assertEqual(result["path"], "test_file.txt")
        self.assertTrue(result["processed"])
    
    def test_process_batch_default_implementation(self) -> None:
        """Test the default implementation of process_batch."""
        # Arrange
        import tempfile
        import os
        
        # Create temporary files
        temp_files = []
        for i in range(3):
            fd, path = tempfile.mkstemp(suffix=".txt")
            os.write(fd, f"test content {i}".encode('utf-8'))
            os.close(fd)
            temp_files.append(path)
        
        try:
            # Act
            results = self.processor.process_batch(temp_files)
            
            # Assert
            self.assertEqual(len(results), 3)
        finally:
            # Clean up temporary files
            for path in temp_files:
                if os.path.exists(path):
                    os.unlink(path)
        # Check that the results match our temp files
        self.assertEqual(results[0]["path"], temp_files[0])
        self.assertEqual(results[1]["path"], temp_files[1])
        self.assertEqual(results[2]["path"], temp_files[2])
    
    def test_process_batch_handles_errors(self) -> None:
        """Test that process_batch continues despite errors in individual files."""
        # Arrange
        import tempfile
        import os
        
        # Create temporary files
        temp_files = []
        for i in range(3):
            fd, path = tempfile.mkstemp(suffix=".txt")
            os.write(fd, f"test content {i}".encode('utf-8'))
            os.close(fd)
            temp_files.append(path)
        
        # Mock process_file to fail for the second file
        original_process_file = self.processor.process_file
        
        def mock_process_file(file_path: str) -> Dict[str, Any]:
            if file_path == temp_files[1]:  # Fail on the second file
                raise Exception("Processing error")
            return original_process_file(file_path)
        
        self.processor.process_file = mock_process_file
        
        try:
            # Act
            results = self.processor.process_batch(temp_files)
            
            # Assert
            self.assertEqual(len(results), 2)  # Only 2 successful files
        finally:
            # Restore original method
            self.processor.process_file = original_process_file
            # Clean up temporary files
            for path in temp_files:
                if os.path.exists(path):
                    os.unlink(path)
        # Check that we have the first and third files (second should have failed)
        self.assertEqual(results[0]["path"], temp_files[0])
        self.assertEqual(results[1]["path"], temp_files[2])
    
    def test_process_batch_with_empty_list(self) -> None:
        """Test process_batch with an empty file list."""
        # Act
        results = self.processor.process_batch([])
        
        # Assert
        self.assertEqual(len(results), 0)
        self.assertEqual(len(self.processor.errors), 0)


# Add pytest marker for categorization
pytestmark = pytest.mark.pre_processor

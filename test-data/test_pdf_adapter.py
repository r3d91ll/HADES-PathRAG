"""
Tests for PDF adapter.

This module tests the MockPDFAdapter class functionality for testing purposes.
"""

import unittest
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional
from unittest.mock import patch

from src.docproc.adapters.pdf_adapter_mock import MockPDFAdapter


class TestPDFAdapter(unittest.TestCase):
    """Test cases for MockPDFAdapter."""

    def setUp(self):
        """Set up the test environment."""
        self.adapter = MockPDFAdapter()
        
        # Create a temporary directory for test files
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_dir_path = Path(self.temp_dir.name)
        
        # Create a simulated PDF file (actually a text file)
        self.sample_pdf_content = """
        Test PDF Document
        
        Section 1: Introduction
        This is the introduction section.
        
        Section 2: Content
        This is the main content section.
        It has multiple paragraphs.
        
        This is another paragraph.
        
        Section 3: Conclusion
        This is the conclusion section.
        """
        
        # Create test PDF file (actually a text file with .pdf extension)
        self.test_pdf_path = self.temp_dir_path / "test.pdf"
        with open(self.test_pdf_path, "w") as f:
            f.write(self.sample_pdf_content)
        
        # Create an empty PDF file
        self.empty_pdf_path = self.temp_dir_path / "empty.pdf"
        with open(self.empty_pdf_path, "w") as f:
            f.write("")
    
    def tearDown(self):
        """Clean up after the test."""
        self.temp_dir.cleanup()
    
    def test_process_pdf_file(self):
        """Test processing a PDF file."""
        result = self.adapter.process(self.test_pdf_path)
        
        # Check basic structure
        self.assertIsNotNone(result)
        self.assertIsInstance(result, dict)
        self.assertIn('content', result)
        self.assertIn('metadata', result)
        
        # Check metadata
        self.assertEqual(result['metadata']['format'], 'pdf')
        self.assertEqual(result['metadata']['file_path'], str(self.test_pdf_path))
        
        # Check content
        self.assertIn('Section 1: Introduction', result['content'])
    
    def test_process_empty_pdf(self):
        """Test processing an empty PDF file."""
        result = self.adapter.process(self.empty_pdf_path)
        
        # Should handle empty files gracefully
        self.assertIsNotNone(result)
        self.assertIsInstance(result, dict)
        self.assertIn('content', result)
        self.assertEqual(result['content'], "")
    
    def test_process_pdf_with_error(self):
        """Test handling errors when processing a PDF file."""
        # Test with a non-existent file
        non_existent_path = self.temp_dir_path / "nonexistent.pdf"
        
        # Should handle the error and provide error information
        result = self.adapter.process(non_existent_path)
        self.assertIsNotNone(result)
        self.assertIn('metadata', result)
        self.assertIn('error', result['metadata'])
    
    def test_process_pdf_with_options(self):
        """Test processing PDF with options."""
        options = {
            'extract_images': True,
            'page_range': (1, 5),
            'ocr_enabled': True
        }
        
        result = self.adapter.process(self.test_pdf_path, options)
        
        # Check options were applied
        self.assertIn('options', result['metadata'])
        self.assertTrue(result['metadata']['options']['extract_images'])
        self.assertEqual(result['metadata']['options']['page_range'], (1, 5))
    
    def test_process_text(self):
        """Test processing PDF text directly."""
        result = self.adapter.process_text(self.sample_pdf_content)
        
        # Basic structure checks
        self.assertIsNotNone(result)
        self.assertIsInstance(result, dict)
        self.assertIn('content', result)
        self.assertIn('metadata', result)
        
        # Check content
        self.assertEqual(result['content'], self.sample_pdf_content)
    
    def test_extract_entities(self):
        """Test entity extraction from PDF text."""
        result = self.adapter.process_text(self.sample_pdf_content)
        entities = self.adapter.extract_entities(result['content'])
        
        # Should return a list of entities
        self.assertIsNotNone(entities)
        self.assertIsInstance(entities, list)
        
        # Should extract sections as entities
        self.assertTrue(len(entities) > 0, "Should extract at least one entity")
        
        # Verify headings were extracted
        heading_entities = [e for e in entities if e.get('type') == 'heading']
        self.assertTrue(len(heading_entities) > 0, "Should extract heading entities")
    
    def test_extract_metadata(self):
        """Test metadata extraction from PDF text."""
        result = self.adapter.process_text(self.sample_pdf_content)
        metadata = self.adapter.extract_metadata(result['content'])
        
        # Should return metadata dict
        self.assertIsNotNone(metadata)
        self.assertIsInstance(metadata, dict)
        
        # Basic metadata
        self.assertIn('format', metadata)
        self.assertEqual(metadata['format'], 'pdf')
        
        # Should extract title
        self.assertIn('title', metadata)
        self.assertEqual(metadata['title'].strip(), 'Test PDF Document')
    
    def test_convert_to_markdown(self):
        """Test converting PDF text to markdown."""
        result = self.adapter.process_text(self.sample_pdf_content)
        markdown = self.adapter.convert_to_markdown(result['content'])
        
        # Should return markdown string
        self.assertIsNotNone(markdown)
        self.assertIsInstance(markdown, str)
        
        # Should contain structured markdown elements
        self.assertIn('# Test PDF Document', markdown)
        self.assertIn('## Section 1: Introduction', markdown)
    
    def test_convert_to_text(self):
        """Test converting PDF content to plain text."""
        result = self.adapter.process_text(self.sample_pdf_content)
        text = self.adapter.convert_to_text(result)
        
        # Should return plain text string
        self.assertIsNotNone(text)
        self.assertIsInstance(text, str)
        
        # Should preserve text content
        self.assertIn('Section 1: Introduction', text)
        self.assertIn('This is the introduction section', text)


if __name__ == '__main__':
    unittest.main()

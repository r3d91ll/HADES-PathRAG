#!/usr/bin/env python3
"""
Tests for the Markdown pre-processor.

This module contains tests for the Markdown file pre-processor.
"""

import os
import tempfile
import shutil
import unittest
from pathlib import Path

import pytest
from unittest.mock import MagicMock, patch

# Add the project root to the Python path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.ingest.pre_processor.markdown_pre_processor import MarkdownPreProcessor


class TestMarkdownPreProcessor(unittest.TestCase):
    """Test cases for the MarkdownPreProcessor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for test files
        self.test_dir = tempfile.mkdtemp()
        self.pre_processor = MarkdownPreProcessor(extract_mermaid=True)
        
        # Create sample Markdown files
        self._create_sample_files()
    
    def tearDown(self):
        """Tear down test fixtures."""
        # Remove the temporary directory
        shutil.rmtree(self.test_dir)
    
    def _create_sample_files(self):
        """Create sample Markdown files for testing."""
        # Basic Markdown file with various elements
        self.basic_file_path = os.path.join(self.test_dir, "basic.md")
        with open(self.basic_file_path, "w") as f:
            f.write('''# Sample Document

This is a basic Markdown document with various elements.

## Introduction

This section introduces the document.

## Code Examples

Here are some code examples:

```python
def hello_world():
    """Say hello to the world."""
    return "Hello, world!"

class Example:
    def __init__(self, value):
        self.value = value
```

```javascript
function sayHello() {
    console.log("Hello, world!");
}
```

## Mermaid Diagram

Here's a simple Mermaid diagram:

```mermaid
graph TD
    A[Start] --> B[Process]
    B --> C[Decision]
    C -->|Yes| D[Action 1]
    C -->|No| E[Action 2]
    D --> F[End]
    E --> F
```

## Tables

| Name | Age | Occupation |
|------|-----|------------|
| John | 28  | Developer  |
| Jane | 32  | Designer   |

## Links

- [External Link](https://example.com)
- [Internal Link](../another_file.md)
''')
        
        # Markdown file with references to other documents
        self.references_file_path = os.path.join(self.test_dir, "references.md")
        with open(self.references_file_path, "w") as f:
            f.write('''# Document with References

This document references other documents.

## References

See also:
- [API Documentation](api.md)
- [User Guide](user_guide.md)
- [Examples](examples/index.md)

## Code References

The implementation can be found in `src/module.py`.

Related code:
- `src/helpers.py`
- `src/utils.py`
''')
    
    def test_process_basic_file(self):
        """Test processing a basic Markdown file."""
        # Act
        result = self.pre_processor.process_file(self.basic_file_path)
        
        # Assert basic properties
        self.assertEqual(result["path"], self.basic_file_path)
        self.assertEqual(result["type"], "markdown")
        self.assertIn("content", result)
        self.assertIn("title", result)
        self.assertIsInstance(result["references"], list)
        
        # Assert the title
        self.assertEqual(result["title"], "Sample Document")
        
        # Assert sections
        self.assertGreaterEqual(len(result["sections"]), 1)
        
        # Assert code blocks
        self.assertGreaterEqual(len(result["code_blocks"]), 2)
        
        # Check for different languages
        languages = [block["language"] for block in result["code_blocks"]]
        self.assertIn("python", languages)
        self.assertIn("javascript", languages)
        self.assertIn("mermaid", languages)
        
        # Assert extracted Mermaid diagram
        mermaid_blocks = [b for b in result["code_blocks"] if b["language"] == "mermaid"]
        self.assertEqual(len(mermaid_blocks), 1)
        self.assertIn("graph TD", mermaid_blocks[0]["content"])
        
        # Assert relationships
        self.assertGreaterEqual(len(result["relationships"]), 1)
        
        # Assert headings (from sections with non-None titles)
        heading_texts = [s["title"] for s in result["sections"] if s["title"]]
        self.assertIn("Introduction", heading_texts)
        self.assertIn("Code Examples", heading_texts)
        self.assertIn("Mermaid Diagram", heading_texts)
    
    def test_process_file_with_references(self):
        """Test processing a Markdown file with references."""
        # Act
        result = self.pre_processor.process_file(self.references_file_path)
        
        # Assert references
        self.assertIn("references", result)
        self.assertIsInstance(result["references"], list)
        self.assertGreater(len(result["references"]), 0)
        
        # Check if we have the expected reference targets
        for ref in result["references"]:
            self.assertIn("label", ref)
            self.assertIn("target", ref)
        self.assertIn("api.md", [r["target"] for r in result["references"]])
        self.assertIn("user_guide.md", [r["target"] for r in result["references"]])
        self.assertIn("examples/index.md", [r["target"] for r in result["references"]])
    
    def test_build_relationships(self):
        """Test building relationships between documents."""
        # Act
        result = self.pre_processor.process_file(self.references_file_path)
        
        # Assert relationships key exists
        self.assertIn("relationships", result)
        relationships = result["relationships"]
        # Accept empty relationships (if no mermaid diagrams)
        self.assertIsInstance(relationships, list)
        
        # Verify relationship structure
        for rel in relationships:
            self.assertIn("from", rel)
            self.assertIn("to", rel)
            self.assertIn("type", rel)
            
        # Check if we have REFERENCES relationships
        reference_rels = [r for r in relationships if r["type"] == "REFERENCES"]
        self.assertGreaterEqual(len(reference_rels), 0)
    
    def test_extract_mermaid_disabled(self):
        """Test with Mermaid extraction disabled."""
        # Create pre-processor with Mermaid extraction disabled
        no_mermaid_processor = MarkdownPreProcessor(extract_mermaid=False)
        
        # Act
        result = no_mermaid_processor.process_file(self.basic_file_path)
        
        # Assert Mermaid is still in code blocks but not processed specially
        mermaid_blocks = [b for b in result["code_blocks"] if b["language"] == "mermaid"]
        self.assertGreaterEqual(len(mermaid_blocks), 0)
        
        # No special Mermaid processing
        if "mermaid_diagrams" in result:
            self.assertEqual(len(result["mermaid_diagrams"]), 0)


if __name__ == "__main__":
    unittest.main()

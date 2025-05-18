#!/usr/bin/env python
"""
Comprehensive document processing verification script.

This script tests various document types and processing methods to verify:
1. The structure of the JSON output
2. Required fields are present
3. Proper error handling
4. Format detection functionality
"""

import json
import sys
import tempfile
from pathlib import Path
from pprint import pprint

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.docproc.manager import DocumentProcessorManager
from src.docproc.core import process_document, process_text
from src.docproc.core import get_format_for_document


def print_json_structure(data, prefix="", is_last=True, max_str_length=80):
    """Print the structure of a JSON object with types."""
    if isinstance(data, dict):
        print(f"{prefix}{'└── ' if is_last else '├── '}dict ({len(data)} keys)")
        prefix = prefix + ('    ' if is_last else '│   ')
        items = list(data.items())
        for i, (k, v) in enumerate(items):
            is_last_item = i == len(items) - 1
            print(f"{prefix}{'└── ' if is_last_item else '├── '}{k} ({type(v).__name__}):", end="")
            
            if isinstance(v, (dict, list)):
                print()  # Newline for nested structures
                print_json_structure(v, prefix + ('    ' if is_last_item else '│   '), True, max_str_length)
            else:
                str_value = str(v)
                if len(str_value) > max_str_length:
                    str_value = str_value[:max_str_length] + "..."
                print(f" {str_value}")
    
    elif isinstance(data, list):
        print(f"{prefix}{'└── ' if is_last else '├── '}list ({len(data)} items)")
        if data:
            prefix = prefix + ('    ' if is_last else '│   ')
            # Just show the first item's structure if list is not empty
            print(f"{prefix}└── [0] ({type(data[0]).__name__}):", end="")
            if isinstance(data[0], (dict, list)):
                print()
                print_json_structure(data[0], prefix + '    ', True, max_str_length)
            else:
                str_value = str(data[0])
                if len(str_value) > max_str_length:
                    str_value = str_value[:max_str_length] + "..."
                print(f" {str_value}")


def save_sample_files():
    """Create sample files of different formats for testing."""
    test_dir = project_root / "tests" / "data"
    test_dir.mkdir(exist_ok=True, parents=True)
    
    # Markdown file
    markdown_path = test_dir / "sample.md"
    markdown_content = """# Sample Markdown Document
    
## Introduction
This is a sample markdown document used for testing.

## Features
- Feature 1
- Feature 2
- Feature 3

## Code Example
```python
def hello_world():
    print("Hello, world!")
```
"""
    markdown_path.write_text(markdown_content)
    
    # Plain text file
    txt_path = test_dir / "sample.txt"
    txt_content = """Sample Text Document
    
This is a plain text document used for testing.
It contains multiple paragraphs and some formatting.

Line 1
Line 2
Line 3
"""
    txt_path.write_text(txt_content)
    
    # Python file
    py_path = test_dir / "sample.py"
    py_content = """#!/usr/bin/env python
'''Sample Python module for testing document processing.'''

def example_function(param1, param2=None):
    '''Example function with docstring.
    
    Args:
        param1: First parameter
        param2: Optional second parameter
    
    Returns:
        A result value
    '''
    result = f"Processing {param1}"
    if param2:
        result += f" with {param2}"
    return result

class ExampleClass:
    '''Example class for testing.'''
    
    def __init__(self, name):
        '''Initialize with name.'''
        self.name = name
    
    def greet(self):
        '''Return greeting.'''
        return f"Hello, {self.name}!"

if __name__ == "__main__":
    obj = ExampleClass("World")
    print(obj.greet())
"""
    py_path.write_text(py_content)
    
    return {
        "markdown": markdown_path,
        "text": txt_path,
        "python": py_path
    }


def test_with_manager():
    """Test using DocumentProcessorManager directly."""
    print("\n=== Testing with DocumentProcessorManager ===")
    
    # Create manager instance
    manager = DocumentProcessorManager()
    
    # Test with direct text content
    print("\n--- Testing with markdown text content ---")
    markdown_content = "# Test Document\n\nThis is test content."
    try:
        result = manager.process_document(content=markdown_content, doc_type="markdown")
        print("\nMarkdown Result Structure:")
        print_json_structure(result)
    except Exception as e:
        print(f"Error processing markdown content: {e}")
    
    # Test with text content
    print("\n--- Testing with plain text content ---")
    text_content = "This is plain text content for testing."
    try:
        result = manager.process_document(content=text_content, doc_type="text")
        print("\nPlain Text Result Structure:")
        print_json_structure(result)
    except Exception as e:
        print(f"Error processing text content: {e}")


def test_with_core_functions(sample_files):
    """Test using core module functions directly."""
    print("\n=== Testing with core module functions ===")
    
    # Test process_document with markdown file
    print("\n--- Testing process_document with markdown file ---")
    try:
        result = process_document(sample_files["markdown"])
        print("\nMarkdown File Structure:")
        print_json_structure(result)
    except Exception as e:
        print(f"Error processing markdown file: {e}")
    
    # Test process_document with Python file
    print("\n--- Testing process_document with Python file ---")
    try:
        result = process_document(sample_files["python"])
        print("\nPython File Structure:")
        print_json_structure(result)
    except Exception as e:
        print(f"Error processing Python file: {e}")
    
    # Test process_text
    print("\n--- Testing process_text with markdown content ---")
    markdown_content = "# Direct Test\n\nThis is a test document processed directly."
    try:
        # Explicitly specify markdown format
        result = process_text(
            markdown_content, 
            format_type="markdown",
            options={"validation_level": "warn"}  # Allow validation warnings
        )
        print("\nProcess Text Result Structure:")
        print_json_structure(result)
    except Exception as e:
        print(f"Error in process_text: {e}")


def test_format_detection(sample_files):
    """Test format detection functionality."""
    print("\n=== Testing format detection ===")
    
    for format_name, file_path in sample_files.items():
        detected = get_format_for_document(file_path)
        print(f"File: {file_path.name}, Expected: {format_name}, Detected: {detected}")
    
    # Test with content and extension
    with tempfile.NamedTemporaryFile(suffix=".unknown") as tmp:
        tmp.write(b"# Markdown content\n\nWith unknown extension")
        tmp.flush()
        
        detected = get_format_for_document(tmp.name)
        print(f"File with unknown extension: {Path(tmp.name).name}, Detected: {detected}")
        
        # Try with content analysis - passing the content directly
        with open(tmp.name, 'r') as f:
            content = f.read()
            detected = get_format_for_document(tmp.name, content=content)
            print(f"File with content analysis: {Path(tmp.name).name}, Detected: {detected}")


def main():
    """Run all tests."""
    sample_files = save_sample_files()
    print(f"Created sample files in {project_root / 'tests' / 'data'}")
    
    test_with_manager()
    test_with_core_functions(sample_files)
    test_format_detection(sample_files)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""
Test script to verify the structure of the JSON object produced by document processing.
This script processes a sample document and prints the structure of the resulting JSON.
"""

import json
import sys
from pathlib import Path
from pprint import pprint

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.docproc import process_document, process_text


def print_json_structure(data, prefix="", is_last=True, max_str_length=100):
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


def test_document_processing():
    """Test processing a document and verify JSON structure."""
    # Test with a markdown file first
    test_dir = project_root / "tests" / "data"
    
    print("\n=== Testing with sample markdown document ===")
    markdown_file = test_dir / "sample.md"
    
    # Create test file if it doesn't exist
    if not markdown_file.exists():
        markdown_content = """# Sample Document
        
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
        markdown_file.parent.mkdir(exist_ok=True)
        markdown_file.write_text(markdown_content)
        print(f"Created test file: {markdown_file}")
    
    # Process the markdown file
    result = process_document(markdown_file)
    
    # Print the structure
    print("\nDocument Structure:")
    print_json_structure(result)
    
    # Also test with process_text
    print("\n=== Testing with direct text processing ===")
    text_result = process_text("# Direct Test\nThis is a test document processed directly.")
    
    print("\nText Processing Structure:")
    print_json_structure(text_result)


if __name__ == "__main__":
    test_document_processing()

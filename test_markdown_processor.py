#!/usr/bin/env python
"""
Test script to verify markdown processing functionality.

This script tests the document processing module with a markdown file and
outputs the resulting JSON to see how the system handles markdown.
"""

import json
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

from src.docproc.adapters.docling_adapter import DoclingAdapter
from src.docproc.serializers import save_to_json_file


def test_markdown_processing():
    """Test processing a markdown document and print the results."""
    print("Testing markdown processing...")
    
    # Set up the adapter
    adapter = DoclingAdapter()
    
    # Process the markdown file
    markdown_file = Path("data/docproc.md")
    if not markdown_file.exists():
        print(f"Error: File {markdown_file} not found.")
        return
    
    print(f"Processing markdown file: {markdown_file}")
    result = adapter.process(markdown_file)
    
    # Save to output file
    output_dir = Path("test-output")
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / "markdown_test_output.json"
    save_to_json_file(result, output_file)
    
    # Print summary of the processing
    print(f"\nProcessing complete!")
    print(f"Format detected: {result.get('format', 'unknown')}")
    print(f"Content length: {len(result.get('content', ''))} characters")
    print(f"Metadata extracted:")
    for key, value in result.get('metadata', {}).items():
        print(f"  - {key}: {value}")
    
    print(f"\nFull result saved to: {output_file}")
    
    # Show the structure of entities found (if any)
    entities = result.get('entities', [])
    if entities:
        print(f"\nFound {len(entities)} entities:")
        
        # Group entities by type for cleaner output
        entity_types = {}
        for entity in entities:
            entity_type = entity.get('type', 'unknown')
            if entity_type not in entity_types:
                entity_types[entity_type] = []
            entity_types[entity_type].append(entity)
        
        # Print summary by type
        for entity_type, type_entities in entity_types.items():
            print(f"  - {entity_type}: {len(type_entities)} entities")
            
            # Show examples of each type (up to 3)
            examples = type_entities[:3]
            for example in examples:
                if 'heading' in entity_type:
                    print(f"      * {example.get('value', 'Unknown')[:40]}{'...' if len(example.get('value', '')) > 40 else ''}")
                elif entity_type == 'code_block':
                    print(f"      * Language: {example.get('language', 'unknown')}")
                elif entity_type == 'link':
                    print(f"      * {example.get('text', 'Unknown')} -> {example.get('url', 'Unknown')}")
                else:
                    print(f"      * {example.get('name', 'unnamed')[:40]}")
            
            if len(type_entities) > 3:
                print(f"      * ... and {len(type_entities) - 3} more {entity_type} entities")
    else:
        print("\nNo entities found in the markdown document.")
        print("Debug info:")
        print(f"  Content length: {len(result.get('content', ''))}")
        print(f"  Format: {result.get('format')}")
        print(f"  Entity list exists: {'entities' in result}")
        print(f"  Entity list type: {type(result.get('entities', None))}")
        print(f"  Entity list is empty: {result.get('entities', None) == []}")
        print(f"  Entity list length: {len(result.get('entities', []))}")



if __name__ == "__main__":
    test_markdown_processing()

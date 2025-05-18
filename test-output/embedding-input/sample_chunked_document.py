#!/usr/bin/env python3
"""Sample chunked document for embedding module development."""

# This is the exact JSON structure passed from chunking to embedding
SAMPLE_CHUNKED_DOCUMENT = {
    "id": "sample_text_doc",
    "path": "test-data/sample_text.txt",
    "content": "# Sample Text Document for Testing\n\nThis is a sample text document used for testing the chunking and validation functionality of the HADES-PathRAG system.\n\n## Introduction\n\nText documents are a common format for storing information. They can contain various types of content, including:\n\n1. Plain text\n2. Structured text with headings\n3. Lists and enumerations\n4. Code snippets\n\n## Processing Text Documents\n\nWhen processing text documents, the system should:\n\n- Extract the content\n- Identify any structure (headings, lists, etc.)\n- Chunk the content appropriately\n- Validate the chunks against the schema\n\n## Sample Code\n\nHere's a sample code snippet that might be found in a text document:\n\n```python\ndef process_text(text_content):\n    \"\"\"Process text content and return chunks.\"\"\"\n    chunks = []\n    paragraphs = text_content.split(\"\\n\\n\")\n    for i, paragraph in enumerate(paragraphs):\n        if paragraph.strip():\n            chunks.append({\n                \"content\": paragraph,\n                \"index\": i,\n                \"type\": \"paragraph\"\n            })\n    return chunks\n```\n\n## Conclusion\n\nText documents are versatile and can contain a mix of content types. The chunking system should be able to handle this variety and produce meaningful chunks that preserve the semantic meaning of the content.\n\nThis sample document includes multiple paragraphs, headings, lists, and a code snippet to test various aspects of the chunking system.\n",
    "type": "text",
    "metadata": {
        "format": "text",
        "language": "en",
        "creation_date": "2025-05-10",
        "author": "HADES-PathRAG Team"
    },
    "chunks": [
        {
            "id": "sample_text_doc_p0",
            "parent": "sample_text_doc",
            "parent_id": "sample_text_doc",
            "path": "test-data/sample_text.txt",
            "type": "text",
            "content": "# Sample Text Document for Testing\n\nThis is a sample text document used for testing the chunking and validation functionality of the HADES-PathRAG system.\n\n## Introduction\n\nText documents are a common format for storing information. They can contain various types of content, including:\n\n1. Plain text\n2. Structured text with headings\n3. Lists and enumerations\n4. Code snippets\n\n## Processing Text Documents\n\nWhen processing text documents, the system should:\n\n- Extract the content\n- Identify any structure (headings, lists, etc.)\n- Chunk the content appropriately\n- Validate the chunks against the schema\n\n## Sample Code\n\nHere's a sample code snippet that might be found in a text document:\n\n```python\ndef process_text(text_content):\n    \"\"\"Process text content and return chunks.\"\"\"\n    chunks = []\n    paragraphs = text_content.split(\"\\n\\n\")\n    for i, paragraph in enumerate(paragraphs):\n        if paragraph.strip():\n            chunks.append({\n                \"content\": paragraph,\n                \"index\": i,\n                \"type\": \"paragraph\"\n            })\n    return chunks\n```\n\n## Conclusion\n\nText documents are versatile and can contain a mix of content types. The chunking system should be able to handle this variety and produce meaningful chunks that preserve the semantic meaning of the content.\n\nThis sample document includes multiple paragraphs, headings, lists, and a code snippet to test various aspects of the chunking system.\n",
            "overlap_context": {
                "pre": "",
                "post": "",
                "position": 0,
                "total": 1
            },
            "symbol_type": "paragraph",
            "name": "paragraph_0",
            "chunk_index": 0,
            "start_offset": 0,
            "end_offset": 1450,
            "line_start": 0,
            "line_end": 0,
            "token_count": 197,
            "content_hash": "851ab8b07e006fc8ab95cd72140cf53d",
            "embedding": null
        }
    ]
}

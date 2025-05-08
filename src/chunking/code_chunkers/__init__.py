"""Code chunkers package.

Exports a single helper ``chunk_code`` that selects the appropriate
chunking strategy for a given language.  Today we only implement the
Python AST-based splitter, but the dispatcher pattern allows easy
extension to other languages (e.g., JavaScript, Java) later.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from .ast_chunker import chunk_python_code


# Language dispatcher mapping
_LANG_DISPATCH = {
    "python": chunk_python_code,
    # Future languages can be added here
    # "javascript": chunk_js_code,
}


def chunk_code(
    document: Dict[str, Any], *, max_tokens: int = 2048, output_format: str = "python"
) -> Union[List[Dict[str, Any]], str]:
    """Chunk a pre-processed *code* document into deterministic pieces.

    Args:
        document: Output from a pre-processor (e.g. ``PythonPreProcessor``) that
            contains at least ``source`` (raw code), ``path`` (file path), and 
            symbol metadata like ``functions``/``classes``.
        max_tokens: Token budget per chunk (approximate; chunker will fall back to 
            line-based slicing if a single node exceeds the limit).
        output_format: Format of the output (e.g. "python", "json")

    Returns:
        A list of chunk dictionaries, each containing:
        - id: Unique ID for the chunk
        - path: Original file path
        - type: Language type (e.g. "python")
        - content: The actual code for this chunk
        - line_start: Starting line number in original file
        - line_end: Ending line number in original file
        - symbol_type: Type of symbol (e.g. "function", "class", "module")
        - parent: ID of parent node (file or class) if applicable
        - name: Symbol name if applicable
        Or a string in the specified output format.

    Raises:
        ValueError: If no chunker is registered for the document's language
    """
    # Determine language from document
    lang = document.get("type") or document.get("language") or "unknown"
    lang = lang.lower()
    
    # Get appropriate chunker
    chunker = _LANG_DISPATCH.get(lang)
    if chunker is None:
        raise ValueError(f"No chunker registered for language: {lang}")
    
    # Apply chunking strategy
    return chunker(document, max_tokens=max_tokens, output_format=output_format)

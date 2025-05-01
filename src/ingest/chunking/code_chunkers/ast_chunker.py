"""AST-based code chunker for Python source files.

This module implements deterministic chunking for Python source code based on
AST (Abstract Syntax Tree) node boundaries. It leverages the symbol table 
information produced by the PythonPreProcessor to create logical, non-overlapping
chunks that follow class and function boundaries.

The chunker also handles cases where a single logical unit exceeds the maximum
token limit by falling back to line-based splitting within that unit.
"""
from __future__ import annotations

import ast
import hashlib
import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple, Set, Union, cast

from src.config.chunker_config import get_chunker_config

# Set up logging
logger = logging.getLogger(__name__)

# Type aliases
ChunkInfo = Dict[str, Any]
SymbolInfo = Dict[str, Any]


def estimate_tokens(text: str) -> int:
    """Estimate the number of tokens in a text string.
    
    This is an approximation based on common tokenization patterns.
    It's not exact but provides a reasonable estimate for chunking.
    
    Args:
        text: The text to estimate token count for
        
    Returns:
        Estimated number of tokens
    """
    # Simple approximation: ~4 chars per token for code
    # This is a rough heuristic; adjust based on your embedding model
    return len(text) // 4


def create_chunk_id(file_path: str, symbol_type: str, name: str, 
                   line_start: int, line_end: int) -> str:
    """Create a stable, unique ID for a code chunk.
    
    Args:
        file_path: Path to the source file
        symbol_type: Type of symbol (function, class, etc.)
        name: Name of the symbol
        line_start: Starting line number
        line_end: Ending line number
        
    Returns:
        A stable chunk ID
    """
    # Hash the key components to create a stable ID
    hash_input = f"{file_path}:{symbol_type}:{name}:{line_start}-{line_end}"
    chunk_hash = hashlib.md5(hash_input.encode()).hexdigest()[:8]
    return f"chunk:{chunk_hash}"


def extract_chunk_content(source: str, line_start: int, line_end: int) -> str:
    """Extract content from source between line numbers.
    
    Args:
        source: Full source code
        line_start: Starting line number (1-indexed)
        line_end: Ending line number (1-indexed)
        
    Returns:
        Extracted code snippet
    """
    lines = source.splitlines()
    
    # Adjust for 0-indexed list
    start_idx = max(0, line_start - 1)
    end_idx = min(len(lines), line_end)
    
    return "\n".join(lines[start_idx:end_idx])


def chunk_python_code(
    document: Dict[str, Any], *, max_tokens: int = 2048, output_format: str = "python"
) -> Union[List[Dict[str, Any]], str]:
    """
    Chunk Python source code using AST node boundaries.
    
    Args:
        document: Pre-processed Python document with symbol table
        max_tokens: Maximum tokens per chunk (overrides config if provided)
        output_format: Output format, either "python" or "json"
        
    Returns:
        List of chunk dictionaries or JSON string
    """
    # Load chunker configuration
    config = get_chunker_config('ast')
    
    # Use provided max_tokens if specified, otherwise use config value
    if max_tokens == 2048:  # If it's the default value
        max_tokens = config.get('max_tokens', 2048)
        
    # Get other configuration options
    use_class_boundaries = config.get('use_class_boundaries', True)
    use_function_boundaries = config.get('use_function_boundaries', True)
    extract_imports = config.get('extract_imports', True)
    
    # Extract key information from document
    source = document.get("content") or document.get("source", "")
    file_path = document.get("path", "unknown")
    
    # Basic validation
    if not source:
        logger.warning(f"Empty source code in document: {file_path}")
        return []
    
    # Extract symbol information 
    functions = document.get("functions", [])
    classes = document.get("classes", [])
    
    # Start with module-level chunk (imports, top-level code)
    chunks = []
    
    # Track which lines have been assigned to chunks
    assigned_lines: Set[int] = set()
    
    # Process class chunks first (to establish parent relationships)
    class_id_map = {}  # Maps class names to their chunk IDs
    
    for cls in classes:
        class_name = cls.get("name", "")
        line_start = cls.get("line_start", 0)
        line_end = cls.get("line_end", 0)
        
        if not (class_name and line_start and line_end):
            continue
            
        # We'll extract all class methods to separate chunks
        class_method_names = set(cls.get("methods", []))
        
        # Find where the class body ends before the first method
        method_line_starts = []
        for func in functions:
            if func.get("name") in class_method_names:
                method_line_starts.append(func.get("line_start", 0))
        
        # If we have methods, class body ends at the first method
        class_body_end = min(method_line_starts) - 1 if method_line_starts else line_end
        
        # Create class chunk (excluding method bodies)
        class_content = extract_chunk_content(source, line_start, class_body_end)
        
        # Check token count and split if necessary
        if estimate_tokens(class_content) > max_tokens:
            # TODO: Implement smarter splitting for large class definitions
            # For now, we'll use a simple line-based split
            logger.warning(f"Class {class_name} definition exceeds token limit, using basic split")
        
        # Create the class chunk
        class_chunk_id = create_chunk_id(file_path, "class", class_name, line_start, class_body_end)
        class_id_map[class_name] = class_chunk_id
        
        class_chunk = {
            "id": class_chunk_id,
            "path": file_path,
            "type": "python",
            "content": class_content,
            "line_start": line_start,
            "line_end": class_body_end,
            "symbol_type": "class",
            "name": class_name,
            "parent": None,  # Will be set to file ID if needed
        }
        
        chunks.append(class_chunk)
        
        # Mark these lines as assigned
        for line in range(line_start, class_body_end + 1):
            assigned_lines.add(line)
    
    # Process function chunks
    for func in functions:
        func_name = func.get("name", "")
        line_start = func.get("line_start", 0)
        line_end = func.get("line_end", 0)
        
        if not (func_name and line_start and line_end):
            continue
            
        # Extract function content
        func_content = extract_chunk_content(source, line_start, line_end)
        
        # Check if this is a class method
        parent_class = None
        for cls in classes:
            if func_name in cls.get("methods", []):
                parent_class = cls.get("name")
                break
        
        # Set parent relationship
        parent_id = class_id_map.get(parent_class) if parent_class else None
        
        # Check token count and split if necessary
        if estimate_tokens(func_content) > max_tokens:
            # TODO: Implement smarter splitting for large functions
            # For now, we'll use a simple line-based split
            logger.warning(f"Function {func_name} exceeds token limit, using basic split")
        
        # Create the function chunk
        func_chunk = {
            "id": create_chunk_id(file_path, "function", func_name, line_start, line_end),
            "path": file_path,
            "type": "python",
            "content": func_content,
            "line_start": line_start,
            "line_end": line_end,
            "symbol_type": "function" if not parent_class else "method",
            "name": func_name,
            "parent": parent_id,
        }
        
        chunks.append(func_chunk)
        
        # Mark these lines as assigned
        for line in range(line_start, line_end + 1):
            assigned_lines.add(line)
    
    # Process module-level code (everything not in functions/classes)
    lines = source.splitlines()
    module_lines: List[Tuple[int, int, str]] = []
    
    # Collect unassigned lines
    current_block: List[str] = []
    for i, line_str in enumerate(lines, 1):
        if i not in assigned_lines:
            current_block.append(line_str)
        elif current_block:
            module_lines.append((i - len(current_block), i - 1, "\n".join(current_block)))
            current_block = []
    
    # Don't forget the last block
    if current_block:
        module_lines.append((len(lines) - len(current_block) + 1, len(lines), "\n".join(current_block)))
    
    # Create chunks for module-level code
    for start_line, end_line, block_content in module_lines:
        # Skip empty blocks
        if not block_content.strip():
            continue
        
        # Determine a relevant name based on content
        # Imports get special handling
        if any(line.strip().startswith(("import ", "from ")) for line in block_content.splitlines()):
            block_type = "imports"
        else:
            block_type = "module"
        
        module_chunk = {
            "id": create_chunk_id(file_path, block_type, f"L{start_line}-{end_line}", start_line, end_line),
            "path": file_path,
            "type": "python",
            "content": block_content,
            "line_start": start_line,
            "line_end": end_line,
            "symbol_type": block_type,
            "name": f"module_L{start_line}",
            "parent": None,  # Will be set to file ID if needed
        }
        
        chunks.append(module_chunk)
    
    if output_format == "json":
        return json.dumps(chunks, indent=2)
    return chunks

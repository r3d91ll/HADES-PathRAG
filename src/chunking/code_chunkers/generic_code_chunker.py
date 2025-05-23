"""
Generic code chunker for HADES-PathRAG.

This module provides a generic code chunker that can handle various code formats
by applying simple line-based chunking with code-specific boundary detection.
"""

import logging
import hashlib
from typing import Dict, List, Any, Optional, Union
from pathlib import Path

from ..base import BaseChunker

logger = logging.getLogger(__name__)


class GenericCodeChunker(BaseChunker):
    """Generic code chunker that works with various code formats.
    
    This chunker divides code into logical chunks by detecting common code boundaries
    such as function definitions, class definitions, and import blocks, while also
    respecting a maximum chunk size.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the generic code chunker.
        
        Args:
            config: Configuration for the chunker including max_chunk_size, 
                   min_chunk_size, and overlap
        """
        super().__init__(name="generic_code", config=config or {})
        
        # Set defaults for configuration
        self.max_chunk_size = self.config.get("max_chunk_size", 400)
        self.min_chunk_size = self.config.get("min_chunk_size", 50)
        self.overlap = self.config.get("overlap", 20)
        self.respect_boundaries = self.config.get("respect_boundaries", True)
        
        logger.info(f"Initialized GenericCodeChunker with max_size={self.max_chunk_size}, min_size={self.min_chunk_size}")

    def chunk(self, content=None, text=None, doc_id=None, path="unknown", doc_type="code", max_tokens=None, output_format="json", **kwargs):
        """Chunk code content using generic code boundaries.
        
        Args:
            text: Text to chunk (alternative to content)
            content: Content to chunk (alternative to text)
            doc_id: Document ID
            path: Path to the original document
            doc_type: Document type
            max_tokens: Maximum tokens per chunk (overrides config if provided)
            output_format: Output format (json or dict)
            
        Returns:
            List of chunks
        """
        # Use content if provided, otherwise use text
        if content is not None:
            code_content = content
        elif text is not None:
            code_content = text
        else:
            logger.error("No content or text provided for chunking")
            return []
            
        # Use max_tokens if provided, otherwise use max_chunk_size from config
        max_size = max_tokens if max_tokens is not None else self.max_chunk_size
        
        # Generate document ID if not provided
        if doc_id is None:
            content_hash = hashlib.md5(code_content.encode()).hexdigest()[:8]
            doc_id = f"code_{content_hash}"
            
        # Split content into lines
        lines = code_content.split("\n")
        
        # Find potential chunk boundaries (functions, classes, blocks)
        boundaries = self._find_code_boundaries(lines)
        
        # Create chunks based on boundaries and size constraints
        chunks = self._create_chunks(lines, boundaries, max_size, doc_id, path, doc_type)
        
        return chunks
        
    def _find_code_boundaries(self, lines: List[str]) -> List[int]:
        """Find potential code boundaries for chunking.
        
        Looks for common code structure elements like function definitions,
        class definitions, import blocks, etc.
        
        Args:
            lines: List of code lines
            
        Returns:
            List of line indices that represent potential chunk boundaries
        """
        boundaries = [0]  # Always include the start of the file
        
        # Common code block starters
        code_block_patterns = [
            # Function/method definitions
            ("def ", True),
            # Class definitions
            ("class ", True),
            # Import statements
            ("import ", False),
            ("from ", False),
            # Module-level constants (ALL_CAPS)
            # If line is ALL_CAPS and contains =
            (lambda line: line.strip() and line.strip()[0].isupper() and "=" in line and 
                         line.strip().split("=")[0].strip().isupper(), False),
            # Block comments and docstrings
            ('"""', False),
            ("'''", False),
            # Control structures that often define logical blocks
            (" if ", False),
            (" for ", False),
            (" while ", False),
            (" try:", False),
            (" except ", False),
            # Blank lines after non-blank lines can be soft boundaries
            (lambda idx, lines: idx > 0 and lines[idx].strip() == "" and lines[idx-1].strip() != "", False)
        ]
        
        for i, line in enumerate(lines):
            if i == 0:
                continue  # Skip the first line as it's already a boundary
                
            line_stripped = line.strip()
            
            # Check for patterns
            for pattern, is_hard_boundary in code_block_patterns:
                if callable(pattern):
                    # Handle callable patterns differently based on args
                    if pattern.__code__.co_argcount == 1:
                        is_match = pattern(line_stripped)
                    else:
                        is_match = pattern(i, lines)
                        
                    if is_match:
                        boundaries.append(i)
                        break
                elif isinstance(pattern, str) and pattern in line_stripped:
                    boundaries.append(i)
                    break
        
        # Add the end of the file
        boundaries.append(len(lines))
        
        return sorted(set(boundaries))  # Remove duplicates and ensure order
    
    def _create_chunks(self, lines: List[str], boundaries: List[int], max_size: int, doc_id: str, path: str, doc_type: str) -> List[Dict[str, Any]]:
        """Create code chunks based on detected boundaries and size constraints.
        
        Args:
            lines: List of code lines
            boundaries: List of line indices for chunk boundaries
            max_size: Maximum chunk size in characters
            doc_id: Document ID
            path: Path to the original document
            doc_type: Document type
            
        Returns:
            List of chunk dictionaries
        """
        chunks = []
        
        # Process each section between boundaries
        for i in range(len(boundaries) - 1):
            start_line = boundaries[i]
            end_line = boundaries[i + 1]
            
            # Get the content between boundaries
            section_lines = lines[start_line:end_line]
            section_text = "\n".join(section_lines)
            
            # If the section is too large, split it further
            if len(section_text) > max_size and len(section_lines) > 1:
                # Simple approach: Split into roughly equal parts
                mid_point = len(section_lines) // 2
                
                # Create first chunk
                first_half = "\n".join(section_lines[:mid_point])
                chunk_id = f"{doc_id}_chunk_{len(chunks) + 1}"
                chunks.append({
                    "id": chunk_id,
                    "doc_id": doc_id,
                    "content": first_half,
                    "metadata": {
                        "source": path,
                        "start_line": start_line + 1,  # 1-indexed for user readability
                        "end_line": start_line + mid_point,
                        "doc_type": doc_type,
                        "chunk_type": "code"
                    }
                })
                
                # Create second chunk
                second_half = "\n".join(section_lines[mid_point:])
                chunk_id = f"{doc_id}_chunk_{len(chunks) + 1}"
                chunks.append({
                    "id": chunk_id,
                    "doc_id": doc_id,
                    "content": second_half,
                    "metadata": {
                        "source": path,
                        "start_line": start_line + mid_point + 1,  # 1-indexed
                        "end_line": end_line,
                        "doc_type": doc_type,
                        "chunk_type": "code"
                    }
                })
            else:
                # Section fits within max_size, create a single chunk
                chunk_id = f"{doc_id}_chunk_{len(chunks) + 1}"
                chunks.append({
                    "id": chunk_id,
                    "doc_id": doc_id,
                    "content": section_text,
                    "metadata": {
                        "source": path,
                        "start_line": start_line + 1,  # 1-indexed for user readability
                        "end_line": end_line,
                        "doc_type": doc_type,
                        "chunk_type": "code"
                    }
                })
        
        return chunks

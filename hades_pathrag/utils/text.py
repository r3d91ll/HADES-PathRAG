"""
Text processing utilities for PathRAG.

This module provides functions for processing text into chunks
and extracting entities and relationships.
"""
from typing import Dict, List, Optional, Tuple, Any, Set
import re
import logging
from dataclasses import dataclass

# Try to import tiktoken if available, otherwise use a simple fallback
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class TextChunk:
    """Represents a chunk of text with metadata."""
    content: str
    index: int
    tokens: int
    metadata: Optional[Dict[str, Any]] = None


def count_tokens(text: str, model_name: str = "gpt-4o") -> int:
    """
    Count the number of tokens in a text using tiktoken if available.
    
    Args:
        text: Text to count tokens for
        model_name: Model name for tokenizer
        
    Returns:
        Number of tokens
    """
    if not TIKTOKEN_AVAILABLE:
        # Fallback to approximate count (average 4 chars per token)
        return len(text) // 4
    
    try:
        encoding = tiktoken.encoding_for_model(model_name)
        return len(encoding.encode(text))
    except Exception as e:
        logger.warning(f"Error counting tokens: {e}")
        # Fallback
        return len(text) // 4


def chunk_text(
    text: str,
    chunk_size: int = 1024,
    chunk_overlap: int = 128,
    model_name: str = "gpt-4o",
    metadata: Optional[Dict[str, Any]] = None
) -> List[TextChunk]:
    """
    Split text into overlapping chunks of approximately equal token length.
    
    Args:
        text: Text to split
        chunk_size: Target size of each chunk in tokens
        chunk_overlap: Number of overlapping tokens between chunks
        model_name: Model name for tokenizer
        metadata: Optional metadata to include with each chunk
        
    Returns:
        List of text chunks with metadata
    """
    if not text.strip():
        return []
    
    # If text is shorter than chunk_size, return it as a single chunk
    if count_tokens(text, model_name) <= chunk_size:
        return [TextChunk(
            content=text.strip(),
            index=0,
            tokens=count_tokens(text, model_name),
            metadata=metadata
        )]
    
    # For longer text, we need to split it
    chunks: List[TextChunk] = []
    
    if TIKTOKEN_AVAILABLE:
        try:
            # Use tiktoken for precise splitting
            encoding = tiktoken.encoding_for_model(model_name)
            tokens = encoding.encode(text)
            
            for i in range(0, len(tokens), chunk_size - chunk_overlap):
                chunk_tokens = tokens[i:i + chunk_size]
                chunk_text = encoding.decode(chunk_tokens)
                
                chunks.append(TextChunk(
                    content=chunk_text.strip(),
                    index=len(chunks),
                    tokens=len(chunk_tokens),
                    metadata=metadata
                ))
                
            return chunks
        except Exception as e:
            logger.warning(f"Error chunking with tiktoken: {e}")
            # Fall back to simpler method
    
    # Simple splitting by paragraphs if tiktoken fails or is unavailable
    paragraphs = re.split(r'\n\s*\n', text)
    current_chunk = ""
    current_size = 0
    
    for para in paragraphs:
        para_size = count_tokens(para, model_name)
        
        if current_size + para_size <= chunk_size:
            # Add to current chunk
            if current_chunk:
                current_chunk += "\n\n" + para
            else:
                current_chunk = para
            current_size += para_size
        else:
            # Current chunk is full, save it
            if current_chunk:
                chunks.append(TextChunk(
                    content=current_chunk.strip(),
                    index=len(chunks),
                    tokens=current_size,
                    metadata=metadata
                ))
            
            # Start new chunk
            current_chunk = para
            current_size = para_size
    
    # Add the last chunk if not empty
    if current_chunk:
        chunks.append(TextChunk(
            content=current_chunk.strip(),
            index=len(chunks),
            tokens=current_size,
            metadata=metadata
        ))
    
    return chunks


def extract_entities(text: str) -> List[Dict[str, Any]]:
    """
    Extract entities from text.
    
    This is a placeholder for a more sophisticated NLP-based extraction.
    In a real implementation, this would use NER and other techniques.
    
    Args:
        text: Text to extract entities from
        
    Returns:
        List of entity dictionaries
    """
    # Simple extraction based on capitalized phrases
    # This is just a placeholder for demonstration
    entities = []
    
    # Find capitalized phrases (simple heuristic)
    cap_pattern = r'\b([A-Z][a-z]+(?: [A-Z][a-z]+)*)\b'
    matches = re.findall(cap_pattern, text)
    
    # Deduplicate
    unique_matches = set(matches)
    
    for i, entity in enumerate(unique_matches):
        if len(entity.split()) >= 1:  # Only phrases with at least one word
            entities.append({
                "id": f"entity-{i}",
                "name": entity,
                "type": "CONCEPT",  # Default type
                "text": entity,
                "mention_count": text.count(entity)
            })
    
    return entities

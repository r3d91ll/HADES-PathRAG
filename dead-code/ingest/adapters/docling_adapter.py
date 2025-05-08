"""
Docling Adapter for Document Parsing in HADES-PathRAG

This module provides a typed interface to Docling for converting various document formats (PDF, HTML, DOCX, etc.) into a unified structure for downstream processing.
"""
from typing import Any, Dict, Optional, List, Union
from pathlib import Path
import re

try:
    from docling.document_converter import DocumentConverter
    from docling.datamodel.base_models import InputFormat
    DOCLING_AVAILABLE = True
except ImportError:
    DOCLING_AVAILABLE = False
    # Define placeholder types for type checking
    class DocumentConverter:  # type: ignore
        """Placeholder for DocumentConverter when docling is not available."""
        def convert(self, file_path: Union[str, Path], **kwargs: Any) -> Any:
            """Placeholder convert method."""
            raise ImportError("Docling is not installed")
        
    class InputFormat:  # type: ignore
        """Placeholder for InputFormat when docling is not available."""
        pass

class DoclingAdapter:
    """
    Adapter for Docling document parsing.
    
    Implements an interface for various document analysis operations:
    - Basic document parsing
    - Text and file analysis
    - Entity, relationship, and keyword extraction
    """
    def __init__(self, options: Optional[Dict[str, Any]] = None) -> None:
        self.options = options or {}
        if DocumentConverter is None:
            raise ImportError("Docling is not installed. Please install docling to use this adapter.")
        self.converter = DocumentConverter()

    def parse(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Parse a document using Docling and return a unified structure.

        Args:
            file_path: Path to the document file (PDF, HTML, etc.)
        Returns:
            Dict with keys: source (str), content (str), docling_document (Any), format (Optional[str])
        """
        file_path = Path(file_path)
        # Infer input format if possible
        input_format = self._infer_format(file_path)
        # Convert the document
        converter = DocumentConverter()
        
        # Only attempt format handling if docling is available
        if DOCLING_AVAILABLE:
            # Try to use input_format if the API supports it
            try:
                if hasattr(InputFormat, 'AUTO'):
                    format_value = None
                    if not input_format:
                        if hasattr(InputFormat, 'AUTO'):
                            format_value = getattr(InputFormat, 'AUTO')
                    else:
                        format_value = getattr(InputFormat, input_format.upper(), getattr(InputFormat, 'AUTO', None))
                    
                    # Only pass input_format if we have a valid format
                    if format_value is not None:
                        # Use a try/except block to handle potential API differences
                        try:
                            # Use **kwargs to avoid type errors with input_format
                            kwargs = {"input_format": format_value}
                            result = converter.convert(file_path, **kwargs)
                        except TypeError:
                            # Fallback if input_format is not a valid parameter
                            result = converter.convert(file_path)
                    else:
                        result = converter.convert(file_path)
                else:
                    result = converter.convert(file_path)
            except (AttributeError, TypeError):
                # Fallback if the API doesn't match our expectations
                result = converter.convert(file_path)
        else:
            # Simple fallback when docling isn't available
            result = converter.convert(file_path)
        doc = result.document
        return {
            "source": str(file_path),
            "content": doc.export_to_markdown(),
            "docling_document": doc,
            "format": input_format if isinstance(input_format, str) else None
        }

    def analyze_text(self, text: str) -> Dict[str, Any]:
        """
        Analyze text using Docling.
        
        Args:
            text: Text content to analyze
            
        Returns:
            Docling analysis result
        """
        # In a real implementation, this would use Docling's API to analyze text
        # This is a placeholder implementation to satisfy the interface
        return {"text": text, "analysis": {"type": "basic_analysis"}}

    def analyze_file(self, file_path: Union[str, Path]) -> Any:
        """
        Analyze a file using Docling.
        
        Args:
            file_path: Path to the file to analyze
            
        Returns:
            Docling analysis result
        """
        # Convert str to Path if needed
        if isinstance(file_path, str):
            file_path = Path(file_path)
        
        # Check file extension to determine handling method
        ext = file_path.suffix.lower()
        
        # Handle PDF files - these are binary files
        if ext == '.pdf':
            # Use Docling's PDF handling capabilities
            # This relies on the actual Docling implementation
            if DOCLING_AVAILABLE:
                try:
                    # Use parse() which handles binary files
                    parsed_doc = self.parse(file_path)
                    return parsed_doc
                except Exception as e:
                    raise ValueError(f"Error processing PDF file {file_path}: {e}")
            else:
                raise ImportError("Docling is required for PDF processing")
        
        # Handle text-based files
        else:
            try:
                # Read file and analyze its content
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Delegate to analyze_text
                return self.analyze_text(content)
            except UnicodeDecodeError:
                # If we can't read as text, it might be a binary file
                raise ValueError(f"File {file_path} appears to be binary. Use the appropriate handler for this file type.")
            except Exception as e:
                raise ValueError(f"Error analyzing file {file_path}: {e}")
    
    def extract_entities(self, file_path: Union[str, Path]) -> List[Dict[str, Any]]:
        """
        Extract entities from a file.
        
        Args:
            file_path: Path to the file to analyze
            
        Returns:
            List of entities
        """
        if isinstance(file_path, str):
            file_path = Path(file_path)
        
        # Check file extension to determine handling method
        ext = file_path.suffix.lower()
        
        # For PDF files, we need to use Docling's PDF handling capabilities
        if ext == '.pdf':
            if DOCLING_AVAILABLE:
                try:
                    # First parse the PDF using Docling
                    parsed_doc = self.parse(file_path)
                    
                    # Then extract entities from the parsed content
                    if 'docling_document' in parsed_doc:
                        # Use Docling's entity extraction on the parsed document
                        # This would be the actual implementation using Docling's API
                        return self._extract_entities_from_docling_doc(parsed_doc['docling_document'])
                    else:
                        # Fallback to extracting from markdown content
                        return self._extract_entities_from_text(parsed_doc.get('content', ''))
                except Exception as e:
                    # Log error and return empty list
                    print(f"Error extracting entities from PDF {file_path}: {e}")
                    return []
            else:
                # If Docling is not available, return an empty list
                return []
        
        # For text files, read and extract entities
        else:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                return self._extract_entities_from_text(content)
            except UnicodeDecodeError:
                # If we can't read as text, it might be a binary file
                print(f"Warning: File {file_path} appears to be binary and not a supported format")
                return []
            except Exception as e:
                print(f"Error reading file {file_path}: {e}")
                return []
    
    def _extract_entities_from_text(self, text: str) -> List[Dict[str, Any]]:
        """Extract entities from text content."""
        if DOCLING_AVAILABLE:
            # Use Docling's entity extraction if available
            # This would be replaced with actual Docling API calls
            pass
        
        # For now, return a minimal implementation
        return [{"type": "entity", "value": "placeholder", "confidence": 0.9}]
    
    def _extract_entities_from_docling_doc(self, doc: Any) -> List[Dict[str, Any]]:
        """Extract entities from a Docling document object."""
        # This would use Docling-specific API to extract entities
        # For now, return a minimal implementation
        return [{"type": "entity", "value": "placeholder", "confidence": 0.9}]
    
    def extract_relationships(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract relationships from text.
        
        Args:
            text: Text to analyze
            
        Returns:
            List of relationships
        """
        # In a real implementation, this would use Docling's relationship extraction
        # This is a placeholder implementation to satisfy the interface
        return [{"type": "relationship", "source": "placeholder_source", "target": "placeholder_target", "confidence": 0.8}]
    
    def extract_keywords(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract keywords from text.
        
        Args:
            text: Text to analyze
            
        Returns:
            List of keywords with scores
        """
        if DOCLING_AVAILABLE:
            try:
                # Use Docling's keyword extraction if available
                # This would be replaced with actual Docling API calls for keyword extraction
                # For now, implementing a basic extraction approach
                words = []
                
                # Simple keyword extraction based on term frequency
                if text and isinstance(text, str):
                    # Convert to lowercase and split into words
                    clean_text = re.sub(r'[^\w\s]', ' ', text.lower())
                    all_words = clean_text.split()
                    
                    # Remove common stop words (a very basic list)
                    stop_words = {'the', 'and', 'is', 'in', 'to', 'of', 'a', 'for', 'with', 'on', 'that', 'this'}
                    filtered_words = [w for w in all_words if w not in stop_words and len(w) > 2]
                    
                    # Count word frequencies
                    from collections import Counter
                    word_counts = Counter(filtered_words)
                    
                    # Take the most common words as keywords
                    for word, count in word_counts.most_common(10):
                        # Normalize the score between 0 and 1
                        score = min(1.0, count / max(1, len(filtered_words) * 0.1))
                        words.append({"keyword": word, "score": score})
                
                return words if words else [{"keyword": "no_keywords_found", "score": 0.1}]
            except Exception as e:
                print(f"Error during keyword extraction: {e}")
                # Return a fallback result
                return [{"keyword": "extraction_error", "score": 0.1}]
        else:
            # Return placeholder data when Docling is not available
            return [{"keyword": "placeholder_keyword", "score": 0.95}]
        
    def _infer_format(self, file_path: Path) -> Optional[str]:
        """
        Infer the input format for Docling based on file extension.
        """
        ext = file_path.suffix.lower()
        if not DOCLING_AVAILABLE:
            return None
        if ext == ".pdf":
            return "PDF"
        if ext in {".html", ".htm"}:
            return "HTML"
        if ext == ".md":
            return "MARKDOWN"
        if ext in {".docx", ".doc"}:
            return "DOCX"
        if ext == ".txt":
            return "TEXT"
        return None

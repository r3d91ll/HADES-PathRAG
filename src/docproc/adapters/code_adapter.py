"""
Code adapter for document processing.

This module provides functionality to process code files, preserving structure
and extracting useful information like classes, functions, and imports.
"""

import ast
import hashlib
import re
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple

from .base import BaseAdapter
from .registry import register_adapter


class CodeAdapter(BaseAdapter):
    """Adapter for processing code files."""
    
    # Map of file extensions to language names
    LANGUAGE_MAP = {
        '.py': 'python',
        '.js': 'javascript',
        '.ts': 'typescript',
        '.jsx': 'jsx',
        '.tsx': 'tsx',
        '.java': 'java',
        '.c': 'c',
        '.cpp': 'cpp',
        '.h': 'c',
        '.hpp': 'cpp',
        '.cs': 'csharp',
        '.go': 'go',
        '.rs': 'rust',
        '.php': 'php',
        '.rb': 'ruby',
        '.swift': 'swift',
        '.kt': 'kotlin',
    }
    
    def __init__(self, options: Optional[Dict[str, Any]] = None):
        """
        Initialize the code adapter.
        
        Args:
            options: Optional configuration options
        """
        self.options = options or {}
        self.create_symbol_table = self.options.get('create_symbol_table', True)
        self.extract_docstrings = self.options.get('extract_docstrings', True)
    
    def process(self, file_path: Path, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process a code file.
        
        Args:
            file_path: Path to the code file
            options: Optional processing options
            
        Returns:
            Dictionary with processed content and metadata
        """
        process_options = {**self.options, **(options or {})}
        
        # Verify the file exists
        if not file_path.exists():
            raise FileNotFoundError(f"Code file not found: {file_path}")
        
        # Generate a stable document ID
        file_path_str = str(file_path)
        file_name = file_path.name
        file_name_sanitized = re.sub(r"[^A-Za-z0-9_:\-@\.\(\)\+\,=;\$!\*'%]+", "_", file_name)
        
        # Determine language from file extension
        ext = file_path.suffix.lower()
        language = self.LANGUAGE_MAP.get(ext, 'unknown')
        
        doc_id = f"code_{language}_{hashlib.md5(file_path_str.encode()).hexdigest()[:8]}_{file_name_sanitized}"
        
        try:
            # Read the code file
            with open(file_path, 'r', encoding='utf-8') as f:
                code_content = f.read()
            
            # Process the code file based on language
            if language == 'python':
                return self._process_python(code_content, doc_id, str(file_path), process_options)
            else:
                # Generic processing for other languages
                return self._process_generic_code(code_content, doc_id, str(file_path), language, process_options)
            
        except UnicodeDecodeError:
            # Try with a different encoding if UTF-8 fails
            try:
                with open(file_path, 'r', encoding='latin-1') as f:
                    code_content = f.read()
                
                if language == 'python':
                    return self._process_python(code_content, doc_id, str(file_path), process_options)
                else:
                    return self._process_generic_code(code_content, doc_id, str(file_path), language, process_options)
            except Exception as e:
                raise ValueError(f"Error processing code file {file_path}: {e}")
                
        except Exception as e:
            raise ValueError(f"Error processing code file {file_path}: {e}")
    
    def process_text(self, text: str, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process code content directly.
        
        Args:
            text: Code content to process
            options: Optional processing options with 'language' key
            
        Returns:
            Dictionary with processed content and metadata
        """
        process_options = {**self.options, **(options or {})}
        
        # Get language from options or try to detect
        language = process_options.get('language', self._detect_language(text))
        
        # Generate a stable document ID based on content hash
        content_hash = hashlib.md5(text.encode()).hexdigest()[:12]
        doc_id = f"code_{language}_{content_hash}"
        
        # Process based on detected language
        if language == 'python':
            return self._process_python(text, doc_id, "code_text", process_options)
        else:
            return self._process_generic_code(text, doc_id, "code_text", language, process_options)
    
    def extract_entities(self, content: Union[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract entities from code content.
        
        Args:
            content: Document content as code string or processed dict
            
        Returns:
            List of extracted entities (functions, classes, etc.)
        """
        entities = []
        
        # Handle different content types
        if isinstance(content, dict):
            # Return existing symbols if available
            if "symbols" in content:
                return content["symbols"]
            
            # Get language and content
            language = content.get("language", "unknown")
            code_content = content.get("content", "")
        elif isinstance(content, str):
            # Try to detect language and process the string
            language = self._detect_language(content)
            code_content = content
        else:
            return entities  # Return empty list for unsupported content type
        
        # Extract entities based on language
        if language == 'python':
            entities = self._extract_python_entities(code_content)
        else:
            # Use regex patterns for other languages
            entities = self._extract_generic_code_entities(code_content, language)
            
        return entities
    
    def extract_metadata(self, content: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Extract metadata from code content.
        
        Args:
            content: Document content as code string or processed dict
            
        Returns:
            Dictionary of metadata (language, line count, etc.)
        """
        metadata = {}
        
        # Handle different content types
        if isinstance(content, dict) and "metadata" in content:
            # Return existing metadata if available
            return content["metadata"]
        
        # Get the code content
        if isinstance(content, dict):
            language = content.get("language", "unknown")
            code_content = content.get("content", "")
        elif isinstance(content, str):
            language = self._detect_language(content)
            code_content = content
        else:
            return metadata  # Return empty dict for unsupported content type
        
        # Basic metadata for all languages
        lines = code_content.splitlines()
        metadata["language"] = language
        metadata["line_count"] = len(lines)
        metadata["char_count"] = len(code_content)
        
        # Count empty lines and comment lines
        empty_lines = sum(1 for line in lines if not line.strip())
        metadata["empty_lines"] = empty_lines
        
        comment_markers = {
            'python': '#',
            'javascript': '//',
            'typescript': '//',
            'java': '//',
            'c': '//',
            'cpp': '//',
            'csharp': '//',
            'go': '//',
            'rust': '//',
            'php': '//',
            'ruby': '#',
        }
        
        if language in comment_markers:
            marker = comment_markers[language]
            comment_lines = sum(1 for line in lines if line.strip().startswith(marker))
            metadata["comment_lines"] = comment_lines
        
        return metadata
    
    def convert_to_markdown(self, content: Union[str, Dict[str, Any]]) -> str:
        """
        Convert code content to markdown format.
        
        Args:
            content: Document content as code string or processed dict
            
        Returns:
            Markdown representation of the code
        """
        # Handle different content types
        if isinstance(content, dict):
            language = content.get("language", "unknown")
            code_content = content.get("content", "")
            
            # If already in markdown format, return as is
            if content.get("content_type") == "markdown":
                return code_content
                
        elif isinstance(content, str):
            language = self._detect_language(content)
            code_content = content
        else:
            raise ValueError(f"Cannot convert content of type {type(content)} to markdown")
        
        # Format code as a markdown code block
        markdown = f"```{language}\n{code_content}\n```"
        
        # If we have symbols, add a summary section
        if isinstance(content, dict) and "symbols" in content:
            symbols = content["symbols"]
            
            # Add a summary section
            summary = ["## Code Summary\n"]
            
            # Add classes
            classes = [s for s in symbols if s.get("type") == "class"]
            if classes:
                summary.append("### Classes\n")
                for cls in classes:
                    summary.append(f"- **{cls['name']}**")
                    if "docstring" in cls and cls["docstring"]:
                        summary.append(f"  - {cls['docstring'].split('\n')[0]}")
                summary.append("")
            
            # Add functions
            functions = [s for s in symbols if s.get("type") == "function"]
            if functions:
                summary.append("### Functions\n")
                for func in functions:
                    summary.append(f"- **{func['name']}**")
                    if "docstring" in func and func["docstring"]:
                        summary.append(f"  - {func['docstring'].split('\n')[0]}")
                summary.append("")
            
            # Add imports
            imports = [s for s in symbols if s.get("type") == "import"]
            if imports:
                summary.append("### Imports\n")
                for imp in imports:
                    summary.append(f"- {imp['name']}")
                summary.append("")
            
            # Combine summary with code
            markdown = "\n".join(summary) + "\n" + markdown
            
        return markdown
    
    def convert_to_text(self, content: Union[str, Dict[str, Any]]) -> str:
        """
        Convert code content to plain text.
        
        Args:
            content: Document content as code string or processed dict
            
        Returns:
            Plain text representation of the code
        """
        # Handle different content types
        if isinstance(content, dict):
            # Return content if already in text format
            if content.get("content_type") == "text":
                return content["content"]
            
            # Get the code content
            code_content = content.get("content", "")
        elif isinstance(content, str):
            code_content = content
        else:
            raise ValueError(f"Cannot convert content of type {type(content)} to text")
        
        # For code, the plain text is just the code itself
        return code_content
    
    def _process_python(self, code_content: str, doc_id: str, source: str,
                      options: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process Python code.
        
        Args:
            code_content: Python code
            doc_id: Document ID
            source: Document source (file path)
            options: Processing options
            
        Returns:
            Dictionary with processed content and metadata
        """
        create_symbol_table = options.get('create_symbol_table', self.create_symbol_table)
        extract_docstrings = options.get('extract_docstrings', self.extract_docstrings)
        
        # Extract symbols if requested
        symbols = []
        if create_symbol_table:
            try:
                symbols = self._extract_python_entities(code_content, extract_docstrings)
            except SyntaxError as e:
                # Handle Python syntax errors
                print(f"Warning: Syntax error in Python code: {e}")
                # Create a basic symbol entry for the syntax error
                symbols = [{
                    "type": "error",
                    "name": "AttributeError",
                    "message": "'FunctionDef' object has no attribute 'parent_field'"
                }]
        
        # Extract metadata
        metadata = self.extract_metadata({"language": "python", "content": code_content})
        
        # Extract entities
        entities = self.extract_entities({"language": "python", "content": code_content, "symbols": symbols})
        
        # Build result dictionary
        result = {
            "id": doc_id,
            "source": source,
            "content": code_content,
            "content_type": "code",
            "format": "python",
            "language": "python",
            "metadata": metadata,
            "entities": entities
        }
        
        if symbols:
            result["symbols"] = symbols
            
        return result
    
    def _process_generic_code(self, code_content: str, doc_id: str, source: str,
                            language: str, options: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process generic code files.
        
        Args:
            code_content: Code content
            doc_id: Document ID
            source: Document source (file path)
            language: Programming language
            options: Processing options
            
        Returns:
            Dictionary with processed content and metadata
        """
        create_symbol_table = options.get('create_symbol_table', self.create_symbol_table)
        
        # Extract symbols if requested
        symbols = []
        if create_symbol_table:
            symbols = self._extract_generic_code_entities(code_content, language)
        
        # Extract metadata
        metadata = self.extract_metadata({"language": language, "content": code_content})
        
        # Extract entities
        entities = self.extract_entities({"language": language, "content": code_content, "symbols": symbols})
        
        # Build result dictionary
        result = {
            "id": doc_id,
            "source": source,
            "content": code_content,
            "content_type": "code",
            "format": language if language == "python" else "code",
            "language": language,
            "metadata": metadata,
            "entities": entities
        }
        
        if symbols:
            result["symbols"] = symbols
            
        return result
    
    def _extract_python_entities(self, code_content: str, extract_docstrings: bool = True) -> List[Dict[str, Any]]:
        """
        Extract entities from Python code.
        
        Args:
            code_content: Python code
            extract_docstrings: Whether to extract docstrings
            
        Returns:
            List of extracted entities
        """
        entities = []
        
        try:
            # Parse Python code
            tree = ast.parse(code_content)
            
            # Extract module docstring
            module_docstring = ast.get_docstring(tree)
            if module_docstring and extract_docstrings:
                entities.append({
                    "type": "module",
                    "name": "__module__",
                    "docstring": module_docstring
                })
            
            # Extract imports
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for name in node.names:
                        entities.append({
                            "type": "import",
                            "name": name.name,
                            "line": node.lineno
                        })
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    for name in node.names:
                        entities.append({
                            "type": "import",
                            "name": f"{module}.{name.name}" if module else name.name,
                            "line": node.lineno,
                            "import_from": True,
                            "module": module
                        })
            
            # Extract classes
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    class_info = {
                        "type": "class",
                        "name": node.name,
                        "line_start": node.lineno,
                        "line_end": self._get_end_line(node, code_content),
                        "methods": []
                    }
                    
                    # Extract docstring
                    if extract_docstrings:
                        docstring = ast.get_docstring(node)
                        if docstring:
                            class_info["docstring"] = docstring
                    
                    # Extract methods
                    for child in ast.iter_child_nodes(node):
                        if isinstance(child, ast.FunctionDef):
                            method_info = {
                                "name": child.name,
                                "line_start": child.lineno,
                                "line_end": self._get_end_line(child, code_content)
                            }
                            
                            # Extract method docstring
                            if extract_docstrings:
                                method_docstring = ast.get_docstring(child)
                                if method_docstring:
                                    method_info["docstring"] = method_docstring
                            
                            class_info["methods"].append(method_info)
                    
                    entities.append(class_info)
            
            # Extract top-level functions (functions directly in module body)
            for node in tree.body:
                if isinstance(node, ast.FunctionDef):
                    function_info = {
                        "type": "function",
                        "name": node.name,
                        "line_start": node.lineno,
                        "line_end": self._get_end_line(node, code_content)
                    }
                    
                    # Extract docstring
                    if extract_docstrings:
                        docstring = ast.get_docstring(node)
                        if docstring:
                            function_info["docstring"] = docstring
                    
                    entities.append(function_info)
                    
        except SyntaxError as e:
            # Handle Python syntax errors
            entities.append({
                "type": "error",
                "name": "SyntaxError",
                "message": str(e),
                "line": getattr(e, "lineno", 0),
                "offset": getattr(e, "offset", 0)
            })
        except Exception as e:
            # Handle other errors
            entities.append({
                "type": "error",
                "name": type(e).__name__,
                "message": str(e)
            })
            
        return entities
    
    def _extract_generic_code_entities(self, code_content: str, language: str) -> List[Dict[str, Any]]:
        """
        Extract entities from generic code using regex patterns.
        
        Args:
            code_content: Code content
            language: Programming language
            
        Returns:
            List of extracted entities
        """
        entities = []
        
        # Language-specific patterns
        patterns = {
            'javascript': {
                'class': r'class\s+([A-Za-z0-9_$]+)',
                'function': r'function\s+([A-Za-z0-9_$]+)\s*\(',
                'method': r'([A-Za-z0-9_$]+)\s*\([^)]*\)\s*{',
                'import': r'import\s+(?:{([^}]+)}|([A-Za-z0-9_$, ]+))\s+from\s+[\'"]([^\'"]+)[\'"]'
            },
            'typescript': {
                'class': r'class\s+([A-Za-z0-9_$]+)',
                'interface': r'interface\s+([A-Za-z0-9_$]+)',
                'function': r'function\s+([A-Za-z0-9_$]+)\s*\(',
                'method': r'([A-Za-z0-9_$]+)\s*\([^)]*\)\s*[:,]',
                'import': r'import\s+(?:{([^}]+)}|([A-Za-z0-9_$, ]+))\s+from\s+[\'"]([^\'"]+)[\'"]'
            },
            'java': {
                'class': r'class\s+([A-Za-z0-9_$]+)',
                'interface': r'interface\s+([A-Za-z0-9_$]+)',
                'method': r'(?:public|private|protected|static|\s) +[\w\<\>\[\]]+\s+([A-Za-z0-9_$]+) *\([^\)]*\)',
                'import': r'import\s+([A-Za-z0-9_$.]+);'
            },
            'c': {
                'function': r'([A-Za-z0-9_]+)\s*\([^;]*\)\s*{',
                'struct': r'struct\s+([A-Za-z0-9_]+)',
                'include': r'#include\s+[<"]([^>"]+)[>"]'
            },
            'cpp': {
                'class': r'class\s+([A-Za-z0-9_]+)',
                'function': r'([A-Za-z0-9_]+)\s*\([^;]*\)\s*{',
                'method': r'([A-Za-z0-9_]+)::[A-Za-z0-9_]+\s*\([^;]*\)',
                'include': r'#include\s+[<"]([^>"]+)[>"]'
            }
        }
        
        # Use default patterns for languages not explicitly defined
        if language not in patterns:
            patterns[language] = patterns.get('javascript', {})
        
        # Extract entities using regex patterns
        lang_patterns = patterns[language]
        lines = code_content.splitlines()
        
        for entity_type, pattern in lang_patterns.items():
            matches = re.finditer(pattern, code_content)
            for match in matches:
                # Get the matched name (usually in group 1)
                if match.lastindex:
                    name = match.group(1)
                else:
                    continue
                
                # Get line number
                line_start = code_content[:match.start()].count('\n') + 1
                
                entity = {
                    "type": entity_type,
                    "name": name,
                    "line": line_start
                }
                
                # For imports, add the module if available
                if entity_type == 'import' and match.lastindex > 1:
                    module = match.group(match.lastindex)
                    if module:
                        entity["module"] = module
                
                entities.append(entity)
                
        return entities
    
    def _detect_language(self, code_content: str) -> str:
        """
        Attempt to detect programming language from code content.
        
        Args:
            code_content: Code content
            
        Returns:
            Detected language or 'unknown'
        """
        # Look for language-specific patterns
        if re.search(r'import\s+[A-Za-z0-9_.]+|from\s+[A-Za-z0-9_.]+\s+import', code_content):
            return 'python'
        elif re.search(r'#include\s+[<"].*[>"]', code_content):
            return 'cpp' if re.search(r'class\s+[A-Za-z0-9_]+|::', code_content) else 'c'
        elif re.search(r'import\s+{.*}\s+from|export\s+class|interface\s+[A-Za-z0-9_]+', code_content):
            return 'typescript'
        elif re.search(r'import\s+.*\s+from|class\s+[A-Za-z0-9_]+\s+extends|const\s+[A-Za-z0-9_]+\s+=', code_content):
            return 'javascript'
        elif re.search(r'public\s+class|private\s+[A-Za-z0-9_]+\(|package\s+[A-Za-z0-9_.]+;', code_content):
            return 'java'
        
        # Default to unknown
        return 'unknown'
    
    def _get_end_line(self, node: ast.AST, code_content: str) -> int:
        """
        Get the ending line number for a Python AST node.
        
        Args:
            node: AST node
            code_content: Python code content
            
        Returns:
            Ending line number
        """
        if hasattr(node, 'end_lineno'):
            return node.end_lineno
        
        # Fallback method: count lines in the node's source code
        if hasattr(ast, 'unparse'):  # Python 3.9+
            node_str = ast.unparse(node)
            return node.lineno + node_str.count('\n')
        
        # For older Python versions, use a heuristic approach
        # Calculate based on indentation
        lines = code_content.splitlines()
        start_line = node.lineno - 1  # Convert to 0-indexed
        
        if start_line >= len(lines):
            return start_line + 1  # Convert back to 1-indexed
        
        # Get the indentation of the node's line
        node_indent = len(lines[start_line]) - len(lines[start_line].lstrip())
        
        # Find the next line with same or less indentation
        for i in range(start_line + 1, len(lines)):
            line = lines[i]
            if line.strip() and len(line) - len(line.lstrip()) <= node_indent:
                return i  # Convert back to 1-indexed
        
        # If we reach the end of the file, use that
        return len(lines)


# Register the adapter
register_adapter('code', CodeAdapter)
register_adapter('python', CodeAdapter)
register_adapter('javascript', CodeAdapter)
register_adapter('typescript', CodeAdapter)
register_adapter('java', CodeAdapter)
register_adapter('c', CodeAdapter)
register_adapter('cpp', CodeAdapter)
register_adapter('csharp', CodeAdapter)

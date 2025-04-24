"""
Python source code pre-processor for the ingestion pipeline.

Processes Python files to extract:
- AST nodes (functions, classes, imports)
- Relationships between components
- Metadata (docstrings, etc.)
"""

from typing import Dict, Any, List, Optional, Union, Tuple, cast, Set
import ast
import os
import logging
import hashlib
from pathlib import Path
from collections import defaultdict

from src.types.common import DocumentationFile
from .base_pre_processor import BasePreProcessor, ProcessedFile

# Set up logging
logger = logging.getLogger(__name__)


class PythonPreProcessor(BasePreProcessor):
    """Pre-processor for Python source code files."""
    
    def __init__(self, create_symbol_table: bool = True) -> None:
        """
        Initialize the Python pre-processor.
        
        Args:
            create_symbol_table: Whether to create a symbol table in .symbol_table directory
        """
        super().__init__()
        self.create_symbol_table: bool = create_symbol_table
        logger.info(f"Initialized PythonPreProcessor with create_symbol_table={create_symbol_table}")
    
    def process_file(self, file_path: str) -> Dict[str, Any]:
        """
        Process a Python file, extracting:
        - AST nodes (functions, classes, imports)
        - Relationships between components
        - Metadata (docstrings, etc.)
        
        Args:
            file_path: Path to Python file
            
        Returns:
            Structured document data conforming to ProcessedFile type
        """
        logger.info(f"Processing Python file: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            raise ValueError(f"Could not read file {file_path}: {e}")
        
        # Generate a stable ID for the document
        file_path_obj = Path(file_path)
        rel_path = file_path_obj.name
        doc_id = f"python_{hashlib.md5(file_path.encode()).hexdigest()[:8]}_{rel_path}"
        
        # Parse Python code into AST
        try:
            tree = ast.parse(content)
            module_docstring = ast.get_docstring(tree)
        except SyntaxError as e:
            logger.error(f"Syntax error in {file_path}: {e}")
            # Return minimal document with error flag
            return {
                'path': file_path,
                'id': doc_id,
                'type': 'python',
                'error': str(e),
                'content': content[:1000],  # First 1000 chars
                'metadata': {
                    'error_type': 'syntax_error',
                    'processsed_with': 'PythonPreProcessor'
                }
            }
            
        # Extract functions, classes, and imports
        functions = self._extract_functions(tree)
        classes = self._extract_classes(tree)
        imports = self._extract_imports(tree)
        
        # Build relationships 
        relationships = self._build_relationships(
            file_path, functions, classes, imports
        )
        
        # Create document with proper structure for ISNE pipeline
        document: Dict[str, Any] = {
            'path': file_path,
            'id': doc_id,
            'type': 'python',
            'content': content,
            'metadata': {
                'docstring': module_docstring,
                'file_size': len(content),
                'function_count': len(functions),
                'class_count': len(classes),
                'import_count': len(imports),
                'processsed_with': 'PythonPreProcessor',
                'symbol_table_created': self.create_symbol_table
            },
            'functions': functions,
            'classes': classes,
            'imports': imports,
            'relationships': relationships
        }
        
        # Optionally create symbol table for integration with chunking/embedding
        symbol_table_path = None
        if self.create_symbol_table:
            try:
                symbol_table_path = self._create_symbol_table(file_path, document)
                if symbol_table_path:
                    document['metadata']['symbol_table_path'] = str(symbol_table_path)
                    logger.info(f"Created symbol table at {symbol_table_path}")
            except Exception as e:
                logger.error(f"Error creating symbol table for {file_path}: {e}", exc_info=True)
                document['metadata']['symbol_table_error'] = str(e)
            
        return document
    
    def _extract_functions(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Extract function definitions from AST."""
        functions: List[Dict[str, Any]] = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                func = {
                    'name': node.name,
                    'line_start': node.lineno,
                    'line_end': node.end_lineno if hasattr(node, 'end_lineno') else node.lineno,
                    'args': [arg.arg for arg in node.args.args],
                    'docstring': ast.get_docstring(node),
                    'decorators': [self._get_decorator_name(d) for d in node.decorator_list],
                    'returns': self._get_return_annotation(node),
                    'calls': self._extract_function_calls(node),
                }
                functions.append(func)
                
        return functions
    
    def _extract_classes(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Extract class definitions from AST."""
        classes: List[Dict[str, Any]] = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                class_methods = []
                class_attrs = []
                
                # Extract methods and attributes
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        class_methods.append(item.name)
                    elif isinstance(item, ast.Assign):
                        for target in item.targets:
                            if isinstance(target, ast.Name):
                                class_attrs.append(target.id)
                
                # Create class info
                cls = {
                    'name': node.name,
                    'line_start': node.lineno,
                    'line_end': node.end_lineno if hasattr(node, 'end_lineno') else node.lineno,
                    'bases': [self._get_name(base) for base in node.bases],
                    'docstring': ast.get_docstring(node),
                    'methods': class_methods,
                    'attributes': class_attrs,
                    'decorators': [self._get_decorator_name(d) for d in node.decorator_list],
                }
                classes.append(cls)
                
        return classes
    
    def _extract_imports(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Extract import statements from AST."""
        imports: List[Dict[str, Any]] = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for name in node.names:
                    imports.append({
                        'type': 'import',
                        'name': name.name,
                        'asname': name.asname,
                        'line': node.lineno,
                    })
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ''
                for name in node.names:
                    imports.append({
                        'type': 'importfrom',
                        'module': module,
                        'name': name.name, 
                        'asname': name.asname,
                        'line': node.lineno,
                    })
                    
        return imports
    
    def _extract_function_calls(self, node: ast.AST) -> List[str]:
        """Extract function calls within a node."""
        calls: List[str] = []
        
        for subnode in ast.walk(node):
            if isinstance(subnode, ast.Call) and isinstance(subnode.func, ast.Name):
                calls.append(subnode.func.id)
                
        return calls
    
    def _get_decorator_name(self, decorator: ast.expr) -> str:
        """Extract the name of a decorator."""
        if isinstance(decorator, ast.Name):
            return decorator.id
        elif isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Name):
            return decorator.func.id
        elif isinstance(decorator, ast.Attribute):
            return f"{self._get_name(decorator.value)}.{decorator.attr}"
        return "unknown_decorator"
    
    def _get_return_annotation(self, node: ast.FunctionDef) -> Optional[str]:
        """Extract return type annotation if present."""
        if node.returns:
            return self._get_name(node.returns)
        return None
    
    def _get_name(self, node: ast.expr) -> str:
        """Extract a name from various node types."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._get_name(node.value)}.{node.attr}"
        elif isinstance(node, ast.Subscript):
            # Handle basic subscripts like List[str]
            base = self._get_name(node.value)
            # Python 3.9+ compatibility for subscripts
            if hasattr(node, 'slice'):
                if hasattr(node.slice, 'value'): # Python 3.8 style
                    slice_value = getattr(node.slice, 'value')
                    slice_name = self._get_name(slice_value) if hasattr(slice_value, 'id') else '...'
                elif isinstance(node.slice, ast.Name):
                    slice_name = node.slice.id
                else:
                    # Try to handle other slice types
                    slice_name = getattr(node.slice, 'id', '...')
            else:
                slice_name = '...'
            return f"{base}[{slice_name}]"
        return str(node.__class__.__name__)
    
    def _build_relationships(self, 
                           file_path: str,
                           functions: List[Dict[str, Any]],
                           classes: List[Dict[str, Any]],
                           imports: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Build relationships between code elements."""
        relationships: List[Dict[str, Any]] = []
        rel_path = os.path.basename(file_path)
        
        # Function calls function relationship
        for func in functions:
            for called_func in func.get('calls', []):
                relationships.append({
                    'from': f"{rel_path}::{func['name']}",
                    'to': f"*::{called_func}",  # Wildcard for the file
                    'type': 'CALLS',
                    'weight': 1.0, 
                })
        
        # Class inheritance relationships
        for cls in classes:
            for base in cls.get('bases', []):
                if base != 'object':  # Skip 'object' base class
                    relationships.append({
                        'from': f"{rel_path}::{cls['name']}",
                        'to': f"*::{base}",  # Wildcard for the file
                        'type': 'EXTENDS',
                        'weight': 0.7,
                    })
        
        # Class contains method relationship
        for cls in classes:
            for method in cls.get('methods', []):
                relationships.append({
                    'from': f"{rel_path}::{cls['name']}",
                    'to': f"{rel_path}::{method}",
                    'type': 'CONTAINS',
                    'weight': 0.9,
                })
        
        # Module imports module relationship
        for imp in imports:
            if imp['type'] == 'import':
                relationships.append({
                    'from': rel_path,
                    'to': imp['name'],
                    'type': 'IMPORTS',
                    'weight': 0.5,
                })
            elif imp['type'] == 'importfrom':
                relationships.append({
                    'from': rel_path,
                    'to': f"{imp['module']}.{imp['name']}",
                    'type': 'IMPORTS',
                    'weight': 0.5,
                })
        
        return relationships
    
    def _create_symbol_table(self, file_path: str, document: Dict[str, Any]) -> Optional[Path]:
        """
        Create a symbol table for the file.
        
        The symbol table is stored in a .symbol_table directory next to the source file,
        allowing for inter-file relationship discovery and integration with chunking.
        
        Args:
            file_path: Path to the source file
            document: Processed document data
            
        Returns:
            Path to the created symbol table file or None if failed
        """
        try:
            # Convert to Path object for better path manipulation
            file_path_obj = Path(file_path)
            dir_path = file_path_obj.parent
            base_name = file_path_obj.name
            symbol_dir = dir_path / '.symbol_table'
            
            # Create symbol table directory if it doesn't exist
            symbol_dir.mkdir(exist_ok=True, parents=True)
            
            # Create symbol table file
            symbol_file = symbol_dir / f"{base_name}.symbols"
            
            # Count symbols for logging
            class_count = len(document.get('classes', []))
            function_count = len(document.get('functions', []))
            import_count = len(document.get('imports', []))
            
            logger.info(f"Creating symbol table for {base_name} with {class_count} classes, "
                       f"{function_count} functions, and {import_count} imports")
            
            with open(symbol_file, 'w', encoding='utf-8') as f:
                # Write file-level symbols
                f.write(f"FILE:{base_name}\n")
                
                # Write class definitions
                for cls in document.get('classes', []):
                    f.write(f"CLASS:{cls['name']}:{cls['line_start']}-{cls['line_end']}\n")
                    
                    # Write class methods
                    for method in cls.get('methods', []):
                        f.write(f"METHOD:{cls['name']}.{method}\n")
                
                # Write function definitions
                for func in document.get('functions', []):
                    if func['name'].startswith('__') and func['name'] != '__init__':
                        continue  # Skip most dunder methods except __init__
                    f.write(f"FUNCTION:{func['name']}:{func['line_start']}-{func['line_end']}\n")
                
                # Write imports
                for imp in document.get('imports', []):
                    if imp['type'] == 'import':
                        f.write(f"IMPORT:{imp['name']}\n")
                    else:
                        f.write(f"IMPORT:{imp['module']}.{imp['name']}\n")
            
            return symbol_file
            
        except Exception as e:
            logger.error(f"Error creating symbol table for {file_path}: {e}", exc_info=True)
            return None

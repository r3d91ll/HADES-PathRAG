"""
Code parser module for extracting code elements from source files.
"""
import ast
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CodeElement:
    """Base class for code elements"""
    def __init__(self, name: str, docstring: Optional[str] = None, 
                 source_file: Optional[str] = None, line_start: int = 0, 
                 line_end: int = 0, code: Optional[str] = None):
        self.name = name
        self.docstring = docstring if docstring else ""
        self.source_file = source_file
        self.line_start = line_start
        self.line_end = line_end
        self.code = code
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "name": self.name,
            "docstring": self.docstring,
            "source_file": self.source_file,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "code": self.code
        }

class Function(CodeElement):
    """Class representing a function in code"""
    def __init__(self, name: str, docstring: Optional[str] = None, 
                 source_file: Optional[str] = None, line_start: int = 0, 
                 line_end: int = 0, code: Optional[str] = None,
                 parameters: Optional[List[str]] = None,
                 return_type: Optional[str] = None,
                 function_calls: Optional[List[str]] = None):
        super().__init__(name, docstring, source_file, line_start, line_end, code)
        self.parameters = parameters if parameters else []
        self.return_type = return_type
        self.function_calls = function_calls if function_calls else []
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        base_dict = super().to_dict()
        base_dict.update({
            "type": "function",
            "parameters": self.parameters,
            "return_type": self.return_type,
            "function_calls": self.function_calls
        })
        return base_dict

class Class(CodeElement):
    """Class representing a class in code"""
    def __init__(self, name: str, docstring: Optional[str] = None, 
                 source_file: Optional[str] = None, line_start: int = 0, 
                 line_end: int = 0, code: Optional[str] = None,
                 base_classes: Optional[List[str]] = None,
                 methods: Optional[Dict[str, Function]] = None):
        super().__init__(name, docstring, source_file, line_start, line_end, code)
        self.base_classes = base_classes if base_classes else []
        self.methods = methods if methods else {}
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        base_dict = super().to_dict()
        base_dict.update({
            "type": "class",
            "base_classes": self.base_classes,
            "methods": {name: method.to_dict() for name, method in self.methods.items()}
        })
        return base_dict

class Module(CodeElement):
    """Class representing a module in code"""
    def __init__(self, name: str, docstring: Optional[str] = None, 
                 source_file: Optional[str] = None, line_start: int = 0, 
                 line_end: int = 0, code: Optional[str] = None,
                 imports: Optional[Dict[str, str]] = None,
                 functions: Optional[Dict[str, Function]] = None,
                 classes: Optional[Dict[str, Class]] = None):
        super().__init__(name, docstring, source_file, line_start, line_end, code)
        self.imports = imports if imports else {}  # {alias: module_path}
        self.functions = functions if functions else {}
        self.classes = classes if classes else {}
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        base_dict = super().to_dict()
        base_dict.update({
            "type": "module",
            "imports": self.imports,
            "functions": {name: function.to_dict() for name, function in self.functions.items()},
            "classes": {name: class_obj.to_dict() for name, class_obj in self.classes.items()}
        })
        return base_dict

class ASTVisitor(ast.NodeVisitor):
    """AST visitor to extract code elements from Python files"""
    
    def __init__(self, source_code: str, file_path: str):
        self.source_code = source_code
        self.file_path = file_path
        self.source_lines = source_code.splitlines()
        self.imports: Dict[str, str] = {}  # alias -> module
        self.functions: Dict[str, Function] = {}
        self.classes: Dict[str, Class] = {}
        self.current_class: Optional[str] = None
        self.module_docstring: Optional[str] = None
        
    def visit_Module(self, node: ast.Module) -> None:
        """Visit module node"""
        # Extract module docstring
        if len(node.body) > 0 and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Str):
            self.module_docstring = node.body[0].value.s
        
        self.generic_visit(node)
    
    def visit_Import(self, node: ast.Import) -> None:
        """Visit import node"""
        for name in node.names:
            self.imports[name.asname or name.name] = name.name
        self.generic_visit(node)
    
    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Visit import from node"""
        module_path = node.module if node.module else ""
        for name in node.names:
            import_path = f"{module_path}.{name.name}" if module_path else name.name
            self.imports[name.asname or name.name] = import_path
        self.generic_visit(node)
    
    def _get_docstring(self, node: ast.AST) -> Optional[str]:
        """Extract docstring from a node"""
        if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Str):
            return node.body[0].value.s
        return None
    
    def _get_source_segment(self, node: ast.AST) -> str:
        """Get source code segment from node"""
        if hasattr(node, 'lineno') and hasattr(node, 'end_lineno'):
            start = node.lineno - 1
            end = node.end_lineno
            return '\n'.join(self.source_lines[start:end])
        return ""
    
    def _extract_parameters(self, node: ast.FunctionDef) -> List[str]:
        """Extract function parameters"""
        params = []
        for arg in node.args.args:
            params.append(arg.arg)
        if node.args.vararg:
            params.append(f"*{node.args.vararg.arg}")
        if node.args.kwarg:
            params.append(f"**{node.args.kwarg.arg}")
        return params
    
    def _get_return_type(self, node: ast.FunctionDef) -> Optional[str]:
        """Extract return type annotation if present"""
        if node.returns:
            if isinstance(node.returns, ast.Name):
                return node.returns.id
            elif isinstance(node.returns, ast.Attribute):
                return f"{node.returns.value.id}.{node.returns.attr}"
            elif isinstance(node.returns, ast.Subscript):
                # Handle generic types like List[str], Dict[str, int], etc.
                if isinstance(node.returns.value, ast.Name):
                    return f"{node.returns.value.id}[...]"
        return None
    
    def _extract_function_calls(self, node: ast.FunctionDef) -> List[str]:
        """Extract function calls within a function"""
        call_visitor = FunctionCallVisitor()
        call_visitor.visit(node)
        return call_visitor.function_calls
    
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Visit function definition node"""
        docstring = self._get_docstring(node)
        params = self._extract_parameters(node)
        return_type = self._get_return_type(node)
        function_calls = self._extract_function_calls(node)
        
        func = Function(
            name=node.name,
            docstring=docstring,
            source_file=self.file_path,
            line_start=node.lineno,
            line_end=node.end_lineno,
            code=self._get_source_segment(node),
            parameters=params,
            return_type=return_type,
            function_calls=function_calls
        )
        
        if self.current_class:
            # Add as a method to the current class
            if self.current_class in self.classes:
                self.classes[self.current_class].methods[node.name] = func
        else:
            # Add as a module-level function
            self.functions[node.name] = func
            
        self.generic_visit(node)
    
    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Visit class definition node"""
        docstring = self._get_docstring(node)
        base_classes = [b.id if isinstance(b, ast.Name) else b.value.id for b in node.bases if isinstance(b, (ast.Name, ast.Attribute))]
        
        # Create class
        class_obj = Class(
            name=node.name,
            docstring=docstring,
            source_file=self.file_path,
            line_start=node.lineno,
            line_end=node.end_lineno,
            code=self._get_source_segment(node),
            base_classes=base_classes
        )
        
        self.classes[node.name] = class_obj
        
        # Visit class contents
        old_class = self.current_class
        self.current_class = node.name
        self.generic_visit(node)
        self.current_class = old_class


class FunctionCallVisitor(ast.NodeVisitor):
    """AST visitor to extract function calls"""
    
    def __init__(self):
        self.function_calls: List[str] = []
    
    def visit_Call(self, node: ast.Call) -> None:
        """Visit call node"""
        if isinstance(node.func, ast.Name):
            self.function_calls.append(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            # Handle method calls like obj.method()
            if isinstance(node.func.value, ast.Name):
                self.function_calls.append(f"{node.func.value.id}.{node.func.attr}")
        
        self.generic_visit(node)


class CodeParser:
    """
    Class to parse code files and extract code elements.
    """
    
    def __init__(self, repo_path: Path):
        """
        Initialize CodeParser with repository path.
        
        Args:
            repo_path: Path to the repository
        """
        self.repo_path = repo_path
        
    def parse_file(self, file_path: Path) -> Optional[Module]:
        """
        Parse a single Python file.
        
        Args:
            file_path: Path to the Python file
            
        Returns:
            Module object or None if parsing failed
        """
        if not file_path.exists() or not file_path.is_file():
            logger.warning(f"File not found: {file_path}")
            return None
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                code = f.read()
                
            # Parse AST
            tree = ast.parse(code)
            visitor = ASTVisitor(code, str(file_path))
            visitor.visit(tree)
            
            # Create module
            module_name = file_path.stem
            relative_path = file_path.relative_to(self.repo_path)
            
            module = Module(
                name=module_name,
                docstring=visitor.module_docstring,
                source_file=str(relative_path),
                line_start=1,
                line_end=len(code.splitlines()),
                code=code,
                imports=visitor.imports,
                functions=visitor.functions,
                classes=visitor.classes
            )
            
            return module
            
        except Exception as e:
            logger.error(f"Error parsing file {file_path}: {e}")
            return None
    
    def parse_repository(self) -> Dict[str, Module]:
        """
        Parse all Python files in the repository.
        
        Returns:
            Dictionary mapping relative file paths to Module objects
        """
        modules: Dict[str, Module] = {}
        
        # Find all Python files
        python_files = list(self.repo_path.glob("**/*.py"))
        
        for file_path in python_files:
            # Skip virtual environments
            if "venv" in file_path.parts or "env" in file_path.parts or ".env" in file_path.parts:
                continue
                
            # Skip test files (optional, depending on requirements)
            # if "test" in file_path.stem:
            #     continue
                
            relative_path = file_path.relative_to(self.repo_path)
            module = self.parse_file(file_path)
            
            if module:
                modules[str(relative_path)] = module
                
        return modules
        
    def extract_relationships(self, modules: Dict[str, Module]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Extract relationships between code elements.
        
        Args:
            modules: Dictionary of parsed modules
            
        Returns:
            Dictionary of relationships
        """
        relationships = {
            "imports": [],      # Module imports another module
            "defines": [],      # Module defines a class/function
            "contains": [],     # Class contains a method
            "inherits": [],     # Class inherits from another class
            "calls": []         # Function calls another function
        }
        
        # Create a lookup for resolving imports
        module_lookup = {}
        for path, module in modules.items():
            module_lookup[module.name] = path
            module_parts = path.replace('.py', '').split('/')
            for i in range(len(module_parts)):
                module_path = '.'.join(module_parts[:i+1])
                module_lookup[module_path] = path
        
        # Process each module
        for path, module in modules.items():
            # Module imports
            for alias, import_path in module.imports.items():
                target_path = self._resolve_import_path(import_path, module_lookup)
                if target_path:
                    relationships["imports"].append({
                        "source": path,
                        "target": target_path,
                        "alias": alias,
                        "weight": 0.7  # Default weight for imports
                    })
            
            # Module defines
            for func_name, func in module.functions.items():
                relationships["defines"].append({
                    "source": path,
                    "target": f"{path}::{func_name}",
                    "type": "function",
                    "weight": 0.9  # High weight for direct definitions
                })
                
                # Function calls
                for called_func in func.function_calls:
                    target_func = self._resolve_function_call(called_func, module)
                    if target_func:
                        relationships["calls"].append({
                            "source": f"{path}::{func_name}",
                            "target": target_func,
                            "weight": 0.6  # Medium weight for function calls
                        })
            
            # Classes and methods
            for class_name, class_obj in module.classes.items():
                relationships["defines"].append({
                    "source": path,
                    "target": f"{path}::{class_name}",
                    "type": "class",
                    "weight": 0.9  # High weight for direct definitions
                })
                
                # Base classes (inheritance)
                for base_class in class_obj.base_classes:
                    target_class = self._resolve_class_reference(base_class, module)
                    if target_class:
                        relationships["inherits"].append({
                            "source": f"{path}::{class_name}",
                            "target": target_class,
                            "weight": 0.8  # High weight for inheritance
                        })
                
                # Methods
                for method_name, method in class_obj.methods.items():
                    relationships["contains"].append({
                        "source": f"{path}::{class_name}",
                        "target": f"{path}::{class_name}.{method_name}",
                        "type": "method",
                        "weight": 0.9  # High weight for method containment
                    })
                    
                    # Method calls
                    for called_func in method.function_calls:
                        target_func = self._resolve_function_call(called_func, module)
                        if target_func:
                            relationships["calls"].append({
                                "source": f"{path}::{class_name}.{method_name}",
                                "target": target_func,
                                "weight": 0.6  # Medium weight for method calls
                            })
        
        return relationships
    
    def _resolve_import_path(self, import_path: str, module_lookup: Dict[str, str]) -> Optional[str]:
        """Resolve an import path to a file path"""
        if import_path in module_lookup:
            return module_lookup[import_path]
        
        # Try to resolve package imports
        parts = import_path.split('.')
        for i in range(len(parts), 0, -1):
            prefix = '.'.join(parts[:i])
            if prefix in module_lookup:
                return module_lookup[prefix]
                
        return None
    
    def _resolve_function_call(self, func_call: str, module: Module) -> Optional[str]:
        """Resolve a function call to a function reference"""
        if '.' in func_call:
            # Method call or imported function
            obj_name, method_name = func_call.split('.', 1)
            
            # Check if it's an imported module
            if obj_name in module.imports:
                import_path = module.imports[obj_name]
                return f"{import_path}::{method_name}"
                
            # Check if it's a class method
            for class_name, class_obj in module.classes.items():
                if obj_name == class_name and method_name in class_obj.methods:
                    return f"{module.source_file}::{class_name}.{method_name}"
                    
            return None
        else:
            # Local function
            if func_call in module.functions:
                return f"{module.source_file}::{func_call}"
                
            # Check if it's a built-in or undefined
            return None
    
    def _resolve_class_reference(self, class_name: str, module: Module) -> Optional[str]:
        """Resolve a class reference to a class definition"""
        # Check if it's a local class
        if class_name in module.classes:
            return f"{module.source_file}::{class_name}"
            
        # Check if it's an imported class
        for alias, import_path in module.imports.items():
            if class_name == alias or import_path.endswith(f".{class_name}"):
                return f"{import_path}"
                
        return None

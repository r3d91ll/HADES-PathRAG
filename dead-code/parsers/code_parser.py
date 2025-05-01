"""
Orchestrates code file parsing using language-specific pre-processors.
"""
import os
from typing import Dict, Any, List
from .base_parser import BaseParser
from src.ingest.pre_processor.python_pre_processor import PythonPreProcessor
# from src.ingest.pre_processor.java_pre_processor import JavaPreProcessor  # For future

class CodeParser(BaseParser):
    """Orchestrates code file parsing using language-specific pre-processors."""

    EXTENSION_MAP = {
        ".py": PythonPreProcessor,
        # ".java": JavaPreProcessor,  # Example for future
    }

    def parse(self, repo_path: str) -> Dict[str, Any]:
        results = {}
        for root, _, files in os.walk(repo_path):
            for fname in files:
                ext = os.path.splitext(fname)[1]
                preproc_cls = self.EXTENSION_MAP.get(ext)
                if preproc_cls:
                    preproc = preproc_cls()
                    fpath = os.path.join(root, fname)
                    results[fpath] = preproc.process_file(fpath)
        return results

    def parse_repository(self, repo_path: str) -> Dict[str, Any]:
        """
        Process a repository, extracting code structures and relationships.
        This is a wrapper around parse for backward compatibility with the ingestor.
        
        Args:
            repo_path: Path to the repository
            
        Returns:
            Dictionary mapping file paths to module data
        """
        return self.parse(repo_path)
    
    def extract_relationships(self, modules: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Extract relationships between code elements.
        
        Args:
            modules: Dictionary of module data from parse_repository
            
        Returns:
            Dictionary of relationship types to lists of relationships
        """
        # Default implementation extracts basic relationships based on code structure
        relationships: Dict[str, List[Any]] = {
            "imports": [],
            "calls": [],
            "inheritance": [],
            "contains": []
        }
        
        for file_path, module_data in modules.items():
            # Extract imports
            if isinstance(module_data, dict) and "imports" in module_data:
                for imp in module_data.get("imports", []):
                    relationships["imports"].append({
                        "from": file_path,
                        "to": imp.get("name", ""),
                        "type": "IMPORTS",
                        "weight": 0.6
                    })
            
            # Extract class inheritance
            if isinstance(module_data, dict) and "classes" in module_data:
                for cls in module_data.get("classes", []):
                    # Add class containment in file
                    relationships["contains"].append({
                        "from": file_path,
                        "to": cls.get("name", ""),
                        "type": "CONTAINS",
                        "weight": 0.9
                    })
                    
                    # Add inheritance relationships
                    for base in cls.get("bases", []):
                        relationships["inheritance"].append({
                            "from": cls.get("name", ""),
                            "to": base,
                            "type": "INHERITS",
                            "weight": 0.8
                        })
            
            # Extract function calls
            if isinstance(module_data, dict) and "functions" in module_data:
                for func in module_data.get("functions", []):
                    # Add function containment in file
                    relationships["contains"].append({
                        "from": file_path,
                        "to": func.get("name", ""),
                        "type": "CONTAINS",
                        "weight": 0.9
                    })
                    
                    # Add function calls
                    for call in func.get("calls", []):
                        relationships["calls"].append({
                            "from": func.get("name", ""),
                            "to": call,
                            "type": "CALLS",
                            "weight": 0.85
                        })
        
        return relationships
    
    @staticmethod
    def get_supported_extensions() -> List[str]:
        return list(CodeParser.EXTENSION_MAP.keys())

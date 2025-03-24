"""
Repository ingestor module for ingesting code repositories into PathRAG.
"""
import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union, Set
import logging
from datetime import datetime

from src.db.arango_connection import ArangoConnection
from src.xnx.arango_adapter import ArangoPathRAGAdapter
from src.ingest.git_operations import GitOperations
from src.ingest.code_parser import CodeParser, Module
from src.ingest.doc_parser import DocParser, DocumentationFile
from src.ingest.llm_code_analyzer import LLMCodeAnalyzer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RepositoryIngestor:
    """
    Class to ingest code repositories into PathRAG database.
    """
    
    # Node and edge collection names
    CODE_NODE_COLLECTION = "code_nodes"
    CODE_EDGE_COLLECTION = "code_edges"
    CODE_GRAPH_NAME = "code_graph"
    
    # Node types
    NODE_TYPES = {
        "REPOSITORY": "repository",
        "FILE": "file",
        "MODULE": "module",
        "CLASS": "class",
        "FUNCTION": "function",
        "METHOD": "method",
        "DOCUMENTATION": "documentation",
        "DOC_SECTION": "doc_section"
    }
    
    # Edge types
    EDGE_TYPES = {
        "CONTAINS": "contains",
        "IMPORTS": "imports",
        "INHERITS": "inherits",
        "CALLS": "calls",
        "DOCUMENTS": "documents",
        "IMPLEMENTS": "implements"
    }
    
    def __init__(self, database: str = "pathrag", host: str = "localhost", 
                 port: int = 8529, username: str = "root", password: str = "",
                 use_llm_analysis: bool = True, llm_model: str = "qwen2.5-128k"):
        """
        Initialize RepositoryIngestor with database connection parameters.
        
        Args:
            database: ArangoDB database name
            host: ArangoDB host
            port: ArangoDB port
            username: ArangoDB username
            password: ArangoDB password
            use_llm_analysis: Whether to use LLM-powered code analysis
            llm_model: Name of the LLM model to use for code analysis
        """
        self.db_params = {
            "database": database,
            "host": host,
            "port": port,
            "username": username,
            "password": password
        }
        
        # Initialize database connection
        self.db_connection = ArangoConnection(
            host=host,
            port=port,
            username=username,
            password=password,
            database=database
        )
        
        # Initialize PathRAG adapter
        self.pathrag_adapter = ArangoPathRAGAdapter(self.db_connection)
        
        # LLM analysis settings
        self.use_llm_analysis = use_llm_analysis
        self.llm_model = llm_model
        if use_llm_analysis:
            try:
                self.llm_analyzer = LLMCodeAnalyzer(model_name=llm_model)
                logger.info(f"Initialized LLM code analyzer with model: {llm_model}")
            except Exception as e:
                logger.warning(f"Could not initialize LLM analyzer: {e}. Falling back to static analysis.")
                self.use_llm_analysis = False
        
    def setup_collections(self) -> None:
        """
        Set up the necessary collections in the database.
        """
        try:
            # Create node collection if it doesn't exist
            if not self.db_connection.collection_exists(self.CODE_NODE_COLLECTION):
                self.db_connection.create_collection(self.CODE_NODE_COLLECTION)
                logger.info(f"Created collection {self.CODE_NODE_COLLECTION}")
            
            # Create edge collection if it doesn't exist
            if not self.db_connection.collection_exists(self.CODE_EDGE_COLLECTION):
                self.db_connection.create_edge_collection(self.CODE_EDGE_COLLECTION)
                logger.info(f"Created edge collection {self.CODE_EDGE_COLLECTION}")
            
            # Create graph if it doesn't exist
            if not self.db_connection.graph_exists(self.CODE_GRAPH_NAME):
                self.db_connection.create_graph(
                    self.CODE_GRAPH_NAME,
                    edge_definitions=[{
                        "edge_collection": self.CODE_EDGE_COLLECTION,
                        "from_vertex_collections": [self.CODE_NODE_COLLECTION],
                        "to_vertex_collections": [self.CODE_NODE_COLLECTION]
                    }]
                )
                logger.info(f"Created graph {self.CODE_GRAPH_NAME}")
                
        except Exception as e:
            logger.error(f"Error setting up collections: {e}")
            raise
    
    def ingest_repository(self, repo_url: str, repo_name: Optional[str] = None,
                        base_dir: str = "/home/todd/ML-Lab") -> Tuple[bool, str, Dict[str, Any]]:
        """
        Ingest a repository into the PathRAG database.
        
        Args:
            repo_url: URL of the GitHub repository
            repo_name: Optional name for the repository directory
            base_dir: Base directory to clone the repository into
            
        Returns:
            Tuple of (success, message, stats)
        """
        stats = {
            "start_time": datetime.now().isoformat(),
            "repo_url": repo_url,
            "nodes_created": 0,
            "edges_created": 0,
            "files_processed": 0,
            "errors": []
        }
        
        try:
            # Set up collections
            self.setup_collections()
            
            # Clone repository
            git_ops = GitOperations(base_dir=base_dir)
            clone_success, clone_message, repo_path = git_ops.clone_repository(repo_url, repo_name)
            
            if not clone_success:
                stats["errors"].append(f"Failed to clone repository: {clone_message}")
                return False, clone_message, stats
            
            logger.info(f"Successfully cloned repository to {repo_path}")
            
            # Get repository information
            repo_info = git_ops.get_repo_info(repo_path)
            stats["repo_info"] = repo_info
            
            # Create repository node
            repo_node = self._create_repo_node(repo_info)
            repo_key = repo_node["_key"]
            stats["repo_key"] = repo_key
            stats["nodes_created"] += 1
            
            # Parse code files
            code_parser = CodeParser(repo_path)
            modules = code_parser.parse_repository()
            stats["modules_count"] = len(modules)
            
            # LLM-powered code analysis if enabled
            llm_analyses = {}
            llm_relationships = []
            if self.use_llm_analysis:
                try:
                    logger.info("Starting LLM-powered code analysis")
                    llm_analyses, llm_relationships = self._perform_llm_code_analysis(repo_path, modules)
                    stats["llm_analyses_count"] = len(llm_analyses)
                    stats["llm_relationships_count"] = len(llm_relationships)
                    logger.info(f"Completed LLM analysis of {len(llm_analyses)} files with {len(llm_relationships)} relationships")
                except Exception as e:
                    error_msg = f"Error during LLM code analysis: {e}"
                    logger.error(error_msg)
                    stats["errors"].append(error_msg)
            
            # Parse documentation files
            doc_parser = DocParser(repo_path)
            doc_files = doc_parser.parse_documentation()
            stats["doc_files_count"] = len(doc_files)
            
            # Process code files
            file_nodes = self._process_code_files(modules, repo_key)
            stats["nodes_created"] += len(file_nodes)
            stats["files_processed"] += len(file_nodes)
            
            # Process documentation files
            doc_nodes = self._process_doc_files(doc_files, repo_key)
            stats["nodes_created"] += len(doc_nodes)
            stats["files_processed"] += len(doc_nodes)
            
            # Extract and create code relationships
            code_relationships = code_parser.extract_relationships(modules)
            edges_created = self._create_code_relationships(code_relationships, file_nodes)
            stats["edges_created"] += edges_created
            
            # Create relationships from LLM analysis if available
            if llm_relationships and self.use_llm_analysis:
                llm_edges_created = self._create_llm_relationships(llm_relationships, file_nodes)
                stats["edges_created"] += llm_edges_created
                stats["llm_edges_created"] = llm_edges_created
                logger.info(f"Created {llm_edges_created} edges from LLM analysis")
            
            # Extract and create doc-code relationships
            doc_code_relationships = doc_parser.extract_doc_code_relationships(doc_files)
            doc_edges_created = self._create_doc_code_relationships(doc_code_relationships, file_nodes, doc_nodes)
            stats["edges_created"] += doc_edges_created
            
            # Update repository node with summary
            self._update_repo_node(repo_key, stats)
            
            stats["end_time"] = datetime.now().isoformat()
            return True, f"Successfully ingested repository {repo_url}", stats
            
        except Exception as e:
            error_message = f"Error ingesting repository: {str(e)}"
            logger.error(error_message)
            stats["errors"].append(error_message)
            stats["end_time"] = datetime.now().isoformat()
            return False, error_message, stats
    
    def _create_repo_node(self, repo_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a repository node in the database.
        
        Args:
            repo_info: Repository information
            
        Returns:
            Created node
        """
        repo_key = repo_info["repo_name"].replace(" ", "_").replace("-", "_").lower()
        
        node_data = {
            "_key": repo_key,
            "name": repo_info["repo_name"],
            "repo_url": repo_info["remote_url"],
            "type": self.NODE_TYPES["REPOSITORY"],
            "branches": repo_info["branches"],
            "current_branch": repo_info["current_branch"],
            "commit_count": repo_info["commit_count"],
            "last_commit": repo_info["last_commit"],
            "contributors": repo_info["contributors"],
            "ingested_at": datetime.now().isoformat()
        }
        
        return self.db_connection.insert_document(self.CODE_NODE_COLLECTION, node_data)
    
    def _update_repo_node(self, repo_key: str, stats: Dict[str, Any]) -> None:
        """
        Update repository node with ingestion stats.
        
        Args:
            repo_key: Repository node key
            stats: Ingestion statistics
        """
        update_data = {
            "ingestion_stats": {
                "nodes_created": stats["nodes_created"],
                "edges_created": stats["edges_created"],
                "files_processed": stats["files_processed"],
                "modules_count": stats.get("modules_count", 0),
                "doc_files_count": stats.get("doc_files_count", 0),
                "llm_analyses_count": stats.get("llm_analyses_count", 0),
                "llm_relationships_count": stats.get("llm_relationships_count", 0),
                "llm_edges_created": stats.get("llm_edges_created", 0),
                "completed_at": datetime.now().isoformat()
            }
        }
        
        self.db_connection.update_document(
            self.CODE_NODE_COLLECTION,
            repo_key,
            update_data
        )
    
    def _process_code_files(self, modules: Dict[str, Module], repo_key: str) -> Dict[str, Dict[str, str]]:
        """
        Process code files and create nodes for modules, classes, and functions.
        
        Args:
            modules: Dictionary of modules
            repo_key: Repository node key
            
        Returns:
            Dictionary mapping file paths to node IDs
        """
        file_nodes = {}
        
        for path, module in modules.items():
            # Create file node
            file_key = self._normalize_key(f"{repo_key}_{path}")
            file_node_data = {
                "_key": file_key,
                "name": path,
                "type": self.NODE_TYPES["FILE"],
                "file_type": "python",
                "repository": repo_key,
                "content": module.code,
                "docstring": module.docstring or ""
            }
            
            file_node = self.db_connection.insert_document(self.CODE_NODE_COLLECTION, file_node_data)
            file_nodes[path] = {"id": file_node["_id"], "key": file_node["_key"]}
            
            # Create module node
            module_key = self._normalize_key(f"{repo_key}_{path}_module")
            module_node_data = {
                "_key": module_key,
                "name": module.name,
                "type": self.NODE_TYPES["MODULE"],
                "repository": repo_key,
                "file": file_key,
                "docstring": module.docstring or "",
                "imports": module.imports
            }
            
            module_node = self.db_connection.insert_document(self.CODE_NODE_COLLECTION, module_node_data)
            
            # Create edge from file to module
            self._create_edge(
                from_id=file_node["_id"],
                to_id=module_node["_id"],
                edge_type=self.EDGE_TYPES["CONTAINS"],
                weight=1.0
            )
            
            # Process functions
            for func_name, func in module.functions.items():
                func_key = self._normalize_key(f"{module_key}_{func_name}")
                func_node_data = {
                    "_key": func_key,
                    "name": func_name,
                    "type": self.NODE_TYPES["FUNCTION"],
                    "repository": repo_key,
                    "file": file_key,
                    "module": module_key,
                    "docstring": func.docstring or "",
                    "code": func.code,
                    "parameters": func.parameters,
                    "return_type": func.return_type,
                    "function_calls": func.function_calls,
                    "line_start": func.line_start,
                    "line_end": func.line_end
                }
                
                func_node = self.db_connection.insert_document(self.CODE_NODE_COLLECTION, func_node_data)
                
                # Create edge from module to function
                self._create_edge(
                    from_id=module_node["_id"],
                    to_id=func_node["_id"],
                    edge_type=self.EDGE_TYPES["CONTAINS"],
                    weight=1.0
                )
            
            # Process classes
            for class_name, class_obj in module.classes.items():
                class_key = self._normalize_key(f"{module_key}_{class_name}")
                class_node_data = {
                    "_key": class_key,
                    "name": class_name,
                    "type": self.NODE_TYPES["CLASS"],
                    "repository": repo_key,
                    "file": file_key,
                    "module": module_key,
                    "docstring": class_obj.docstring or "",
                    "code": class_obj.code,
                    "base_classes": class_obj.base_classes,
                    "line_start": class_obj.line_start,
                    "line_end": class_obj.line_end
                }
                
                class_node = self.db_connection.insert_document(self.CODE_NODE_COLLECTION, class_node_data)
                
                # Create edge from module to class
                self._create_edge(
                    from_id=module_node["_id"],
                    to_id=class_node["_id"],
                    edge_type=self.EDGE_TYPES["CONTAINS"],
                    weight=1.0
                )
                
                # Process methods
                for method_name, method in class_obj.methods.items():
                    method_key = self._normalize_key(f"{class_key}_{method_name}")
                    method_node_data = {
                        "_key": method_key,
                        "name": method_name,
                        "type": self.NODE_TYPES["METHOD"],
                        "repository": repo_key,
                        "file": file_key,
                        "module": module_key,
                        "class": class_key,
                        "docstring": method.docstring or "",
                        "code": method.code,
                        "parameters": method.parameters,
                        "return_type": method.return_type,
                        "function_calls": method.function_calls,
                        "line_start": method.line_start,
                        "line_end": method.line_end
                    }
                    
                    method_node = self.db_connection.insert_document(self.CODE_NODE_COLLECTION, method_node_data)
                    
                    # Create edge from class to method
                    self._create_edge(
                        from_id=class_node["_id"],
                        to_id=method_node["_id"],
                        edge_type=self.EDGE_TYPES["CONTAINS"],
                        weight=1.0
                    )
        
        return file_nodes
    
    def _process_doc_files(self, doc_files: Dict[str, DocumentationFile], repo_key: str) -> Dict[str, Dict[str, str]]:
        """
        Process documentation files and create nodes.
        
        Args:
            doc_files: Dictionary of documentation files
            repo_key: Repository node key
            
        Returns:
            Dictionary mapping file paths to node IDs
        """
        doc_nodes = {}
        
        for path, doc_file in doc_files.items():
            # Create file node
            file_key = self._normalize_key(f"{repo_key}_doc_{path}")
            file_node_data = {
                "_key": file_key,
                "name": path,
                "type": self.NODE_TYPES["DOCUMENTATION"],
                "file_type": Path(path).suffix[1:],  # Remove leading dot
                "repository": repo_key,
                "content": open(doc_file.file_path, 'r', encoding='utf-8').read(),
            }
            
            file_node = self.db_connection.insert_document(self.CODE_NODE_COLLECTION, file_node_data)
            doc_nodes[path] = {"id": file_node["_id"], "key": file_node["_key"]}
            
            # Process documentation elements
            for i, element in enumerate(doc_file.elements):
                element_key = self._normalize_key(f"{file_key}_section_{i}")
                element_node_data = {
                    "_key": element_key,
                    "name": element.title,
                    "type": self.NODE_TYPES["DOC_SECTION"],
                    "section_type": element.section_type,
                    "repository": repo_key,
                    "file": file_key,
                    "content": element.content,
                    "references": element.references,
                    "line_start": element.line_start,
                    "line_end": element.line_end
                }
                
                element_node = self.db_connection.insert_document(self.CODE_NODE_COLLECTION, element_node_data)
                
                # Create edge from file to element
                self._create_edge(
                    from_id=file_node["_id"],
                    to_id=element_node["_id"],
                    edge_type=self.EDGE_TYPES["CONTAINS"],
                    weight=1.0
                )
        
        return doc_nodes
    
    def _create_code_relationships(self, relationships: Dict[str, List[Dict[str, Any]]], 
                                  file_nodes: Dict[str, Dict[str, str]]) -> int:
        """
        Create edges for code relationships.
        
        Args:
            relationships: Dictionary of relationships
            file_nodes: Dictionary mapping file paths to node IDs
            
        Returns:
            Number of edges created
        """
        edges_created = 0
        
        # Process imports
        for import_rel in relationships.get("imports", []):
            source_path = import_rel["source"]
            target_path = import_rel["target"]
            
            if source_path in file_nodes and target_path in file_nodes:
                self._create_edge(
                    from_id=file_nodes[source_path]["id"],
                    to_id=file_nodes[target_path]["id"],
                    edge_type=self.EDGE_TYPES["IMPORTS"],
                    weight=import_rel.get("weight", 0.7),
                    attributes={"alias": import_rel.get("alias", "")}
                )
                edges_created += 1
        
        # Process inheritance relationships
        for inherits_rel in relationships.get("inherits", []):
            source = inherits_rel["source"]
            target = inherits_rel["target"]
            
            # Try to find the nodes
            source_parts = source.split("::")
            target_parts = target.split("::")
            
            if len(source_parts) >= 2 and len(target_parts) >= 2:
                source_path = source_parts[0]
                target_path = target_parts[0]
                
                if source_path in file_nodes and target_path in file_nodes:
                    # Add prefix to search for the exact node keys
                    source_class = source_parts[1]
                    target_class = target_parts[1]
                    
                    # This is a simplified approach; in a real implementation,
                    # you would query the database to find the actual node IDs
                    source_key = self._normalize_key(f"{file_nodes[source_path]['key']}_{source_class}")
                    target_key = self._normalize_key(f"{file_nodes[target_path]['key']}_{target_class}")
                    
                    try:
                        self._create_edge(
                            from_key=source_key,
                            to_key=target_key,
                            edge_type=self.EDGE_TYPES["INHERITS"],
                            weight=inherits_rel.get("weight", 0.8)
                        )
                        edges_created += 1
                    except Exception as e:
                        logger.warning(f"Failed to create inheritance edge: {e}")
        
        # Process function calls
        for call_rel in relationships.get("calls", []):
            source = call_rel["source"]
            target = call_rel["target"]
            
            # Try to find the nodes
            source_parts = source.split("::")
            target_parts = target.split("::")
            
            if len(source_parts) >= 2 and len(target_parts) >= 2:
                source_path = source_parts[0]
                target_path = target_parts[0]
                
                if source_path in file_nodes and target_path in file_nodes:
                    # Add prefix to search for the exact node keys
                    source_func = source_parts[1]
                    target_func = target_parts[1]
                    
                    # This is a simplified approach; in a real implementation,
                    # you would query the database to find the actual node IDs
                    source_key = self._normalize_key(f"{file_nodes[source_path]['key']}_{source_func}")
                    target_key = self._normalize_key(f"{file_nodes[target_path]['key']}_{target_func}")
                    
                    try:
                        self._create_edge(
                            from_key=source_key,
                            to_key=target_key,
                            edge_type=self.EDGE_TYPES["CALLS"],
                            weight=call_rel.get("weight", 0.6)
                        )
                        edges_created += 1
                    except Exception as e:
                        logger.warning(f"Failed to create call edge: {e}")
        
        return edges_created
    
    def _create_doc_code_relationships(self, relationships: List[Dict[str, Any]],
                                      file_nodes: Dict[str, Dict[str, str]],
                                      doc_nodes: Dict[str, Dict[str, str]]) -> int:
        """
        Create edges between documentation and code.
        
        Args:
            relationships: List of relationships
            file_nodes: Dictionary mapping file paths to node IDs
            doc_nodes: Dictionary mapping documentation paths to node IDs
            
        Returns:
            Number of edges created
        """
        edges_created = 0
        
        for rel in relationships:
            source = rel["source"]
            target = rel["target"]
            
            # Parse source (doc element)
            source_parts = source.split("::")
            if len(source_parts) >= 2:
                doc_path = source_parts[0]
                section_title = source_parts[1]
                
                if doc_path in doc_nodes:
                    # For target, we need to search for potential matches in code nodes
                    # This would typically involve a database query; simplified here
                    
                    # Check if target is a direct file path
                    if target in file_nodes:
                        try:
                            self._create_edge(
                                from_id=doc_nodes[doc_path]["id"],
                                to_id=file_nodes[target]["id"],
                                edge_type=self.EDGE_TYPES["DOCUMENTS"],
                                weight=rel.get("weight", 0.8),
                                attributes={"section_title": section_title}
                            )
                            edges_created += 1
                        except Exception as e:
                            logger.warning(f"Failed to create documentation edge: {e}")
                    else:
                        # This would be a more complex lookup in a real implementation
                        logger.debug(f"Could not resolve target for doc relationship: {target}")
        
        return edges_created
    
    def _create_edge(self, from_id: Optional[str] = None, to_id: Optional[str] = None,
                    from_key: Optional[str] = None, to_key: Optional[str] = None,
                    edge_type: str = "generic", weight: float = 0.5,
                    attributes: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Create an edge between two nodes.
        
        Args:
            from_id: Source node ID
            to_id: Target node ID
            from_key: Source node key
            to_key: Target node key
            edge_type: Type of edge
            weight: Edge weight
            attributes: Additional edge attributes
            
        Returns:
            Created edge
        """
        if not ((from_id and to_id) or (from_key and to_key)):
            raise ValueError("Either (from_id, to_id) or (from_key, to_key) must be provided")
            
        edge_data = {
            "_from": from_id if from_id else f"{self.CODE_NODE_COLLECTION}/{from_key}",
            "_to": to_id if to_id else f"{self.CODE_NODE_COLLECTION}/{to_key}",
            "type": edge_type,
            "weight": weight,
            "created_at": datetime.now().isoformat()
        }
        
        if attributes:
            edge_data.update(attributes)
            
        return self.db_connection.insert_edge(self.CODE_EDGE_COLLECTION, edge_data)
    
    @staticmethod
    def _normalize_key(key: str) -> str:
        """
        Normalize a key to be valid for ArangoDB.
        
        Args:
            key: Key to normalize
            
        Returns:
            Normalized key
        """
        # Replace invalid characters with underscores
        normalized = ''.join(c if c.isalnum() or c == '_' else '_' for c in key)
        
        # Ensure key doesn't start with a number
        if normalized and normalized[0].isdigit():
            normalized = 'n' + normalized
            
        # Truncate if too long (ArangoDB limit is 254 bytes)
        if len(normalized) > 250:
            normalized = normalized[:250]
            
        return normalized

    def _perform_llm_code_analysis(self, repo_path: Path, modules: Dict[str, Module]) -> Tuple[Dict[str, Dict], List[Dict]]:  
        """
        Perform LLM-powered code analysis on files in a repository.
        
        Args:
            repo_path: Path to the repository directory
            modules: Dictionary of parsed modules
            
        Returns:
            Tuple of (analyses_by_file, extracted_relationships)
        """
        analyses = {}
        relationships = []
        analyzed_count = 0
        total_files = len(modules)
        
        # Limit the number of files to analyze to prevent excessive processing time
        max_files_to_analyze = min(50, total_files)  # Adjust this number as needed
        
        # Sort modules by size (smaller files first) to optimize processing
        sorted_modules = sorted(
            modules.items(), 
            key=lambda x: len(Path(x[0]).read_text()) if Path(x[0]).exists() else 0
        )
        
        # Find important files to prioritize (e.g., core modules, adapters, etc.)
        important_files = []
        for file_path, _ in sorted_modules:
            if any(keyword in file_path.lower() for keyword in 
                   ["adapter", "core", "model", "api", "database", "main", "app", "config"]):
                important_files.append(file_path)
        
        # Prepare list of files to analyze with important files first
        files_to_analyze = important_files[:max_files_to_analyze//2]
        remaining_slots = max_files_to_analyze - len(files_to_analyze)
        
        # Add remaining files that weren't marked as important
        for file_path, _ in sorted_modules:
            if file_path not in files_to_analyze and remaining_slots > 0:
                files_to_analyze.append(file_path)
                remaining_slots -= 1
        
        logger.info(f"Selected {len(files_to_analyze)} files for LLM analysis out of {total_files} total files")
        
        # Analyze each selected file
        for file_path in files_to_analyze:
            try:
                if not Path(file_path).exists():
                    logger.warning(f"File {file_path} not found, skipping LLM analysis")
                    continue
                
                logger.info(f"Analyzing file with LLM: {file_path} ({analyzed_count}/{len(files_to_analyze)})")
                
                # Perform LLM analysis on the file
                analysis = self.llm_analyzer.analyze_code_file(file_path)
                
                if analysis:
                    analyses[file_path] = analysis
                    analyzed_count += 1
                    logger.info(f"Completed analysis for {file_path}")
                else:
                    logger.warning(f"LLM analysis returned no results for {file_path}")
            except Exception as e:
                logger.error(f"Error during LLM analysis of {file_path}: {e}")
        
        # Extract relationships between analyzed files
        if len(analyses) >= 2:
            logger.info("Extracting relationships between analyzed files...")
            analyzed_files = list(analyses.keys())
            
            # Limit number of file pairs to analyze to prevent combinatorial explosion
            max_relationships = 50
            relationship_count = 0
            
            for i in range(len(analyzed_files)):
                for j in range(i+1, len(analyzed_files)):
                    if relationship_count >= max_relationships:
                        break
                    
                    file1 = analyzed_files[i]
                    file2 = analyzed_files[j]
                    
                    try:
                        # Extract relationships between the two files
                        file_relationships = self.llm_analyzer.extract_relationships(
                            analyses[file1], analyses[file2], file1=file1, file2=file2
                        )
                        
                        if file_relationships:
                            relationships.extend(file_relationships)
                            relationship_count += len(file_relationships)
                            logger.info(f"Found {len(file_relationships)} relationships between {file1} and {file2}")
                    except Exception as e:
                        logger.error(f"Error extracting relationships between {file1} and {file2}: {e}")
        
        logger.info(f"Completed LLM analysis with {len(analyses)} files and {len(relationships)} relationships")
        return analyses, relationships
    
    def _create_llm_relationships(self, relationships: List[Dict], file_nodes: Dict[str, Dict[str, str]]) -> int:
        """
        Create edges for relationships identified by LLM analysis.
        
        Args:
            relationships: List of relationships extracted by LLM
            file_nodes: Dictionary mapping file paths to node IDs
            
        Returns:
            Number of edges created
        """
        edges_created = 0
        
        # Create edges for each relationship
        for rel in relationships:
            try:
                # Extract relationship data
                rel_type = rel.get("type", "RELATED").upper()  # Default to RELATED
                from_element = rel.get("from_element")
                to_element = rel.get("to_element")
                description = rel.get("description", "")
                confidence = float(rel.get("confidence", 0.5))  # Default to 0.5
                from_file = rel.get("from_file")
                to_file = rel.get("to_file")
                
                # Skip if required information is missing
                if not from_element or not to_element or not from_file or not to_file:
                    logger.warning(f"Skipping relationship due to missing information: {rel}")
                    continue
                
                # Map to edge type
                edge_type = self.EDGE_TYPES.get(rel_type, "related")
                if edge_type not in self.EDGE_TYPES.values():
                    # If not a known edge type, map to a default
                    if "CALLS" in rel_type or "INVOKES" in rel_type:
                        edge_type = self.EDGE_TYPES["CALLS"]
                    elif "INHERITS" in rel_type or "EXTENDS" in rel_type:
                        edge_type = self.EDGE_TYPES["INHERITS"]
                    elif "IMPORTS" in rel_type or "USES" in rel_type:
                        edge_type = self.EDGE_TYPES["IMPORTS"]
                    elif "IMPLEMENTS" in rel_type:
                        edge_type = self.EDGE_TYPES["IMPLEMENTS"]
                    elif "DOCUMENTS" in rel_type or "DESCRIBES" in rel_type:
                        edge_type = self.EDGE_TYPES["DOCUMENTS"]
                    else:
                        edge_type = "related"
                
                # Validate that the files exist in our nodes
                if from_file not in file_nodes or to_file not in file_nodes:
                    logger.warning(f"Skipping relationship - files not found: {from_file} or {to_file}")
                    continue
                
                # Find the specific node keys for the elements
                from_module_key = file_nodes[from_file].get("module")
                to_module_key = file_nodes[to_file].get("module")
                
                # If module keys aren't available, use file keys
                if not from_module_key:
                    from_module_key = file_nodes[from_file].get("file")
                if not to_module_key:
                    to_module_key = file_nodes[to_file].get("file")
                
                if not from_module_key or not to_module_key:
                    logger.warning(f"Skipping relationship - module keys not found: {from_module_key} or {to_module_key}")
                    continue
                
                # Create edge attributes
                edge_attrs = {
                    "description": description,
                    "confidence": confidence,
                    "from_element": from_element,
                    "to_element": to_element,
                    "from_file": from_file,
                    "to_file": to_file,
                    "llm_generated": True
                }
                
                # Create the edge
                self._create_edge(
                    from_key=from_module_key,
                    to_key=to_module_key,
                    edge_type=edge_type,
                    weight=confidence,
                    attributes=edge_attrs
                )
                
                edges_created += 1
                
            except Exception as e:
                logger.error(f"Error creating edge for LLM relationship: {e}, {rel}")
        
        return edges_created

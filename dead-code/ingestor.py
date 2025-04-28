"""
HADES-PathRAG Parallel Ingestion Pipeline.

This module provides a parallel and modular ingestion pipeline that efficiently processes
different file types concurrently, using type-specific pre-processors and ISNE embeddings.
"""

import os
from pathlib import Path
import logging
from typing import Dict, List, Any, Optional, Union, Tuple, TypedDict, cast
from concurrent.futures import ThreadPoolExecutor, Future
from datetime import datetime

from src.types.common import EmbeddingVector, Module, DocumentationFile, NodeData, EdgeData

from src.db.arango_connection import ArangoConnection

from src.ingest.file_batcher import FileBatcher, collect_and_batch_files
from src.ingest.pre_processor import get_pre_processor
from src.ingest.parsers.code_parser import CodeParser

# ArangoDB adapter for PathRAG
class ArangoPathRAGAdapter:
    """Interface for ArangoDB operations specific to PathRAG."""
    
    def __init__(self, db_connection: ArangoConnection):
        """Initialize with ArangoDB connection."""
        self.db_connection = db_connection
    
    def get_document(self, document_id: str) -> Dict[str, Any]:
        """Get a document by its ID."""
        result = self.db_connection.get_collection('code_nodes').get(document_id)
        return result if result else {}
    
    def update_document(self, document_id: str, data: Dict[str, Any]) -> bool:
        """Update a document."""
        try:
            self.db_connection.get_collection('code_nodes').update(document_id, data)
            return True
        except Exception:
            return False

from src.utils.git_operations import GitOperations
from src.isne.integrations.storage import ArangoStorage

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IngestStats(TypedDict, total=False):
    """Type definition for ingestion statistics."""

    dataset_name: str
    directory: str
    start_time: str
    end_time: str
    duration_seconds: float
    file_stats: Dict[str, Any]
    document_count: int
    relationship_count: int
    storage_stats: Dict[str, Any]


class RepositoryIngestor:
    """
    Parallel ingestion pipeline for the HADES-PathRAG system.
    
    This class orchestrates the ingestion process with parallel processing
    of different file types, modular pre-processors, and ISNE embedding.
    
    Note: This replaces the previous implementation with a more efficient
    parallel processing approach.
    """
    
    # Node and edge collection names
    CODE_NODE_COLLECTION: str = "code_nodes"
    CODE_EDGE_COLLECTION: str = "code_edges"
    CODE_GRAPH_NAME: str = "code_graph"
    
    # Node types for categorization
    NODE_TYPES: Dict[str, str] = {
        "REPOSITORY": "repository",
        "FILE": "file",
        "MODULE": "module",
        "CLASS": "class",
        "FUNCTION": "function",
        "METHOD": "method",
        "DOCUMENTATION": "documentation",
        "DOC_SECTION": "doc_section"
    }
    
    # Edge types for relationships
    EDGE_TYPES: Dict[str, str] = {
        "CONTAINS": "contains",
        "IMPORTS": "imports",
        "CALLS": "calls",
        "INHERITS": "inherits",
        "REFERENCES": "references",
        "DOCUMENTS": "documents",
        "RELATED": "related"
    }
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the parallel ingestion pipeline.
        
        Args:
            config: Configuration dictionary for the pipeline
        """
        self.config: Dict[str, Any] = config or {}
        
        # Database configuration
        db_config: Dict[str, Any] = self.config.get("database", {})
        self.db_params: Dict[str, Union[str, int]] = {
            "database": db_config.get("database", "pathrag"),
            "host": db_config.get("host", "localhost"),
            "port": db_config.get("port", 8529),
            "username": db_config.get("username", "root"),
            "password": db_config.get("password", "")
        }
        
        # Initialize database connection
        self.db_connection: ArangoConnection = ArangoConnection(
            db_name=cast(str, self.db_params["database"]),
            host=cast(str, self.db_params["host"]),
            username=cast(str, self.db_params["username"]),
            password=cast(str, self.db_params["password"])
        )
        
        # Initialize PathRAG adapter
        self.pathrag_adapter: ArangoPathRAGAdapter = ArangoPathRAGAdapter(self.db_connection)
        
        # Initialize ArangoStorage for ISNE
        self.storage: ArangoStorage = ArangoStorage(self.db_connection)
        
        # Parallel processing configuration
        self.max_workers: int = cast(int, self.config.get("max_workers", 4))
        
        # Initialize batcher
        self.batcher: FileBatcher = FileBatcher()
        
        # Initialize ISNE components (lazy-loaded)
        self._isne_connector: Optional[Any] = None
    
    def setup_collections(self) -> None:
        """
        Set up the necessary collections in the database.
        """
        try:
            # Create graph if it doesn't exist
            if not self.db_connection.graph_exists(self.CODE_GRAPH_NAME):
                # Create edge definitions for the graph
                edge_definitions = [
                    {
                        'edge_collection': self.CODE_EDGE_COLLECTION,
                        'from_vertex_collections': [self.CODE_NODE_COLLECTION],
                        'to_vertex_collections': [self.CODE_NODE_COLLECTION]
                    }
                ]
                self.db_connection.create_graph(self.CODE_GRAPH_NAME, edge_definitions)
            logger.info(f"Created graph {self.CODE_GRAPH_NAME}")
            
            # Create node collection if it doesn't exist
            if not self.db_connection.collection_exists(self.CODE_NODE_COLLECTION):
                self.db_connection.create_collection(self.CODE_NODE_COLLECTION)
                logger.info(f"Created collection {self.CODE_NODE_COLLECTION}")
            
            # Create edge collection if it doesn't exist
            if not self.db_connection.collection_exists(self.CODE_EDGE_COLLECTION):
                self.db_connection.create_edge_collection(self.CODE_EDGE_COLLECTION)
                logger.info(f"Created edge collection {self.CODE_EDGE_COLLECTION}")
            
        except Exception as e:
            logger.error(f"Error setting up collections: {e}")
            raise
    
    def ingest(self, directory: Union[str, Path], dataset_name: Optional[str] = None) -> IngestStats:
        """
        Main entry point for parallel ingestion process.
        
        Args:
            directory: Path to the directory to ingest
            dataset_name: Optional name for the dataset
            
        Returns:
            Dictionary with ingestion statistics
        """
        # Convert to Path object if needed
        directory_path: Path = Path(directory) if isinstance(directory, str) else directory
        
        # Use directory name as dataset name if not provided
        if not dataset_name:
            dataset_name = directory_path.name
        
        logger.info(f"Starting parallel ingestion of {dataset_name} from {directory_path}")
        
        # 1. Set up collections
        self.setup_collections()
        
        start_time: datetime = datetime.now()
        
        # 2. Discover and batch files
        logger.info("Discovering and batching files...")
        # Convert Path to str for collect_files which expects a string
        dir_path_str = str(directory_path) if isinstance(directory_path, Path) else directory_path
        file_batches: Dict[str, List[str]] = self.batcher.collect_files(dir_path_str)
        batch_stats: Dict[str, Any] = self.batcher.get_stats(file_batches)
        logger.info(f"Found {batch_stats['total']} files across {len(file_batches)} types")
        
        # 3. Pre-process files in parallel
        logger.info("Pre-processing files in parallel...")
        processed_docs: List[Dict[str, Any]] = self._parallel_preprocess(file_batches)
        logger.info(f"Pre-processed {len(processed_docs)} documents")
        
        # 4. Extract relationships
        logger.info("Extracting relationships...")
        relationships: List[Dict[str, Any]] = self._extract_relationships(processed_docs)
        logger.info(f"Extracted {len(relationships)} relationships")
        
        # 5. Generate embeddings with ISNE
        logger.info("Generating ISNE embeddings...")
        embedded_docs: List[Dict[str, Any]] = self._generate_embeddings(processed_docs)
        
        # 6. Store in database
        logger.info("Storing in database...")
        storage_results: Dict[str, Any] = self._store_documents(embedded_docs, relationships)
        
        end_time: datetime = datetime.now()
        duration: float = (end_time - start_time).total_seconds()
        
        # 7. Return results
        stats: IngestStats = {
            "dataset_name": dataset_name,
            "directory": str(directory_path),
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration_seconds": duration,
            "file_stats": batch_stats,
            "document_count": len(processed_docs),
            "relationship_count": len(relationships),
            "storage_stats": storage_results,
        }
        
        logger.info(f"Ingestion complete: {stats['document_count']} documents processed in {duration:.2f} seconds")
        return stats
        
    def _parallel_preprocess(self, file_batches: Dict[str, List[str]]) -> List[Dict[str, Any]]:
        """
        Process all file batches in parallel.
        
        Args:
            file_batches: Dictionary mapping file types to lists of file paths
            
        Returns:
            List of processed documents
        """
        results: List[Dict[str, Any]] = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_batch: Dict[Future[List[Dict[str, Any]]], str] = {}
            
            # Submit each batch to the executor
            for file_type, files in file_batches.items():
                if not files:
                    continue
                
                try:
                    processor = get_pre_processor(file_type)
                    future = executor.submit(processor.process_batch, files)
                    future_to_batch[future] = file_type
                except ValueError as e:
                    # Skip unsupported file types
                    logger.warning(f"Skipping {len(files)} {file_type} files: {e}")
            
            # Collect results as they complete
            for future in future_to_batch:
                try:
                    file_type = future_to_batch[future]
                    batch_results = future.result()
                    logger.info(f"Processed {len(batch_results)} {file_type} files")
                    results.extend(batch_results)
                except Exception as e:
                    file_type = future_to_batch[future]
                    logger.error(f"Error processing {file_type} files: {e}")
        
        return results
    
    def _extract_relationships(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract relationships from processed documents.
        
        Args:
            documents: List of processed documents
            
        Returns:
            List of relationships
        """
        relationships: List[Dict[str, Any]] = []
        
        # Collect all document relationships
        for doc in documents:
            if 'relationships' in doc and doc['relationships']:
                relationships.extend(doc['relationships'])
        
        # Deduplicate relationships
        unique_relationships: List[Dict[str, Any]] = []
        seen: set[str] = set()
        
        for rel in relationships:
            # Create a hash of the relationship
            rel_hash = f"{rel['from']}::{rel['to']}::{rel['type']}"
            if rel_hash not in seen:
                unique_relationships.append(rel)
                seen.add(rel_hash)
        
        return unique_relationships
    
    def _generate_embeddings(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate ISNE embeddings for documents.
        
        Args:
            documents: List of processed documents
            
        Returns:
            List of documents with embeddings
        """
        from src.ingest.isne_connector import ISNEIngestorConnector
        
        try:
            isne_connector: ISNEIngestorConnector = ISNEIngestorConnector()
            embedded_docs: List[Dict[str, Any]] = isne_connector.process_documents(documents)
            return embedded_docs
        except Exception as e:
            logger.error(f"Error generating ISNE embeddings: {e}")
            # Return original documents as fallback
            return documents
            
    def _store_documents(self, documents: List[Dict[str, Any]], relationships: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Store documents and relationships in ArangoDB.
        
        Args:
            documents: List of documents with embeddings
            relationships: List of relationships
            
        Returns:
            Storage statistics
        """
        # Store documents as nodes
        node_count: int = 0
        for doc in documents:
            try:
                # Create node data
                node_data: Dict[str, Any] = {
                    "_key": self._normalize_key(doc['id']),
                    "type": doc['type'],
                    "content": doc.get('content', ''),
                    "path": doc.get('path', ''),
                    "created_at": datetime.now().isoformat(),
                }
                
                # Add embedding if available
                if 'embedding' in doc:
                    node_data['embedding'] = doc['embedding']
                
                # Store additional metadata
                for key, value in doc.items():
                    if key not in ['id', 'type', 'content', 'path', 'embedding', 'relationships']:
                        node_data[key] = value
                
                # Create node
                self.db_connection.insert_document(self.CODE_NODE_COLLECTION, node_data)
                node_count += 1
            except Exception as e:
                logger.error(f"Error storing document {doc.get('id')}: {e}")
        
        # Store relationships as edges
        edge_count = 0
        for rel in relationships:
            try:
                # Create edge data
                edge_data = {
                    "_from": f"{self.CODE_NODE_COLLECTION}/{self._normalize_key(rel['from'])}",
                    "_to": f"{self.CODE_NODE_COLLECTION}/{self._normalize_key(rel['to'])}",
                    "type": rel['type'],
                    "weight": rel.get('weight', 0.5),
                    "created_at": datetime.now().isoformat(),
                }
                
                # Add additional attributes
                for key, value in rel.items():
                    if key not in ['from', 'to', 'type', 'weight']:
                        edge_data[key] = value
                
                # Create edge
                self.db_connection.insert_edge(self.CODE_EDGE_COLLECTION, edge_data)
                edge_count += 1
            except Exception as e:
                logger.error(f"Error storing relationship from {rel['from']} to {rel['to']}: {e}")
        
        return {
            "node_count": node_count,
            "edge_count": edge_count,
        }
    
    def process_repository_with_isne(self, repo_path: Union[str, Path], repo_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a repository with the ISNE pipeline.
        
        This method uses the new type-safe ISNE pipeline to process
        a code repository, generating embeddings and relationships
        between code elements.
        
        Args:
            repo_path: Path to the repository
            repo_name: Name of the repository (optional)
            
        Returns:
            Dictionary with processing statistics
        """
        # Convert to Path object if needed
        repo_path_obj: Path = Path(repo_path) if isinstance(repo_path, str) else repo_path
        
        # Use directory name as repo name if not provided
        if not repo_name:
            repo_name = repo_path_obj.name
        
        logger.info(f"Processing repository {repo_name} with ISNE pipeline")
        
        # Initialize ISNE pipeline
        from src.isne.pipeline import PathRAGISNEAdapter
        self._isne_connector = PathRAGISNEAdapter(self.db_connection)
            
        dataset: Any = self._isne_connector.process_repository(repo_path_obj, repo_name, store_in_arango=True)
        
        # Create repository stats
        stats: Dict[str, Any] = {
            "repository_name": repo_name,
            "repository_path": str(repo_path_obj),
            "processed_at": datetime.now().isoformat(),
            "pipeline": "isne"
        }
        
        # Add dataset stats if available
        if dataset:
            stats.update({
                "document_count": len(dataset.documents),
                "relation_count": len(dataset.relations),
                "dataset_id": dataset.id,
                "dataset_name": dataset.name
            })
        
        return stats
    
    def update_code_node_embedding(self, node_id: str, embedding: EmbeddingVector, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Update the embedding of a code node in the database.
        
        Args:
            node_id: ID of the code node
            embedding: Embedding vector
            metadata: Additional metadata
            
        Returns:
            True if successful, False otherwise
        """
        try:
            import numpy as np
            
            # Convert embedding to list if it's a numpy array
            if isinstance(embedding, np.ndarray):
                result_embeddings = isne_resp.get("embeddings", [])
            embedding_list = result_embeddings[0] if result_embeddings else []
            
            # Create update document
            update_doc: Dict[str, Any] = {
                "embedding": embedding_list,
                "updated_at": datetime.now().isoformat()
            }
            
            # Add metadata if provided
            if metadata:
                # If node already has metadata, merge with existing
                node: Optional[Dict[str, Any]] = self.db_connection.get_document(self.CODE_NODE_COLLECTION, node_id)
                if node and "metadata" in node:
                    existing_metadata: Dict[str, Any] = node["metadata"]
                    merged_metadata: Dict[str, Any] = {**existing_metadata, **metadata}
                    update_doc["metadata"] = merged_metadata
                else:
                    update_doc["metadata"] = metadata
            
            # Get document properties using pathrag_adapter
            doc_properties = self.pathrag_adapter.get_document(node_id)
            
            # Update node in database
            self.db_connection.update_document(
                self.CODE_NODE_COLLECTION,
                update_doc,
                key=node_id
            )
            
            return True
        except Exception as e:
            logger.error(f"Error updating code node embedding: {e}")
            return False
    
    def find_similar_code(self, code_content: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Find code nodes similar to the provided code content.
        
        This method uses the ISNE pipeline to generate an embedding for the
        provided code content and then searches for similar code nodes in
        the database.
        
        Args:
            code_content: Source code content to find similar nodes for
            limit: Maximum number of results to return
            
        Returns:
            List of similar code nodes with similarity scores
        """
        # Initialize ISNE pipeline
        from src.isne.pipeline import PathRAGISNEAdapter
        self._isne_connector = PathRAGISNEAdapter(self.db_connection)
            
        # Get embedding for code content
        embedding: Optional[EmbeddingVector] = self._isne_connector.get_document_embedding(code_content, "code")
        
        if not embedding:
            logger.error("Failed to generate embedding for code content")
            return []
        
        # Find similar documents
        similar_docs: List[Tuple[Any, float]] = self._isne_connector.find_similar_documents(
            code_content,
            limit=limit,
            min_score=0.6  # Lower threshold for code similarity
        )
        
        # Convert to simplified results format
        results: List[Dict[str, Any]] = []
        for doc, score in similar_docs:
            results.append({
                "id": doc.id,
                "content": doc.content[:200] + "..." if len(doc.content) > 200 else doc.content,
                "title": doc.title or "Untitled",
                "source": doc.source,
                "score": score,
                "metadata": doc.metadata
            })
        
        return results
    
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
        stats: Dict[str, Any] = {
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
            git_ops: GitOperations = GitOperations(base_dir=base_dir)
            clone_success: bool
            clone_message: str
            repo_path: Path
            clone_success, clone_message, repo_path = git_ops.clone_repository(repo_url, repo_name)
            
            if not clone_success:
                stats["errors"].append(f"Failed to clone repository: {clone_message}")
                return False, clone_message, stats
            
            logger.info(f"Successfully cloned repository to {repo_path}")
            
            # Get repository information
            repo_info: Dict[str, Any] = git_ops.get_repo_info(repo_path)
            stats["repo_info"] = repo_info
            
            # Create repository node
            repo_node: Dict[str, Any] = self._create_repo_node(repo_info)
            repo_key: str = repo_node["_key"]
            stats["repo_key"] = repo_key
            stats["nodes_created"] += 1
            
            # Parse code files using the new CodeParser
            code_parser: CodeParser = CodeParser()
            modules: Dict[str, Module] = code_parser.parse(repo_path)
            stats["modules_count"] = len(modules)
            
            # Parse documentation files using the new DocParser
            # doc_parser: DocParser = DocParser()  # Removed: dead code
            # doc_files: Dict[str, DocumentationFile] = doc_parser.parse_documentation(repo_path)  # Removed: dead code
            stats["doc_files_count"] = len(doc_files)
            
            # Process code files
            file_nodes: Dict[str, Dict[str, str]] = self._process_code_files(modules, repo_key)
            stats["nodes_created"] += len(file_nodes)
            stats["files_processed"] += len(file_nodes)
            
            # Process documentation files
            doc_nodes: Dict[str, Dict[str, str]] = self._process_doc_files(doc_files, repo_key)
            stats["nodes_created"] += len(doc_nodes)
            stats["files_processed"] += len(doc_nodes)
            
            # Extract and create code relationships
            code_relationships: Dict[str, List[Dict[str, Any]]] = code_parser.extract_relationships(modules)
            edges_created: int = self._create_code_relationships(code_relationships, file_nodes)
            stats["edges_created"] += edges_created
            
            # Extract and create doc-code relationships
            # doc_code_relationships: List[Dict[str, Any]] = doc_parser.extract_doc_code_relationships(doc_files)  # Removed: dead code
            doc_edges_created: int = self._create_doc_code_relationships(doc_code_relationships, file_nodes, doc_nodes)
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
        repo_key: str = repo_info["repo_name"].replace(" ", "_").replace("-", "_").lower()
        
        node_data: Dict[str, Any] = {
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
        update_data: Dict[str, Any] = {
            "ingestion_stats": {
                "nodes_created": stats["nodes_created"],
                "edges_created": stats["edges_created"],
                "files_processed": stats["files_processed"],
                "modules_count": stats.get("modules_count", 0),
                "doc_files_count": stats.get("doc_files_count", 0),
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
        file_nodes: Dict[str, Dict[str, str]] = {}
        
        for path, module in modules.items():
            # Create file node
            file_key: str = self._normalize_key(f"{repo_key}_{path}")
            file_node_data: Dict[str, Any] = {
                "_key": file_key,
                "name": path,
                "type": self.NODE_TYPES["FILE"],
                "file_type": "python",
                "repository": repo_key,
                "content": module.code,
                "docstring": module.docstring or ""
            }
            
            file_node: Dict[str, Any] = self.db_connection.insert_document(self.CODE_NODE_COLLECTION, file_node_data)
            file_nodes[path] = {"id": file_node["_id"], "key": file_node["_key"]}
            
            # Create module node
            module_key: str = self._normalize_key(f"{repo_key}_{path}_module")
            module_node_data: Dict[str, Any] = {
                "_key": module_key,
                "name": module.name,
                "type": self.NODE_TYPES["MODULE"],
                "repository": repo_key,
                "file": file_key,
                "docstring": module.docstring or "",
                "imports": module.imports
            }
            
            module_node: Dict[str, Any] = self.db_connection.insert_document(self.CODE_NODE_COLLECTION, module_node_data)
            
            # Create edge from file to module
            self._create_edge(
                from_id=file_node["_id"],
                to_id=module_node["_id"],
                edge_type=self.EDGE_TYPES["CONTAINS"],
                weight=1.0
            )
            
            # Process functions
            for func_name, func in module.functions.items():
                func_key: str = self._normalize_key(f"{module_key}_{func_name}")
                func_node_data: Dict[str, Any] = {
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
                
                func_node: Dict[str, Any] = self.db_connection.insert_document(self.CODE_NODE_COLLECTION, func_node_data)
                
                # Create edge from module to function
                self._create_edge(
                    from_id=module_node["_id"],
                    to_id=func_node["_id"],
                    edge_type=self.EDGE_TYPES["CONTAINS"],
                    weight=1.0
                )
            
            # Process classes
            for class_name, class_obj in module.classes.items():
                class_key: str = self._normalize_key(f"{module_key}_{class_name}")
                class_node_data: Dict[str, Any] = {
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
                
                class_node: Dict[str, Any] = self.db_connection.insert_document(self.CODE_NODE_COLLECTION, class_node_data)
                
                # Create edge from module to class
                self._create_edge(
                    from_id=module_node["_id"],
                    to_id=class_node["_id"],
                    edge_type=self.EDGE_TYPES["CONTAINS"],
                    weight=1.0
                )
                
                # Process methods
                for method_name, method in class_obj.methods.items():
                    method_key: str = self._normalize_key(f"{class_key}_{method_name}")
                    method_node_data: Dict[str, Any] = {
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
                    
                    method_node: Dict[str, Any] = self.db_connection.insert_document(self.CODE_NODE_COLLECTION, method_node_data)
                    
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
        doc_nodes: Dict[str, Dict[str, str]] = {}
        
        for path, doc_file in doc_files.items():
            # Create file node
            file_key: str = self._normalize_key(f"{repo_key}_doc_{path}")
            file_node_data: Dict[str, Any] = {
                "_key": file_key,
                "name": path,
                "type": self.NODE_TYPES["DOCUMENTATION"],
                "file_type": Path(path).suffix[1:],  # Remove leading dot
                "repository": repo_key,
                "content": open(doc_file.file_path, 'r', encoding='utf-8').read(),
            }
            
            file_node: Dict[str, Any] = self.db_connection.insert_document(self.CODE_NODE_COLLECTION, file_node_data)
            doc_nodes[path] = {"id": file_node["_id"], "key": file_node["_key"]}
            
            # Process documentation elements
            for i, element in enumerate(doc_file.elements):
                element_key: str = self._normalize_key(f"{file_key}_section_{i}")
                element_node_data: Dict[str, Any] = {
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
                
                element_node: Dict[str, Any] = self.db_connection.insert_document(self.CODE_NODE_COLLECTION, element_node_data)
                
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
        edges_created: int = 0
        
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
                    source_class: str = source_parts[1]
                    target_class: str = target_parts[1]
                    
                    # This is a simplified approach; in a real implementation,
                    # you would query the database to find the actual node IDs
                    source_key: str = self._normalize_key(f"{file_nodes[source_path]['key']}_{source_class}")
                    target_key: str = self._normalize_key(f"{file_nodes[target_path]['key']}_{target_class}")
                    
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
        edges_created: int = 0
        
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
        # Convert keys to IDs if needed
        from_id_final: Optional[str] = from_id
        to_id_final: Optional[str] = to_id
        
        if from_id_final is None and from_key is not None:
            from_id_final = f"{self.CODE_NODE_COLLECTION}/{from_key}"
        
        if to_id_final is None and to_key is not None:
            to_id_final = f"{self.CODE_NODE_COLLECTION}/{to_key}"
        
        if from_id_final is None or to_id_final is None:
            raise ValueError("Either from_id/to_id or from_key/to_key must be provided")
        
        # Create edge data
        edge_data: Dict[str, Any] = {
            "_from": from_id_final,
            "_to": to_id_final,
            "type": edge_type,
            "weight": weight,
            "created_at": datetime.now().isoformat()
        }
        
        # Add additional attributes if provided
        if attributes:
            for key, value in attributes.items():
                edge_data[key] = value
        
        # Create edge in database
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

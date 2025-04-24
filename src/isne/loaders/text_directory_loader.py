"""
Text directory loader for the ISNE pipeline.

This module provides a loader for reading text documents from a directory
structure, extracting relationships, and building an ingest dataset.
"""

import os
import re
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Set, Iterator, Tuple
import logging
from datetime import datetime

from src.isne.types.models import IngestDocument, DocumentRelation, RelationType
from src.isne.loaders.base_loader import BaseLoader, LoaderConfig, LoaderResult

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextDirectoryLoader(BaseLoader):
    """
    Loader for text files from a directory structure.
    
    This loader traverses a directory structure, loads text files,
    and extracts relationships between documents based on content and structure.
    """
    
    def __init__(
        self, 
        config: Optional[LoaderConfig] = None,
        document_type: str = "text",
        create_relationships: bool = True,
        recursive: bool = True,
        file_extensions: Optional[List[str]] = None
    ) -> None:
        """
        Initialize the text directory loader.
        
        Args:
            config: Loader configuration
            document_type: Type to assign to loaded documents
            create_relationships: Whether to extract relationships between documents
            recursive: Whether to recursively traverse subdirectories
            file_extensions: List of file extensions to include (e.g. [".txt", ".md"])
        """
        super().__init__(config)
        
        self.document_type = document_type
        self.create_relationships = create_relationships
        self.recursive = recursive
        self.file_extensions = file_extensions or [".txt", ".md", ".rst", ".html", ".xml", ".json", ".csv"]
        
    def load(self, source: Union[str, Path]) -> LoaderResult:
        """
        Load text documents from a directory structure.
        
        Args:
            source: Path to the directory to load documents from
            
        Returns:
            LoaderResult containing loaded documents and relationships
        """
        source_path = Path(source)
        if not source_path.exists() or not source_path.is_dir():
            raise ValueError(f"Source directory does not exist: {source}")
        
        logger.info(f"Loading documents from {source_path}")
        
        # Find and filter files
        all_files = self._find_files(source_path)
        file_paths = self.filter_files(all_files)
        
        logger.info(f"Found {len(file_paths)} files to process")
        
        # Load documents
        documents: List[IngestDocument] = []
        errors: List[Dict[str, Any]] = []
        
        for file_path in file_paths:
            try:
                document = self._load_document(file_path, source_path)
                documents.append(document)
            except Exception as e:
                errors.append({
                    "path": str(file_path),
                    "error": str(e),
                    "type": type(e).__name__
                })
                logger.error(f"Error loading {file_path}: {e}")
        
        # Extract relationships
        relations: List[DocumentRelation] = []
        if self.create_relationships and self.config.extract_relationships:
            relations = self._extract_relationships(documents, source_path)
        
        # Create dataset
        dataset_name = f"text_directory_{source_path.name}"
        dataset = self.create_dataset(
            name=dataset_name,
            documents=documents,
            relations=relations,
            description=f"Text documents from {source_path}",
            metadata={
                "source_path": str(source_path),
                "document_count": len(documents),
                "relationship_count": len(relations),
                "error_count": len(errors)
            }
        )
        
        return LoaderResult(
            documents=documents,
            relations=relations,
            dataset=dataset,
            errors=errors,
            metadata={
                "source_path": str(source_path),
                "file_count": len(file_paths),
                "loaded_count": len(documents),
                "error_count": len(errors)
            }
        )
    
    def _find_files(self, directory: Path) -> List[Path]:
        """
        Find all relevant files in the directory.
        
        Args:
            directory: Directory to search
            
        Returns:
            List of file paths
        """
        files: List[Path] = []
        
        if self.recursive:
            # Recursive directory traversal
            for root, _, filenames in os.walk(directory):
                root_path = Path(root)
                for filename in filenames:
                    if any(filename.endswith(ext) for ext in self.file_extensions):
                        files.append(root_path / filename)
        else:
            # Non-recursive directory listing
            for item in directory.iterdir():
                if item.is_file() and any(str(item).endswith(ext) for ext in self.file_extensions):
                    files.append(item)
        
        return files
    
    def _load_document(self, file_path: Path, base_path: Path) -> IngestDocument:
        """
        Load a single document from a file.
        
        Args:
            file_path: Path to the file to load
            base_path: Base directory path for relative path calculation
            
        Returns:
            Loaded IngestDocument
        """
        # Generate document ID
        doc_id = str(uuid.uuid4())
        
        # Read file content
        with open(file_path, "r", encoding=self.config.encoding) as f:
            content = f.read()
        
        # Get file metadata
        stat = file_path.stat()
        created_at = datetime.fromtimestamp(stat.st_ctime)
        updated_at = datetime.fromtimestamp(stat.st_mtime)
        
        # Create relative source path
        try:
            rel_path = file_path.relative_to(base_path)
            source = str(rel_path)
        except ValueError:
            source = str(file_path)
        
        # Extract title from filename or first line
        title = file_path.stem.replace("_", " ").replace("-", " ").title()
        first_line = content.strip().split("\n")[0] if content.strip() else ""
        if first_line and len(first_line) < 100 and not first_line.startswith("#"):
            title = first_line
        
        # Create document
        return IngestDocument(
            id=doc_id,
            content=content,
            source=source,
            document_type=self.document_type,
            title=title,
            created_at=created_at,
            updated_at=updated_at,
            metadata={
                "file_path": str(file_path),
                "file_size": stat.st_size,
                "file_extension": file_path.suffix,
                "relative_path": source
            }
        )
    
    def _extract_relationships(self, documents: List[IngestDocument], base_path: Path) -> List[DocumentRelation]:
        """
        Extract relationships between documents.
        
        Args:
            documents: List of documents to extract relationships from
            base_path: Base directory path for structure-based relationships
            
        Returns:
            List of extracted relationships
        """
        relations: List[DocumentRelation] = []
        
        # Build lookup by source path
        doc_by_source: Dict[str, IngestDocument] = {doc.source: doc for doc in documents}
        
        # Create directory structure relationships
        self._create_structure_relationships(documents, relations, base_path)
        
        # Create content-based relationships (references between documents)
        self._create_content_relationships(documents, relations, doc_by_source)
        
        return relations
    
    def _create_structure_relationships(
        self, 
        documents: List[IngestDocument], 
        relations: List[DocumentRelation],
        base_path: Path
    ) -> None:
        """
        Create relationships based on directory structure.
        
        Args:
            documents: List of documents to process
            relations: List to add relationships to
            base_path: Base directory path
        """
        # Build directory structure map
        dir_docs: Dict[str, List[IngestDocument]] = {}
        
        for doc in documents:
            doc_path = Path(doc.source)
            parent_dir = str(doc_path.parent)
            
            if parent_dir not in dir_docs:
                dir_docs[parent_dir] = []
                
            dir_docs[parent_dir].append(doc)
        
        # Create relationships between documents in the same directory
        for dir_path, dir_documents in dir_docs.items():
            if len(dir_documents) <= 1:
                continue
                
            # Link documents in the same directory with PART_OF relationships
            for i, doc1 in enumerate(dir_documents):
                for doc2 in dir_documents[i+1:]:
                    # Create bidirectional PART_OF relationship
                    relations.append(DocumentRelation(
                        source_id=doc1.id,
                        target_id=doc2.id,
                        relation_type=RelationType.PART_OF,
                        bidirectional=True,
                        metadata={
                            "relationship_source": "directory_structure",
                            "shared_directory": dir_path
                        }
                    ))
    
    def _create_content_relationships(
        self,
        documents: List[IngestDocument],
        relations: List[DocumentRelation],
        doc_by_source: Dict[str, IngestDocument]
    ) -> None:
        """
        Create relationships based on document content references.
        
        Args:
            documents: List of documents to process
            relations: List to add relationships to
            doc_by_source: Mapping of source paths to documents
        """
        for doc in documents:
            # Skip empty documents
            if not doc.content:
                continue
                
            # Extract paths and references from content
            references: Set[str] = set()
            
            # Simple file path references
            file_refs = re.findall(r'(?:file|path|import|include)[\s:"\'=]+([^"\'\s()<>]+\.[a-zA-Z0-9]{2,4})', doc.content)
            references.update(file_refs)
            
            # Markdown links
            md_links = re.findall(r'\[.*?\]\(([^)]+)\)', doc.content)
            references.update(md_links)
            
            # HTML links
            html_links = re.findall(r'href=["\'](.*?)["\']', doc.content)
            references.update(html_links)
            
            # Create relationships for valid references
            for ref in references:
                ref_path = self._normalize_path(ref)
                
                if ref_path in doc_by_source and doc_by_source[ref_path].id != doc.id:
                    target_doc = doc_by_source[ref_path]
                    
                    # Create REFERENCES relationship
                    relations.append(DocumentRelation(
                        source_id=doc.id,
                        target_id=target_doc.id,
                        relation_type=RelationType.REFERENCES,
                        bidirectional=False,
                        metadata={
                            "relationship_source": "content_reference",
                            "reference_text": ref
                        }
                    ))
    
    def _normalize_path(self, path: str) -> str:
        """
        Normalize a path for reference matching.
        
        Args:
            path: Path to normalize
            
        Returns:
            Normalized path string
        """
        # Remove URL fragments and query parameters
        path = path.split("#")[0].split("?")[0]
        
        # Convert to Path and back to string for normalization
        try:
            return str(Path(path))
        except:
            return path

"""
Data loaders for the HADES-PathRAG ingestion pipeline.

This module provides loaders for different data sources, converting them
into the internal IngestDataset format for processing.
"""
import csv
import json
import logging
import os
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Set, Any, Union, Iterator, Tuple

from hades_pathrag.ingestion.models import IngestDataset, IngestDocument, DocumentRelation, RelationType

logger = logging.getLogger(__name__)


class DataLoader(ABC):
    """
    Abstract base class for data loaders.
    
    Data loaders are responsible for loading data from various sources and
    converting them into the internal IngestDataset format.
    """
    
    @abstractmethod
    def load(self, source: Any, **kwargs) -> IngestDataset:
        """
        Load data from the source and return an IngestDataset.
        
        Args:
            source: The data source
            **kwargs: Additional arguments for the specific loader
            
        Returns:
            An IngestDataset containing the loaded documents and relationships
        """
        pass


class TextDirectoryLoader(DataLoader):
    """
    Load text documents from a directory.
    
    This loader recursively scans a directory for text files and loads them
    as documents. It can also infer relationships between documents based on
    their content or structure.
    """
    
    def __init__(
        self,
        file_extensions: List[str] = None,
        extract_relationships: bool = True,
        relationship_patterns: Dict[str, str] = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: int = 0,
    ):
        """
        Initialize the directory loader.
        
        Args:
            file_extensions: List of file extensions to include
            extract_relationships: Whether to extract relationships from content
            relationship_patterns: Dict mapping relationship types to regex patterns
            chunk_size: Maximum chunk size for splitting documents
            chunk_overlap: Overlap between chunks when splitting documents
        """
        self.file_extensions = file_extensions or [".txt", ".md", ".rst", ".json", ".csv"]
        self.extract_relationships = extract_relationships
        self.relationship_patterns = relationship_patterns or {
            RelationType.REFERENCES: r"reference[s]?\s*:\s*([^\n]+)",
            RelationType.LINKS_TO: r"link[s]?\s*:\s*([^\n]+)",
            RelationType.CITES: r"cite[s]?\s*:\s*([^\n]+)",
        }
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def _extract_relationships(self, docs: List[IngestDocument]) -> List[DocumentRelation]:
        """
        Extract relationships between documents based on content.
        
        Args:
            docs: List of documents to process
            
        Returns:
            List of extracted relationships
        """
        relationships = []
        id_to_doc = {doc.id: doc for doc in docs}
        title_to_ids = {}
        
        # Build title-to-id mapping for lookup
        for doc in docs:
            if doc.title:
                normalized_title = doc.title.lower().strip()
                if normalized_title not in title_to_ids:
                    title_to_ids[normalized_title] = []
                title_to_ids[normalized_title].append(doc.id)
        
        # For each document, look for patterns that indicate relationships
        for doc in docs:
            for rel_type, pattern in self.relationship_patterns.items():
                matches = re.finditer(pattern, doc.content, re.IGNORECASE)
                for match in matches:
                    # Extract referenced titles/names
                    refs = match.group(1).split(',')
                    for ref in refs:
                        ref = ref.strip().lower()
                        if ref in title_to_ids:
                            for target_id in title_to_ids[ref]:
                                if target_id != doc.id:  # Avoid self-references
                                    relationships.append(DocumentRelation(
                                        source_id=doc.id,
                                        target_id=target_id,
                                        relation_type=rel_type,
                                        weight=1.0
                                    ))
        
        return relationships
    
    def _chunk_document(self, doc: IngestDocument) -> List[IngestDocument]:
        """
        Chunk a document into smaller pieces if needed.
        
        Args:
            doc: The document to chunk
            
        Returns:
            List of chunked documents, or the original document if no chunking needed
        """
        if not self.chunk_size or len(doc.content) <= self.chunk_size:
            return [doc]
        
        chunks = []
        content = doc.content
        chunk_id = 0
        
        # Split the content into chunks
        for i in range(0, len(content), self.chunk_size - self.chunk_overlap):
            if i > 0:
                start = i
            else:
                start = 0
                
            end = min(start + self.chunk_size, len(content))
            
            chunk_doc = IngestDocument(
                id=f"{doc.id}_chunk_{chunk_id}",
                content=content[start:end],
                title=f"{doc.title} (Part {chunk_id + 1})" if doc.title else f"Chunk {chunk_id + 1}",
                metadata={
                    **doc.metadata,
                    "parent_id": doc.id,
                    "chunk_id": chunk_id,
                    "chunk_start": start,
                    "chunk_end": end,
                }
            )
            chunks.append(chunk_doc)
            chunk_id += 1
            
            if end >= len(content):
                break
        
        return chunks
    
    def load(self, source: str, dataset_name: Optional[str] = None, **kwargs) -> IngestDataset:
        """
        Load documents from a directory.
        
        Args:
            source: Path to the directory
            dataset_name: Name for the dataset (defaults to directory name)
            **kwargs: Additional arguments
            
        Returns:
            An IngestDataset containing the loaded documents
        """
        source_path = Path(source)
        if not source_path.exists() or not source_path.is_dir():
            raise ValueError(f"Source path {source} does not exist or is not a directory")
        
        dataset_name = dataset_name or source_path.name
        dataset = IngestDataset(name=dataset_name)
        
        # Find all files with the specified extensions
        all_files = []
        for ext in self.file_extensions:
            all_files.extend(source_path.glob(f"**/*{ext}"))
        
        logger.info(f"Found {len(all_files)} files with extensions {self.file_extensions}")
        
        # Load each file as a document
        documents = []
        for file_path in all_files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                
                relative_path = file_path.relative_to(source_path)
                doc = IngestDocument(
                    id=str(relative_path),
                    content=content,
                    title=file_path.stem,
                    metadata={
                        "path": str(file_path),
                        "relative_path": str(relative_path),
                        "file_type": file_path.suffix[1:],  # Remove the leading dot
                        "size": file_path.stat().st_size,
                        "modified": file_path.stat().st_mtime,
                    }
                )
                
                # Chunk the document if needed
                chunked_docs = self._chunk_document(doc)
                documents.extend(chunked_docs)
                
                # If we chunked the document, create relationships between chunks
                if len(chunked_docs) > 1:
                    for i, chunk in enumerate(chunked_docs):
                        if i > 0:
                            # Previous chunk refers to this one
                            dataset.add_relationship(DocumentRelation(
                                source_id=chunked_docs[i-1].id,
                                target_id=chunk.id,
                                relation_type=RelationType.FOLLOWS,
                                weight=1.0
                            ))
                        
                        # All chunks are part of the original document
                        dataset.add_relationship(DocumentRelation(
                            source_id=chunk.id,
                            target_id=doc.id,
                            relation_type=RelationType.PART_OF,
                            weight=1.0
                        ))
                
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
        
        # Add all documents to the dataset
        dataset.add_documents(documents)
        
        # Extract relationships if needed
        if self.extract_relationships:
            relationships = self._extract_relationships(documents)
            dataset.add_relationships(relationships)
            logger.info(f"Extracted {len(relationships)} relationships between documents")
        
        logger.info(f"Loaded {len(documents)} documents from {source}")
        return dataset


class JSONLoader(DataLoader):
    """
    Load documents and relationships from a JSON file.
    
    The JSON file should follow a specific format that represents documents
    and their relationships.
    """
    
    def load(self, source: str, **kwargs) -> IngestDataset:
        """
        Load data from a JSON file.
        
        Args:
            source: Path to the JSON file
            **kwargs: Additional arguments
            
        Returns:
            An IngestDataset containing the loaded documents and relationships
        """
        source_path = Path(source)
        if not source_path.exists():
            raise ValueError(f"Source path {source} does not exist")
        
        with open(source_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        if isinstance(data, list):
            # Assume a list of documents
            dataset_name = kwargs.get("dataset_name") or source_path.stem
            dataset = IngestDataset(name=dataset_name)
            
            # Load documents
            documents = []
            for doc_data in data:
                doc = IngestDocument(
                    id=doc_data.get("id", str(len(documents))),
                    content=doc_data.get("content", ""),
                    title=doc_data.get("title"),
                    metadata=doc_data.get("metadata", {}),
                )
                documents.append(doc)
            
            dataset.add_documents(documents)
            
            # Extract relationships if they exist
            if "relationships" in data:
                relationships = []
                for rel_data in data["relationships"]:
                    relationship = DocumentRelation(
                        source_id=rel_data["source_id"],
                        target_id=rel_data["target_id"],
                        relation_type=RelationType(rel_data["relation_type"]),
                        weight=rel_data.get("weight", 1.0),
                        metadata=rel_data.get("metadata", {}),
                    )
                    relationships.append(relationship)
                
                dataset.add_relationships(relationships)
        
        elif isinstance(data, dict):
            # Assume a dataset structure
            dataset_name = data.get("name") or kwargs.get("dataset_name") or source_path.stem
            dataset = IngestDataset(
                name=dataset_name,
                metadata=data.get("metadata", {}),
            )
            
            # Load documents
            if "documents" in data:
                documents = []
                for doc_data in data["documents"]:
                    doc = IngestDocument(
                        id=doc_data.get("id", str(len(documents))),
                        content=doc_data.get("content", ""),
                        title=doc_data.get("title"),
                        metadata=doc_data.get("metadata", {}),
                    )
                    documents.append(doc)
                
                dataset.add_documents(documents)
            
            # Load relationships
            if "relationships" in data:
                relationships = []
                for rel_data in data["relationships"]:
                    try:
                        relationship = DocumentRelation(
                            source_id=rel_data["source_id"],
                            target_id=rel_data["target_id"],
                            relation_type=RelationType(rel_data["relation_type"]),
                            weight=rel_data.get("weight", 1.0),
                            metadata=rel_data.get("metadata", {}),
                        )
                        relationships.append(relationship)
                    except (KeyError, ValueError) as e:
                        logger.warning(f"Error loading relationship: {e}")
                
                dataset.add_relationships(relationships)
        
        else:
            raise ValueError("Invalid JSON format")
        
        logger.info(f"Loaded {len(dataset.documents)} documents and {len(dataset.relationships)} relationships from {source}")
        return dataset


class CSVLoader(DataLoader):
    """
    Load documents from a CSV file.
    
    Each row in the CSV file represents a document, with columns mapping to
    document properties.
    """
    
    def __init__(
        self,
        content_column: str = "content",
        title_column: Optional[str] = "title",
        id_column: Optional[str] = "id",
        delimiter: str = ",",
    ):
        """
        Initialize the CSV loader.
        
        Args:
            content_column: Name of the column containing document content
            title_column: Name of the column containing document title
            id_column: Name of the column containing document ID
            delimiter: CSV delimiter character
        """
        self.content_column = content_column
        self.title_column = title_column
        self.id_column = id_column
        self.delimiter = delimiter
    
    def load(self, source: str, **kwargs) -> IngestDataset:
        """
        Load documents from a CSV file.
        
        Args:
            source: Path to the CSV file
            **kwargs: Additional arguments
            
        Returns:
            An IngestDataset containing the loaded documents
        """
        source_path = Path(source)
        if not source_path.exists():
            raise ValueError(f"Source path {source} does not exist")
        
        dataset_name = kwargs.get("dataset_name") or source_path.stem
        dataset = IngestDataset(name=dataset_name)
        
        with open(source_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter=self.delimiter)
            
            if self.content_column not in reader.fieldnames:
                raise ValueError(f"Content column '{self.content_column}' not found in CSV")
            
            documents = []
            for row in reader:
                # Extract the content (required)
                content = row.get(self.content_column, "")
                
                # Extract optional fields
                doc_id = row.get(self.id_column) if self.id_column else None
                title = row.get(self.title_column) if self.title_column else None
                
                # Create metadata from remaining columns
                metadata = {k: v for k, v in row.items() if k not in [self.content_column, self.title_column, self.id_column]}
                
                doc = IngestDocument(
                    id=doc_id or str(len(documents)),
                    content=content,
                    title=title,
                    metadata=metadata,
                )
                documents.append(doc)
        
        dataset.add_documents(documents)
        logger.info(f"Loaded {len(documents)} documents from {source}")
        return dataset

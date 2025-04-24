"""
JSON loader for the ISNE pipeline.

This module provides a loader for reading documents from JSON files,
allowing for flexible document and relationship extraction.
"""

import json
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Set, Iterator, Tuple, cast
import logging
from datetime import datetime

from src.isne.types.models import IngestDocument, DocumentRelation, RelationType
from src.isne.loaders.base_loader import BaseLoader, LoaderConfig, LoaderResult

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class JSONLoader(BaseLoader):
    """
    Loader for documents from JSON files.
    
    This loader extracts documents and relationships from structured JSON data,
    supporting various formats and configurations.
    """
    
    def __init__(
        self, 
        config: Optional[LoaderConfig] = None,
        document_type: str = "json",
        document_key: str = "documents",
        relation_key: Optional[str] = "relations",
        id_field: str = "id",
        content_field: str = "content",
        source_field: str = "source",
        flatten_nested: bool = False
    ) -> None:
        """
        Initialize the JSON loader.
        
        Args:
            config: Loader configuration
            document_type: Type to assign to loaded documents
            document_key: Key in JSON containing documents array
            relation_key: Key in JSON containing relationships array (optional)
            id_field: Field name for document ID
            content_field: Field name for document content
            source_field: Field name for document source
            flatten_nested: Whether to flatten nested JSON structures
        """
        super().__init__(config)
        
        self.document_type = document_type
        self.document_key = document_key
        self.relation_key = relation_key
        self.id_field = id_field
        self.content_field = content_field
        self.source_field = source_field
        self.flatten_nested = flatten_nested
    
    def load(self, source: Union[str, Path]) -> LoaderResult:
        """
        Load documents from a JSON file.
        
        Args:
            source: Path to the JSON file to load documents from
            
        Returns:
            LoaderResult containing loaded documents and relationships
        """
        source_path = Path(source)
        
        if not source_path.exists():
            raise ValueError(f"Source file does not exist: {source}")
        
        if not source_path.is_file():
            raise ValueError(f"Source is not a file: {source}")
        
        logger.info(f"Loading documents from JSON file: {source_path}")
        
        # Load JSON data
        with open(source_path, 'r', encoding=self.config.encoding) as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON file: {e}")
        
        # Extract documents
        documents: List[IngestDocument] = []
        errors: List[Dict[str, Any]] = []
        
        # Get documents from JSON
        if isinstance(data, list):
            # Data is directly a list of documents
            json_docs = data
        elif isinstance(data, dict) and self.document_key in data:
            # Data is an object with a documents array
            json_docs = data[self.document_key]
        else:
            # Try to use the entire object as a single document
            json_docs = [data]
        
        # Process each document
        for i, doc_data in enumerate(json_docs):
            try:
                document = self._process_document(doc_data, source_path, i)
                documents.append(document)
            except Exception as e:
                errors.append({
                    "index": i,
                    "error": str(e),
                    "type": type(e).__name__
                })
                logger.error(f"Error processing document at index {i}: {e}")
        
        # Extract relationships
        relations: List[DocumentRelation] = []
        
        # Get relationships from JSON if specified
        if self.relation_key and isinstance(data, dict) and self.relation_key in data:
            relation_data = data[self.relation_key]
            relations = self._process_relations(relation_data, documents)
        
        # Create dataset
        dataset_name = f"json_{source_path.stem}"
        dataset = self.create_dataset(
            name=dataset_name,
            documents=documents,
            relations=relations,
            description=f"Documents from JSON file: {source_path}",
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
                "document_count": len(documents),
                "error_count": len(errors)
            }
        )
    
    def _process_document(self, doc_data: Dict[str, Any], source_path: Path, index: int) -> IngestDocument:
        """
        Process a single document from JSON data.
        
        Args:
            doc_data: Document data from JSON
            source_path: Path to the source JSON file
            index: Index of the document in the array
            
        Returns:
            Processed IngestDocument
        """
        # Get or generate document ID
        doc_id = str(doc_data.get(self.id_field, str(uuid.uuid4())))
        
        # Get content
        content = doc_data.get(self.content_field, "")
        if not content and self.flatten_nested:
            # Use the entire JSON object as content if no content field
            content = json.dumps(doc_data, indent=2)
        
        # Get source
        source = doc_data.get(self.source_field, f"{source_path.name}[{index}]")
        
        # Extract timestamps if available
        created_at = None
        if "created_at" in doc_data:
            try:
                if isinstance(doc_data["created_at"], str):
                    created_at = datetime.fromisoformat(doc_data["created_at"].replace("Z", "+00:00"))
                elif isinstance(doc_data["created_at"], (int, float)):
                    created_at = datetime.fromtimestamp(doc_data["created_at"])
            except (ValueError, TypeError):
                pass
        
        updated_at = None
        if "updated_at" in doc_data:
            try:
                if isinstance(doc_data["updated_at"], str):
                    updated_at = datetime.fromisoformat(doc_data["updated_at"].replace("Z", "+00:00"))
                elif isinstance(doc_data["updated_at"], (int, float)):
                    updated_at = datetime.fromtimestamp(doc_data["updated_at"])
            except (ValueError, TypeError):
                pass
        
        # Get title
        title = doc_data.get("title", None)
        if not title and "name" in doc_data:
            title = doc_data["name"]
        
        # Extract metadata (exclude known fields)
        exclude_fields = {
            self.id_field, self.content_field, self.source_field, 
            "title", "name", "created_at", "updated_at", "metadata"
        }
        
        metadata = {}
        for key, value in doc_data.items():
            if key not in exclude_fields:
                metadata[key] = value
        
        # Add custom metadata if present
        if "metadata" in doc_data and isinstance(doc_data["metadata"], dict):
            metadata.update(doc_data["metadata"])
        
        # Create document
        return IngestDocument(
            id=doc_id,
            content=content,
            source=source,
            document_type=self.document_type,
            title=title,
            created_at=created_at,
            updated_at=updated_at,
            metadata=metadata
        )
    
    def _process_relations(
        self, 
        relation_data: List[Dict[str, Any]], 
        documents: List[IngestDocument]
    ) -> List[DocumentRelation]:
        """
        Process relationship data from JSON.
        
        Args:
            relation_data: Relationship data from JSON
            documents: List of documents for reference
            
        Returns:
            List of processed DocumentRelation objects
        """
        relations: List[DocumentRelation] = []
        doc_ids = {doc.id for doc in documents}
        
        for i, rel in enumerate(relation_data):
            try:
                # Get required fields
                source_id = rel.get("source_id") or rel.get("from") or rel.get("source")
                target_id = rel.get("target_id") or rel.get("to") or rel.get("target")
                relation_type_str = rel.get("relation_type") or rel.get("type") or "RELATED_TO"
                
                # Validate IDs
                if not source_id or not target_id:
                    logger.warning(f"Skipping relation at index {i}: missing source or target ID")
                    continue
                
                # Skip if documents don't exist in our dataset
                if source_id not in doc_ids or target_id not in doc_ids:
                    logger.warning(f"Skipping relation at index {i}: document not found")
                    continue
                
                # Convert relation type
                try:
                    relation_type = RelationType(relation_type_str.lower())
                except ValueError:
                    relation_type = RelationType.CUSTOM
                
                # Get optional fields
                weight = float(rel.get("weight", 1.0))
                bidirectional = bool(rel.get("bidirectional", False))
                
                # Extract metadata
                metadata = {}
                for key, value in rel.items():
                    if key not in {"source_id", "target_id", "relation_type", "weight", 
                                  "bidirectional", "from", "to", "type", "source", "target"}:
                        metadata[key] = value
                
                # Create relationship
                relation = DocumentRelation(
                    source_id=source_id,
                    target_id=target_id,
                    relation_type=relation_type,
                    weight=weight,
                    bidirectional=bidirectional,
                    metadata=metadata
                )
                relations.append(relation)
                
            except Exception as e:
                logger.warning(f"Error processing relation at index {i}: {e}")
        
        return relations
